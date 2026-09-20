"""Compare installed Ollama models across the same synthetic scenarios."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from html import escape
import json
import os
from pathlib import Path
import sys
from uuid import uuid4

from ollama import Client

from main import load_security_events, resolve_log_path, run_hunt

# Copy exact names from `ollama list`. These are never downloaded automatically.
MODELS = [
    "qwen3:32b",
    "mistral-small3.2:24b",
    "hf.co/fdtn-ai/Foundation-Sec-8B-Instruct-Q8_0-GGUF:latest",
]
TEST_LOGS = [
    "logs/password-spray.jsonl",
    "logs/http-beaconing.jsonl",
    "logs/internal-network-scan.jsonl",
]
OUTPUT_ROOT = Path(__file__).parent / "results"


def prepare_comparison(client: Client, models: list[str], logs: list[str]) -> tuple[list[dict], list[dict]]:
    """Resolve the complete matrix before generating anything."""
    if not models or not logs:
        raise ValueError("Configure at least one model and one log file.")
    installed = {item.model: item for item in client.list().models}
    selected, problems, seen = [], [], set()
    for name in models:
        canonical = name if name in installed else name + ":latest"
        if canonical not in installed:
            problems.append(f"Model is not installed: {name}")
            continue
        if canonical in seen:
            continue
        seen.add(canonical)
        model = installed[canonical]
        try:
            info = client.show(canonical)
            capabilities = info.capabilities or []
            if capabilities and "completion" not in capabilities:
                raise ValueError("model does not support text completion")
            selected.append({"name": canonical, "digest": model.digest,
                             "capabilities": capabilities})
        except Exception as error:
            problems.append(f"Cannot use {name}: {error}")
    cases, seen_paths = [], set()
    for name in logs:
        path = resolve_log_path(name).resolve()
        if path in seen_paths:
            continue
        seen_paths.add(path)
        try:
            events = load_security_events(path)
            if not events:
                raise ValueError("log contains no events")
            cases.append({"name": name, "path": str(path), "events": events,
                          "events_sha256": hashlib.sha256(
                              json.dumps(events, sort_keys=True).encode()
                          ).hexdigest()})
        except Exception as error:
            problems.append(f"Cannot read {name}: {error}")
    if problems:
        raise ValueError("Preflight failed:\n" + "\n".join(problems))
    return selected, cases


def seconds(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}s"


def write_report(directory: Path, manifest: dict, results: list[dict]) -> None:
    """Static, escaped HTML: model responses are displayed only as text."""
    e = lambda value: escape(str(value))
    by_pair = {(r["case"], r["model"]): r for r in results}
    rows, sections = [], []
    for case in manifest["cases"]:
        cards = []
        for model in manifest["models"]:
            result = by_pair.get((case["name"], model["name"]))
            name = e(model["name"])
            if result is None:
                cards.append(f'<article><h3>{name}</h3><p class="pending">Pending</p></article>')
                continue
            status = result["status"]
            verdict = result["sections"].get("Verdict", "—")
            timing = result["timing"]
            wall = seconds(timing.get("wall_seconds"))
            load = seconds(timing.get("load_duration_seconds"))
            rows.append(f'<tr><td>{e(case["name"])}</td><td>{name}</td>'
                        f'<td>{e(verdict)}</td><td class="{e(status)}">{e(status)}</td>'
                        f'<td>{len(result["unknown_evidence_ids"])}</td><td>{wall}</td><td>{load}</td></tr>')
            errors = result["validation_errors"] + ([result["error"]] if result["error"] else [])
            error_html = ''.join(f'<p class="error">{e(error)}</p>' for error in errors)
            thinking = (f'<details><summary>Thinking trace</summary><pre>{e(result["thinking"])}</pre></details>'
                        if result["thinking"] else '')
            think = result["request"].get("think", "unsupported / unavailable")
            cards.append(f'<article><h3>{name}</h3><p class="{e(status)}">{e(status)} · {wall}</p>'
                         f'<p>Thinking: {e(think)} · Load: {load}<br>'
                         f'Prompt processing: {seconds(timing.get("prompt_eval_duration_seconds"))} · '
                         f'Generation: {seconds(timing.get("eval_duration_seconds"))}</p>'
                         f'{error_html}<pre>{e(result["raw_content"]) or "No answer returned."}</pre>{thinking}</article>')
        sections.append(f'<section><h2>{e(case["name"])}</h2><div class="answers">{"".join(cards)}</div></section>')
    total = len(manifest["models"]) * len(manifest["cases"])
    html = '''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="dark">
<title>Threat hunt model comparison</title><style>
*{box-sizing:border-box}html{color-scheme:dark}body{margin:0;padding:32px;font:15px/1.6 system-ui,sans-serif;background:#0f1419;color:#e8eef4}
main{max-width:1800px;margin:auto}h1{font-size:30px;margin-bottom:4px}h2{margin-top:36px;font-size:21px}
h3{font-size:15px;overflow-wrap:anywhere}p{color:#9aa8b8}table{width:100%;border-collapse:collapse;background:#161d26}
th,td{text-align:left;padding:10px 14px;border-bottom:1px solid #2a3544}th{background:#1c2530}
.scroll{overflow:auto}.answers{display:grid;grid-auto-flow:column;grid-auto-columns:minmax(300px,1fr);gap:16px;overflow-x:auto;padding-bottom:12px}
article{padding:20px;border:1px solid #2a3544;border-radius:10px;background:#161d26;min-width:0}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.65 ui-monospace,monospace;color:#d7e0ea}
.ok{color:#3ecf8e}.invalid,.error{color:#f07178}.pending{color:#7d8b99}summary{cursor:pointer}
@media(max-width:600px){body{padding:16px}.answers{grid-auto-flow:row;grid-template-columns:1fr}}
</style></head><body><main>'''
    html += (f'<h1>Threat hunt model comparison</h1><p>{e(manifest["created_at"])} · '
             f'{len(results)} / {total} runs recorded</p><p>Output validity checks format and cited IDs; '
             'it does not establish detection accuracy. Wall time includes model loading. '
             'Runs are sequential, with a fresh conversation for every case.</p>'
             '<div class="scroll"><table><thead><tr><th>Case</th><th>Model</th><th>Verdict</th>'
             '<th>Output status</th><th>Unknown IDs</th><th>Wall time</th><th>Load time</th>'
             f'</tr></thead><tbody>{"".join(rows)}</tbody></table></div>{"".join(sections)}</main></body></html>')
    temp = directory / "report.html.tmp"
    temp.write_text(html, encoding="utf-8")
    temp.replace(directory / "report.html")


def run_comparison(client: Client, models: list[str], logs: list[str], output_root: Path) -> Path:
    selected, cases = prepare_comparison(client, models, logs)
    now = datetime.now(timezone.utc)
    directory = output_root / (now.strftime("%Y%m%dT%H%M%S.%fZ") + "-" + uuid4().hex[:8])
    directory.mkdir(parents=True, exist_ok=False)
    manifest = {"schema_version": 1, "created_at": now.isoformat(), "models": selected,
                "cases": [{k: v for k, v in case.items() if k != "events"} for case in cases]}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    results = []
    write_report(directory, manifest, results)
    print(f"Results: {directory}", flush=True)
    with (directory / "results.jsonl").open("w", encoding="utf-8") as output:
        for model in selected:
            for index, case in enumerate(cases):
                print(f'[{len(results)+1}/{len(selected)*len(cases)}] {model["name"]} · {case["name"]}', flush=True)
                result = run_hunt(model["name"], case["events"], client=client,
                                  capabilities=model["capabilities"],
                                  keep_alive=0 if index == len(cases)-1 else "5m")
                result.update({"schema_version": 1, "case": case["name"], "log_path": case["path"],
                               "events_sha256": case["events_sha256"], "model_digest": model["digest"]})
                output.write(json.dumps(result) + "\n")
                output.flush()
                os.fsync(output.fileno())
                results.append(result)
                write_report(directory, manifest, results)
                print(f'  {result["status"]} · {seconds(result["timing"]["wall_seconds"])}', flush=True)
    return directory


def positive_timeout(value: str) -> float:
    number = float(value)
    if not 0 < number < float("inf"):
        raise argparse.ArgumentTypeError("timeout must be a positive, finite number")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=MODELS, help="Installed Ollama names (default: MODELS in compare.py)")
    parser.add_argument("--logs", nargs="+", default=TEST_LOGS, help="JSONL paths relative to the script, or absolute paths")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT, help="Parent for a new timestamped results directory")
    parser.add_argument("--timeout", type=positive_timeout, default=600, help="HTTP operation timeout in seconds (default: 600)")
    args = parser.parse_args()
    try:
        directory = run_comparison(Client(timeout=args.timeout), args.models, args.logs, args.output_dir)
    except (Exception, KeyboardInterrupt) as error:
        print(f"Comparison stopped: {str(error) or 'interrupted'}. Completed rows and report are retained if a run started.", file=sys.stderr)
        raise SystemExit(1)
    print(f"Report: {directory / 'report.html'}")
    with (directory / "results.jsonl").open(encoding="utf-8") as output:
        if any(json.loads(line)["status"] != "ok" for line in output):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
