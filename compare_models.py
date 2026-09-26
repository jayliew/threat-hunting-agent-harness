"""Compare already-installed Ollama models on the same synthetic hunt scenarios.

Runs each selected model against each JSONL log file, then writes a timestamped
directory under results/ with report.html, results.jsonl, and manifest.json.
Models are never downloaded; names must already appear in `ollama list`.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from html import escape
import json
import os
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

from ollama import Client

import main as hunt_harness
from main import (
    NUM_CTX,
    NUM_PREDICT,
    THINK,
    generation_cap_label,
    preflight_models_and_logs,
    resolve_generation_settings,
    run_hunt,
)

# Copy exact names from `ollama list`. These are never downloaded automatically.
MODELS = [
    "qwen3:32b",
    "mistral-small3.2:24b",
    "foundation-sec-8b-instruct",
]
DEFAULT_SCENARIO_LOGS = [
    "logs/password-spray.jsonl",
    "logs/http-beaconing.jsonl",
    "logs/internal-network-scan.jsonl",
    "logs/shared-vpn-logins.ecs.jsonl",
    "logs/managed-telemetry.ecs.jsonl",
    "logs/scheduled-discovery.ecs.jsonl",
]
OUTPUT_ROOT = Path(__file__).parent / "results"
EASTERN = ZoneInfo("America/New_York")
WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
MONTHS = (
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
)


def results_directory_name(when: datetime) -> str:
    """Readable US Eastern folder name: dd-Mon-yyyy, weekday, and hh-mm am/pm."""
    eastern = when.astimezone(EASTERN)
    hour12 = eastern.hour % 12 or 12
    meridiem = "am" if eastern.hour < 12 else "pm"
    return (
        f"{eastern.day:02d}-{MONTHS[eastern.month - 1]}-{eastern.year}-"
        f"{WEEKDAYS[eastern.weekday()]}_"
        f"{hour12:02d}-{eastern.minute:02d}{meridiem}-ET"
    )


def eastern_now() -> datetime:
    return datetime.now(EASTERN)


def allocate_results_directory(output_root: Path, when: datetime) -> Path:
    base = results_directory_name(when)
    directory = output_root / base
    suffix = 2
    while directory.exists():
        directory = output_root / f"{base}-{suffix}"
        suffix += 1
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def prepare_comparison(client: Client, models: list[str], logs: list[str]) -> tuple[list[dict], list[dict]]:
    """Resolve the complete matrix before generating anything."""
    return preflight_models_and_logs(client, models, logs)


def seconds(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}s"


def display(value) -> str:
    return "—" if value is None or value == "" else str(value)


def framing_note(model: dict) -> str:
    """Show a built-in renderer instead of a placeholder {{ .Prompt }} template."""
    renderer = model.get("renderer") or ""
    if not renderer:
        return ""
    return f"Framing: RENDERER {escape(renderer)}<br>"


def quantization_label(model: dict) -> str:
    parts = [
        model.get("quantization_level"),
        model.get("parameter_size"),
        model.get("format"),
    ]
    return " · ".join(part for part in parts if part) or "—"


def thinking_label(result: dict | None, model: dict, settings: dict) -> str:
    if result is not None:
        if "think" in result.get("request", {}):
            return "enabled" if result["request"]["think"] else "disabled"
        return "not supported"
    if "thinking" not in (model.get("capabilities") or []):
        return "not supported"
    return "enabled" if settings.get("think") else "disabled"


def context_label(allocated, used, model_max) -> str:
    text = f"used {display(used)} / allocated {display(allocated)}"
    if model_max is not None:
        text += f" (model max {model_max})"
    return text


def tokens_label(tokens: dict | None) -> str:
    tokens = tokens or {}
    estimate = " (estimated)" if tokens.get("split") == "estimated" else ""
    return (
        f"Input: {display(tokens.get('input_tokens'))} · "
        f"Thinking: {display(tokens.get('thinking_tokens'))}{estimate} · "
        f"Output: {display(tokens.get('output_tokens'))}{estimate} · "
        f"Cached: {display(tokens.get('prompt_eval_cached_count'))} · "
        f"Uncached: {display(tokens.get('prompt_uncached_count'))}"
    )


def model_heading(model: dict) -> str:
    profile = model.get("profile")
    if profile:
        return f'{model["name"]} ({profile})'
    return model["name"]


def planned_generation(model: dict, case: dict, result: dict | None, settings: dict) -> dict:
    """Settings for a card: the recorded hunt, or the match for a pending one."""
    if result and result.get("generation"):
        return result["generation"]
    options = ((result or {}).get("request") or {}).get("options") or {}
    if "num_ctx" in options and "num_predict" in options:
        return {"num_ctx": options["num_ctx"], "num_predict": options["num_predict"]}
    if result is None:
        return resolve_generation_settings(
            model["name"], model.get("profile"), case["name"]
        )
    return {
        "num_ctx": settings.get("num_ctx", NUM_CTX),
        "num_predict": settings.get("num_predict", NUM_PREDICT),
    }


def write_report(directory: Path, manifest: dict, results: list[dict]) -> None:
    """Static, escaped HTML: model responses are displayed only as text."""
    e = lambda value: escape(str(value))
    settings = manifest.get("request_settings") or {}
    by_pair = {
        (r["case"], r["model"], r.get("profile")): r for r in results
    }
    rows, sections = [], []
    for case in manifest["cases"]:
        cards = []
        for model in manifest["models"]:
            result = by_pair.get((case["name"], model["name"], model.get("profile")))
            name = e(model_heading(model))
            think = thinking_label(result, model, settings)
            quant = quantization_label(model)
            planned = planned_generation(model, case, result, settings)
            cap = generation_cap_label(planned["num_predict"])
            if result is None:
                cards.append(
                    f'<article><h3>{name}</h3><p class="pending">Pending</p>'
                    f'<p>{e(context_label(planned["num_ctx"], None, model.get("context_length")))}<br>'
                    f'Generation cap: {e(cap)}<br>'
                    f'{framing_note(model)}'
                    f'Quantization: {e(quant)}<br>Thinking: {e(think)}<br>'
                    f'{e(tokens_label(None))}</p></article>'
                )
                continue
            status = result["status"]
            verdict = result["sections"].get("Verdict", "—")
            timing = result["timing"]
            wall = seconds(timing.get("wall_seconds"))
            load = seconds(timing.get("load_duration_seconds"))
            tokens = result.get("tokens") or {}
            context = result.get("context") or {}
            allocated = context.get("allocated", settings.get("num_ctx"))
            used = context.get("used")
            model_max = context.get("model_max", model.get("context_length"))
            rows.append(
                f'<tr><td>{e(case["name"])}</td><td>{name}</td>'
                f'<td>{e(verdict)}</td><td class="{e(status)}">{e(status)}</td>'
                f'<td>{len(result["unknown_evidence_ids"])}</td><td>{wall}</td><td>{load}</td>'
                f'<td>{e(quant)}</td><td>{e(think)}</td>'
                f'<td>{e(display(used))}</td><td>{e(display(allocated))}</td>'
                f'<td>{e(display(tokens.get("input_tokens")))}</td>'
                f'<td>{e(display(tokens.get("thinking_tokens")))}</td>'
                f'<td>{e(display(tokens.get("output_tokens")))}</td>'
                f'<td>{e(display(tokens.get("prompt_eval_cached_count")))}</td></tr>'
            )
            errors = result["validation_errors"] + ([result["error"]] if result["error"] else [])
            error_html = ''.join(f'<p class="error">{e(error)}</p>' for error in errors)
            warning_html = ''.join(
                f'<p class="warn">{e(warning)}</p>' for warning in result.get("warnings") or []
            )
            thinking = (f'<details><summary>Thinking trace</summary><pre>{e(result["thinking"])}</pre></details>'
                        if result["thinking"] else '')
            cards.append(
                f'<article><h3>{name}</h3><p class="{e(status)}">{e(status)} · {wall}</p>'
                f'<p>{e(context_label(allocated, used, model_max))}<br>'
                f'Generation cap: {e(cap)}<br>'
                f'{framing_note(model)}'
                f'Quantization: {e(quant)}<br>'
                f'Thinking: {e(think)} · Load: {load}<br>'
                f'{e(tokens_label(tokens))}<br>'
                f'Prompt processing: {seconds(timing.get("prompt_eval_duration_seconds"))} · '
                f'Generation: {seconds(timing.get("eval_duration_seconds"))}</p>'
                f'{warning_html}{error_html}<pre>{e(result["raw_content"]) or "No answer returned."}</pre>{thinking}</article>'
            )
        sections.append(f'<section><h2>{e(case["name"])}</h2><div class="answers">{"".join(cards)}</div></section>')
    total = len(manifest["models"]) * len(manifest["cases"])
    html = '''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Threat hunt model comparison</title><style>
html{color-scheme:dark}*{box-sizing:border-box}body{margin:0;padding:32px;font:15px/1.6 system-ui,sans-serif;background:#0f1419;color:#e7edf3}
main{max-width:1800px;margin:auto}h1{font-size:30px;margin-bottom:4px}h2{margin-top:36px;font-size:21px}
h3{font-size:15px;overflow-wrap:anywhere}p{color:#9aa8b5}table{width:100%;border-collapse:collapse;background:#1a222c}
th,td{text-align:left;padding:10px 14px;border-bottom:1px solid #2e3a48}th{background:#222c38}
.scroll{overflow:auto}.answers{display:grid;grid-auto-flow:column;grid-auto-columns:minmax(300px,1fr);gap:16px;overflow-x:auto;padding-bottom:12px}
article{padding:20px;border:1px solid #2e3a48;border-radius:10px;background:#1a222c;min-width:0}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.65 ui-monospace,monospace}
.ok{color:#3dd68c}.invalid,.error{color:#f07178}.warn{color:#e6b450}.pending{color:#8b9aab}summary{cursor:pointer}
@media(max-width:600px){body{padding:16px}.answers{grid-auto-flow:row;grid-template-columns:1fr}}
</style></head><body><main>'''
    html += (f'<h1>Threat hunt model comparison</h1><p>{e(manifest["created_at"])} · '
             f'{len(results)} / {total} runs recorded</p><p>Output validity checks format and cited IDs; '
             'it does not establish detection accuracy. Wall time includes model loading. '
             'Runs are sequential, with a fresh conversation for every case. '
             'Each run uses the num_ctx and num_predict matched to its model, profile, and log. '
             'num_predict disabled means the generation cap is off. '
             'Context used is prompt tokens plus generated tokens, compared with that run\'s num_ctx. '
             'Thinking and output tokens split that generated count: exact when only one of those texts is present, '
             'estimated by character length when both are present. '
             'A run warns at 90% of num_ctx and is an error at or above num_ctx. '
             'A missing cached-token count is unknown, not zero.</p>'
             '<div class="scroll"><table><thead><tr><th>Case</th><th>Model</th><th>Verdict</th>'
             '<th>Output status</th><th>Unknown IDs</th><th>Wall time</th><th>Load time</th>'
             '<th>Quantization</th><th>Thinking</th><th>Context used</th><th>Context allocated</th>'
             '<th>Input tokens</th><th>Thinking tokens</th><th>Output tokens</th><th>Cached tokens</th>'
             f'</tr></thead><tbody>{"".join(rows)}</tbody></table></div>{"".join(sections)}</main></body></html>')
    temp = directory / "report.html.tmp"
    temp.write_text(html, encoding="utf-8")
    temp.replace(directory / "report.html")


def run_comparison(client: Client, models: list[str], logs: list[str], output_root: Path) -> Path:
    selected, cases = prepare_comparison(client, models, logs)
    now = eastern_now()
    directory = allocate_results_directory(output_root, now)
    manifest = {
        "schema_version": 1,
        "created_at": now.isoformat(),
        "request_settings": {
            "num_ctx": NUM_CTX,
            "num_predict": NUM_PREDICT,
            "think": THINK,
            "generation_settings": hunt_harness.GENERATION_SETTINGS,
        },
        "models": selected,
        "cases": [{k: v for k, v in case.items() if k != "events"} for case in cases],
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    results = []
    write_report(directory, manifest, results)
    print(f"Results: {directory}", flush=True)
    with (directory / "results.jsonl").open("w", encoding="utf-8") as output:
        for model in selected:
            for index, case in enumerate(cases):
                label = model_heading(model)
                print(f'[{len(results)+1}/{len(selected)*len(cases)}] {label} · {case["name"]}', flush=True)
                result = run_hunt(model["name"], case["events"], client=client,
                                  capabilities=model["capabilities"],
                                  chat_template=model.get("chat_template"),
                                  renderer=model.get("renderer") or None,
                                  model_max=model.get("context_length"),
                                  log_file=case["name"],
                                  profile=model.get("profile"),
                                  keep_alive=0 if index == len(cases)-1 else "5m")
                result.update({"schema_version": 1, "case": case["name"], "log_path": case["path"],
                               "events_sha256": case["events_sha256"], "model_digest": model["digest"],
                               "profile": model.get("profile")})
                output.write(json.dumps(result) + "\n")
                output.flush()
                os.fsync(output.fileno())
                results.append(result)
                write_report(directory, manifest, results)
                print(f'  {result["status"]} · {seconds(result["timing"]["wall_seconds"])}', flush=True)
                for warning in result.get("warnings") or []:
                    print(f"  {warning}", flush=True)
    return directory


def positive_timeout(value: str) -> float:
    number = float(value)
    if not 0 < number < float("inf"):
        raise argparse.ArgumentTypeError("timeout must be a positive, finite number")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        default=MODELS,
        help="Installed Ollama names, optionally name@profile (default: MODELS in compare_models.py)",
    )
    parser.add_argument("--logs", nargs="+", default=DEFAULT_SCENARIO_LOGS, help="JSONL paths relative to the script, or absolute paths")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT, help="Parent for a new US Eastern Time named results directory")
    parser.add_argument("--timeout", type=positive_timeout, default=300, help="HTTP operation timeout in seconds (default: 300)")
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
