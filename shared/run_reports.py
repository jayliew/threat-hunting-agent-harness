"""Shared run records, inference configuration snapshots, and HTML reports."""
from __future__ import annotations

from datetime import datetime
from html import escape
import json
import math
import os
from pathlib import Path
from zoneinfo import ZoneInfo

from . import REPO_ROOT
from .harness import KV_CACHE_TYPE, NUM_CTX, NUM_PREDICT, SEED, SHIFT, THINK
from .inference_configurations import display_source

OUTPUT_ROOT = REPO_ROOT / "results"
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


def write_declared_inference_configurations(directory: Path, configurations: list[dict]) -> None:
    """Copy the text recorded before the run into the results directory."""
    if not configurations:
        return
    dest = directory / "declared-inference-configurations"
    dest.mkdir()
    used: set[str] = set()
    for configuration in configurations:
        base = Path(configuration["source"]).name
        if not base.endswith(".conf"):
            base += ".conf"
        stem = base[: -len(".conf")]
        name = base
        suffix = 2
        while name in used:
            name = f"{stem}-{suffix}.conf"
            suffix += 1
        used.add(name)
        configuration["copy_name"] = name
        text = configuration["text"]
        if not text.endswith("\n"):
            text += "\n"
        (dest / name).write_text(text, encoding="utf-8")


def slot_label(model: dict) -> str:
    """Distinguish two setups of the same installed model."""
    source = model.get("inference_configuration_source")
    if source:
        return f"{model['name']} ({source})"
    return model["name"]


def declared_inference_configurations_html(manifest: dict) -> str:
    """Show the archived inference configuration that drove each model's request."""
    blocks = []
    for model in manifest.get("models") or []:
        name = escape(slot_label(model))
        configuration = model.get("declared_inference_configuration")
        if not configuration:
            blocks.append(
                f'<article><h3>{name}</h3>'
                f'<p class="pending">No declared inference configuration</p></article>'
            )
            continue
        note = configuration.get("quantization_note")
        note_html = f"<p>{escape(note)}</p>" if note else ""
        blocks.append(
            f"<article><h3>{name}</h3>"
            f"<p>Recorded from {escape(configuration['source'])}</p>"
            f"{note_html}<pre>{escape(configuration['text'].rstrip())}</pre></article>"
        )
    if not blocks:
        return ""
    return (
        "<h2>Declared inference configurations</h2>"
        "<p>Recorded before inference. Uncommented inference configuration values drive the Ollama "
        "request; this block is the archived source file, including comments and "
        "fields that are not chat options.</p>"
        f'<div class="answers">{"".join(blocks)}</div>'
    )


def seconds(value: float | None) -> str:
    """Format a duration stored in seconds as minutes and seconds."""
    if value is None:
        return "—"
    total = abs(float(value))
    minutes = int(total // 60)
    remainder = total - (minutes * 60)
    text = f"{minutes}m {remainder:.2f}s"
    return f"-{text}" if value < 0 else text


def duration_seconds_one_decimal(value: float | None) -> str:
    """Format seconds rounded up to one decimal place (for prompt/generation timing)."""
    if value is None:
        return "—"
    total = float(value)
    sign = "-" if total < 0 else ""
    rounded = math.ceil(abs(total) * 10) / 10
    return f"{sign}{rounded:.1f}s"


def display(value) -> str:
    return "—" if value is None or value == "" else str(value)


def framing_note(model: dict) -> str:
    """Show a built-in renderer instead of a placeholder {{ .Prompt }} template."""
    renderer = model.get("renderer") or ""
    if not renderer:
        return ""
    return f"Framing: RENDERER {escape(renderer)}<br>"


def quantization_label(model: dict) -> str:
    """Quantization and size for the report. Omit file format (for example gguf)."""
    parts = [
        model.get("quantization_level"),
        model.get("parameter_size"),
    ]
    return " · ".join(part for part in parts if part) or "—"


def think_label(value) -> str:
    if value is True:
        return "enabled"
    if value is False or value is None:
        return "disabled"
    return str(value)


def thinking_label(result: dict | None, model: dict, settings: dict) -> str:
    if result is not None:
        if "think" in result.get("request", {}):
            return think_label(result["request"]["think"])
        return "not supported"
    if "thinking" not in (model.get("capabilities") or []):
        return "not supported"
    chosen = model.get("think")
    if chosen is None:
        chosen = settings.get("think")
    return think_label(chosen)


def context_label(allocated, used, model_max) -> str:
    text = f"used {display(used)} / allocated {display(allocated)}"
    if model_max is not None:
        text += f" (model max {model_max})"
    return text


def context_used_cell(used, allocated) -> str:
    """Summary-table cell: used tokens and percent of allocated context."""
    if used is None or used == "":
        return "—"
    if allocated is None or allocated == "" or allocated == 0:
        return str(used)
    percent = round(100 * float(used) / float(allocated))
    return f"{used} ({percent}%)"


def tokens_label(tokens: dict | None, *, include_thinking: bool = True) -> str:
    tokens = tokens or {}
    estimate = " (estimated)" if tokens.get("split") == "estimated" else ""
    parts = [f"Input: {display(tokens.get('input_tokens'))}"]
    if include_thinking:
        parts.append(f"Thinking: {display(tokens.get('thinking_tokens'))}{estimate}")
    parts.append(f"Output: {display(tokens.get('output_tokens'))}{estimate}")
    return " · ".join(parts)


def collapsible_run_details(inner_html: str) -> str:
    """Wrap run metadata; collapsed until the reader opens it."""
    return (
        '<details class="run-details">'
        "<summary>Show run details</summary>"
        f'<div class="run-details-body">{inner_html}</div>'
        "</details>"
    )


def format_num_predict(value: object) -> str:
    """Label the sent num_predict. Ollama treats -1 as unlimited generation."""
    if value == -1:
        return "-1 (no limit)"
    return str(value)


def write_report(directory: Path, manifest: dict, results: list[dict]) -> None:
    """Static, escaped HTML: model responses are displayed only as text."""
    e = lambda value: escape(str(value))
    settings = manifest.get("request_settings") or {}
    by_pair = {(r["case"], r.get("run_key", r["model"])): r for r in results}
    rows, sections = [], []
    for case in manifest["cases"]:
        cards = []
        for model in manifest["models"]:
            result = by_pair.get((case["name"], model.get("run_key", model["name"])))
            name = e(slot_label(model))
            think = thinking_label(result, model, settings)
            quant = quantization_label(model)
            show_thinking_tokens = think != "not supported"
            if result is None:
                pending_details = (
                    f'{e(context_label(model.get("num_ctx", settings.get("num_ctx")), None, model.get("context_length")))}<br>'
                    f'{framing_note(model)}'
                    f'Quantization: {e(quant)}<br>Thinking: {e(think)}<br>'
                    f'{e(tokens_label(None, include_thinking=show_thinking_tokens))}'
                )
                cards.append(
                    f'<article><h3>{name}</h3><p class="pending">Pending</p>'
                    f'{collapsible_run_details(pending_details)}</article>'
                )
                continue
            status = result["status"]
            verdict = result["sections"].get("Verdict", "—")
            timing = result["timing"]
            evaluated = seconds(timing.get("evaluation_seconds"))
            tokens = result.get("tokens") or {}
            context = result.get("context") or {}
            allocated = context.get("allocated", model.get("num_ctx", settings.get("num_ctx")))
            used = context.get("used")
            model_max = context.get("model_max", model.get("context_length"))
            thinking_tokens_cell = (
                display(tokens.get("thinking_tokens")) if show_thinking_tokens else "—"
            )
            rows.append(
                f'<tr><td>{e(case["name"])}</td><td>{name}</td>'
                f'<td>{e(quant)}</td>'
                f'<td>{e(verdict)}</td><td class="{e(status)}">{e(status)}</td>'
                f'<td>{len(result["unknown_evidence_ids"])}</td><td>{evaluated}</td>'
                f'<td>{e(think)}</td>'
                f'<td>{e(context_used_cell(used, allocated))}</td><td>{e(display(allocated))}</td>'
                f'<td>{e(display(tokens.get("input_tokens")))}</td>'
                f'<td>{e(thinking_tokens_cell)}</td>'
                f'<td>{e(display(tokens.get("output_tokens")))}</td></tr>'
            )
            errors = result["validation_errors"] + ([result["error"]] if result["error"] else [])
            error_html = ''.join(f'<p class="error">{e(error)}</p>' for error in errors)
            warning_html = ''.join(
                f'<p class="warn">{e(warning)}</p>' for warning in result.get("warnings") or []
            )
            thinking = (f'<details><summary>Thinking trace</summary><pre>{e(result["thinking"])}</pre></details>'
                        if result["thinking"] else '')
            run_details = (
                f'{e(context_label(allocated, used, model_max))}<br>'
                f'{framing_note(model)}'
                f'Quantization: {e(quant)}<br>'
                f'Thinking: {e(think)}<br>'
                f'{e(tokens_label(tokens, include_thinking=show_thinking_tokens))}<br>'
                f'Prompt processing: {duration_seconds_one_decimal(timing.get("prompt_eval_duration_seconds"))} · '
                f'Generation: {duration_seconds_one_decimal(timing.get("eval_duration_seconds"))}'
            )
            cards.append(
                f'<article><h3>{name}</h3><p class="{e(status)}">{e(status)} · {evaluated}</p>'
                f'{collapsible_run_details(run_details)}'
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
.run-details{margin:8px 0 12px}.run-details summary{display:inline-block;padding:6px 12px;border:1px solid #2e3a48;border-radius:6px;background:#222c38;color:#e7edf3;font-size:14px;list-style:none}
.run-details summary::-webkit-details-marker{display:none}.run-details-body{margin-top:8px;color:#9aa8b5;line-height:1.6}
@media(max-width:600px){body{padding:16px}.answers{grid-auto-flow:row;grid-template-columns:1fr}}
</style></head><body><main>'''
    html += (f'<h1>Threat hunt model comparison</h1><p>{e(manifest["created_at"])} · '
             f'{len(results)} / {total} runs recorded</p>'
             f'<p>Request settings: num_ctx={e(settings.get("num_ctx"))}, '
             f'num_predict={e(format_num_predict(settings.get("num_predict")))}, '
             f'seed={e(settings.get("seed"))}, '
             f'think={e(settings.get("think"))}, '
             f'kv_cache_type={e(settings.get("kv_cache_type"))} '
             '(server env <code>OLLAMA_KV_CACHE_TYPE</code>; not a chat API option). '
             'These are the harness fallbacks unless an inference configuration overrides them; '
             'they are not necessarily what each model was sent. '
             'Each model card shows the context window and thinking value actually used.</p>'
             '<p>Output validity checks format and cited IDs; '
             'it does not establish detection accuracy. Eval time is prompt processing plus generation for that log and excludes model load and unload. '
             'Wall time includes model loading. '
             'Runs are sequential, with a fresh conversation for every case. '
             'Context used is prompt tokens plus generated tokens, compared with the configured num_ctx. '
             'Thinking and output tokens split that generated count: exact when only one of those texts is present, '
             'estimated by character length when both are present. '
             'A run warns at 90% of num_ctx and is an error at or above num_ctx. '
             'Context shift is disabled for every model.</p>'
             f'{declared_inference_configurations_html(manifest)}'
             '<div class="scroll"><table><thead><tr><th>Case</th><th>Model</th><th>Quantization</th>'
             '<th>Verdict</th><th>Output status</th><th>Unknown IDs</th><th>Eval time</th><th>Thinking</th>'
             '<th>Context used</th><th>Context allocated</th>'
             '<th>Input tokens</th><th>Thinking tokens</th><th>Output tokens</th>'
             f'</tr></thead><tbody>{"".join(rows)}</tbody></table></div>{"".join(sections)}</main></body></html>')
    temp = directory / "report.html.tmp"
    temp.write_text(html, encoding="utf-8")
    temp.replace(directory / "report.html")


def open_recorded_run(
    output_root: Path,
    configurations_directory: Path,
    slots: list[dict],
    cases: list[dict],
) -> tuple[Path, dict]:
    """Create the results directory and write inference configurations before any inference."""
    now = eastern_now()
    directory = allocate_results_directory(output_root, now)
    attached = [
        slot["declared_inference_configuration"]
        for slot in slots
        if slot.get("declared_inference_configuration")
    ]
    write_declared_inference_configurations(directory, attached)
    manifest = {
        "schema_version": 3,
        "created_at": now.isoformat(),
        "inference_config_dir": (
            display_source(configurations_directory)
            if configurations_directory.exists()
            else str(configurations_directory)
        ),
        "inference_configurations": attached,
        "request_settings": {
            "num_ctx": NUM_CTX,
            "num_predict": NUM_PREDICT,
            "seed": SEED,
            "think": THINK,
            "shift": SHIFT,
            "kv_cache_type": KV_CACHE_TYPE,
        },
        "models": slots,
        "cases": [{k: v for k, v in case.items() if k != "events"} for case in cases],
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    write_report(directory, manifest, [])
    print(f"Results: {directory}", flush=True)
    return directory, manifest


def annotate_result(result: dict, slot: dict, case: dict) -> dict:
    result.update({
        "schema_version": 3,
        "case": case["name"],
        "log_path": case["path"],
        "events_sha256": case["events_sha256"],
        "model_digest": slot["digest"],
        "run_key": slot["run_key"],
        "declared_inference_configuration": slot.get("inference_configuration_source"),
    })
    return result


def save_result(directory: Path, manifest: dict, results: list[dict], result: dict) -> None:
    """Append one hunt and refresh the report. The inference configuration is already on disk."""
    with (directory / "results.jsonl").open("a", encoding="utf-8") as output:
        output.write(json.dumps(result) + "\n")
        output.flush()
        os.fsync(output.fileno())
    results.append(result)
    write_report(directory, manifest, results)
