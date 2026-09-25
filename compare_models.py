"""Compare already-installed Ollama models on the same synthetic hunt scenarios.

Runs each selected model against each JSONL log file, then writes a timestamped
directory under results/ with report.html, results.jsonl, and manifest.json.
Models are never downloaded; names must already appear in `ollama list`.

Declared model profiles in profiles/*.profile are copied into that directory
before inference. They record the intended setup; they do not change the request.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
from html import escape
import json
import os
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

from ollama import Client

from main import (
    NUM_CTX,
    NUM_PREDICT,
    THINK,
    chat_template_error,
    installed_renderer,
    load_security_events,
    model_detail_fields,
    native_context_length,
    resolve_log_path,
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
DEFAULT_PROFILES_DIR = Path(__file__).parent / "profiles"
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
            template = info.template or ""
            renderer = installed_renderer(getattr(info, "modelfile", None))
            error = chat_template_error(canonical, template, renderer)
            if error:
                raise ValueError(error)
            selected.append({
                "name": canonical,
                "digest": model.digest,
                "capabilities": capabilities,
                "chat_template": template,
                "renderer": renderer or "",
                "context_length": native_context_length(info),
                **model_detail_fields(info, model),
            })
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


def canonical_model_name(name: str) -> str:
    """Treat an omitted tag and :latest as the same installed model."""
    if name.endswith(":latest"):
        return name[: -len(":latest")]
    return name


def display_source(path: Path) -> str:
    """Prefer a repo-relative path so reports stay portable."""
    repo_root = Path(__file__).parent
    resolved = path.resolve()
    if resolved.is_relative_to(repo_root):
        return str(resolved.relative_to(repo_root))
    return str(resolved)


def normalize_field_name(label: str) -> str:
    parts = label.strip().lower().replace("-", " ").replace("_", " ").split()
    if not parts:
        raise ValueError("blank field name")
    return "_".join(parts)


def parse_profile_text(text: str, source: str) -> dict:
    """Parse a declared profile. Each line is 'Label: value'."""
    if not text.strip():
        raise ValueError(f"{source}: profile is empty")
    fields: dict[str, str] = {}
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip() or line.strip().startswith("#"):
            continue
        if ":" not in line:
            raise ValueError(f"{source}:{number}: expected 'Label: value'")
        label, value = line.split(":", 1)
        label = label.strip()
        value = value.strip()
        if not label or not value:
            raise ValueError(f"{source}:{number}: expected 'Label: value'")
        key = normalize_field_name(label)
        if key in fields:
            raise ValueError(f"{source}:{number}: duplicate field {label}")
        fields[key] = value
    if "model" not in fields:
        raise ValueError(f"{source}: missing Model")
    return {"model": fields["model"], "source": source, "text": text, "fields": fields}


def parse_profile_file(path: Path) -> dict:
    return parse_profile_text(path.read_text(encoding="utf-8"), display_source(path))


def load_profiles(directory: Path) -> list[dict]:
    """Read every *.profile file. A missing directory means nothing was recorded.

    Several files may name the same model. Those are different setups, not duplicates.
    """
    if not directory.exists():
        return []
    if not directory.is_dir():
        raise ValueError(f"Profiles path is not a directory: {directory}")
    problems: list[str] = []
    found: list[dict] = []
    for path in sorted(directory.glob("*.profile")):
        try:
            found.append(parse_profile_file(path))
        except ValueError as error:
            problems.append(str(error))
    if problems:
        raise ValueError("Profile check failed:\n" + "\n".join(problems))
    return found


def load_profile_paths(paths: list[Path]) -> list[dict]:
    """Parse the profile files chosen for this run. The same model may appear more than once."""
    problems: list[str] = []
    found: list[dict] = []
    seen: set[str] = set()
    for path in paths:
        try:
            if not path.is_file():
                raise ValueError(f"Profile not found: {path}")
            profile = parse_profile_file(path)
        except ValueError as error:
            problems.append(str(error))
            continue
        resolved = str(path.resolve())
        if resolved in seen:
            problems.append(f"Profile listed more than once: {profile['source']}")
            continue
        seen.add(resolved)
        found.append(profile)
    if problems:
        raise ValueError("Profile check failed:\n" + "\n".join(problems))
    return found


def model_names_match(left: str, right: str) -> bool:
    return canonical_model_name(left) == canonical_model_name(right)


def profiles_for_model(profiles: list[dict], model_name: str) -> list[dict]:
    return [profile for profile in profiles if model_names_match(profile["model"], model_name)]


def unique_profile(profiles: list[dict], model_name: str) -> dict | None:
    """Return the only profile for a model. Several matches must be chosen with --profile."""
    matches = profiles_for_model(profiles, model_name)
    if len(matches) > 1:
        listed = "\n".join(profile["source"] for profile in matches)
        raise ValueError(
            f"Multiple profiles match {model_name}:\n{listed}\n"
            "Pass --profile with the file for this run."
        )
    return matches[0] if matches else None


def weight_tokens(value: str) -> set[str]:
    """Split a weight-precision note into comparable quant tokens."""
    tokens: set[str] = set()
    current: list[str] = []
    for char in value:
        if char.isalnum() or char == "_":
            current.append(char.lower())
        elif current:
            tokens.add("".join(current))
            current = []
    if current:
        tokens.add("".join(current))
    return tokens


def declared_weight(fields: dict) -> str | None:
    return fields.get("weight_precision") or fields.get("weight_quant")


def quantization_note(declared: str | None, installed: str | None) -> str | None:
    """Warn when the written quant does not match the installed Ollama model."""
    if not declared or not installed:
        return None
    if installed.lower() in weight_tokens(declared):
        return None
    return (
        f"Declared weight quant is {declared}; installed quantization is {installed}."
    )


def recorded_profile(profile: dict, installed: dict) -> dict:
    """Snapshot the profile chosen for one installed model before inference."""
    recorded = {
        "model": profile["model"],
        "matched_model": installed["name"],
        "source": profile["source"],
        "text": profile["text"],
        "fields": profile["fields"],
        "run_key": profile["source"],
    }
    note = quantization_note(
        declared_weight(profile["fields"]),
        installed.get("quantization_level"),
    )
    if note:
        recorded["quantization_note"] = note
    return recorded


def print_recorded_profile(recorded: dict | None, model_name: str) -> None:
    if recorded is None:
        print(f"No declared profile for {model_name}", flush=True)
        return
    print(f"Declared profile ({recorded['source']}):", flush=True)
    print(recorded["text"].rstrip(), flush=True)
    note = recorded.get("quantization_note")
    if note:
        print(note, flush=True)


def assign_run_slots(
    selected: list[dict], loaded: list[dict], explicit: list[dict]
) -> list[dict]:
    """One slot per setup. Two profiles for one model stay separate runs."""
    problems: list[str] = []
    selected_keys = {canonical_model_name(model["name"]) for model in selected}
    for profile in explicit:
        if canonical_model_name(profile["model"]) not in selected_keys:
            problems.append(
                f"Profile {profile['source']} declares {profile['model']}, which is not selected"
            )
    slots = []
    for model in selected:
        pinned = profiles_for_model(explicit, model["name"]) if explicit else []
        if pinned:
            chosen: list[dict | None] = list(pinned)
        else:
            try:
                one = unique_profile(loaded, model["name"])
            except ValueError as error:
                problems.append(str(error))
                continue
            chosen = [one]
        for profile in chosen:
            slot = dict(model)
            if profile is None:
                slot["run_key"] = model["name"]
                slot["profile_source"] = None
                slot["declared_profile"] = None
                print_recorded_profile(None, model["name"])
            else:
                recorded = recorded_profile(profile, model)
                slot["run_key"] = recorded["run_key"]
                slot["profile_source"] = recorded["source"]
                slot["declared_profile"] = recorded
                print_recorded_profile(recorded, model["name"])
            slots.append(slot)
    if problems:
        raise ValueError("Profile check failed:\n" + "\n".join(problems))
    return slots


def write_declared_profiles(directory: Path, profiles: list[dict]) -> None:
    """Copy the text recorded before the run into the results directory."""
    if not profiles:
        return
    dest = directory / "declared-profiles"
    dest.mkdir()
    used: set[str] = set()
    for profile in profiles:
        base = Path(profile["source"]).name
        if not base.endswith(".profile"):
            base += ".profile"
        stem = base[: -len(".profile")]
        name = base
        suffix = 2
        while name in used:
            name = f"{stem}-{suffix}.profile"
            suffix += 1
        used.add(name)
        profile["copy_name"] = name
        text = profile["text"]
        if not text.endswith("\n"):
            text += "\n"
        (dest / name).write_text(text, encoding="utf-8")


def slot_label(model: dict) -> str:
    """Distinguish two setups of the same installed model."""
    source = model.get("profile_source")
    if source:
        return f"{model['name']} ({source})"
    return model["name"]


def declared_profiles_html(manifest: dict) -> str:
    """Show the profile written down before inference, separate from the request."""
    blocks = []
    for model in manifest.get("models") or []:
        name = escape(slot_label(model))
        profile = model.get("declared_profile")
        if not profile:
            blocks.append(
                f'<article><h3>{name}</h3>'
                f'<p class="pending">No declared profile</p></article>'
            )
            continue
        note = profile.get("quantization_note")
        note_html = f"<p>{escape(note)}</p>" if note else ""
        blocks.append(
            f"<article><h3>{name}</h3>"
            f"<p>Recorded from {escape(profile['source'])}</p>"
            f"{note_html}<pre>{escape(profile['text'].rstrip())}</pre></article>"
        )
    if not blocks:
        return ""
    return (
        "<h2>Declared profiles</h2>"
        "<p>Recorded before inference. These notes are the setup you wrote down. "
        "They are not the request sent to Ollama.</p>"
        f'<div class="answers">{"".join(blocks)}</div>'
    )


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


def tokens_label(tokens: dict | None, thinking_enabled: bool) -> str:
    tokens = tokens or {}
    output_note = " (includes thinking)" if thinking_enabled else ""
    return (
        f"Input: {display(tokens.get('prompt_eval_count'))} · "
        f"Cached: {display(tokens.get('prompt_eval_cached_count'))} · "
        f"Uncached: {display(tokens.get('prompt_uncached_count'))} · "
        f"Output: {display(tokens.get('eval_count'))}{output_note}"
    )


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
            if result is None:
                cards.append(
                    f'<article><h3>{name}</h3><p class="pending">Pending</p>'
                    f'<p>{e(context_label(settings.get("num_ctx"), None, model.get("context_length")))}<br>'
                    f'{framing_note(model)}'
                    f'Quantization: {e(quant)}<br>Thinking: {e(think)}<br>'
                    f'{e(tokens_label(None, False))}</p></article>'
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
            thinking_enabled = think == "enabled"
            rows.append(
                f'<tr><td>{e(case["name"])}</td><td>{name}</td>'
                f'<td>{e(verdict)}</td><td class="{e(status)}">{e(status)}</td>'
                f'<td>{len(result["unknown_evidence_ids"])}</td><td>{wall}</td><td>{load}</td>'
                f'<td>{e(quant)}</td><td>{e(think)}</td>'
                f'<td>{e(display(used))}</td><td>{e(display(allocated))}</td>'
                f'<td>{e(display(tokens.get("prompt_eval_count")))}</td>'
                f'<td>{e(display(tokens.get("prompt_eval_cached_count")))}</td>'
                f'<td>{e(display(tokens.get("eval_count")))}</td></tr>'
            )
            errors = result["validation_errors"] + ([result["error"]] if result["error"] else [])
            error_html = ''.join(f'<p class="error">{e(error)}</p>' for error in errors)
            thinking = (f'<details><summary>Thinking trace</summary><pre>{e(result["thinking"])}</pre></details>'
                        if result["thinking"] else '')
            cards.append(
                f'<article><h3>{name}</h3><p class="{e(status)}">{e(status)} · {wall}</p>'
                f'<p>{e(context_label(allocated, used, model_max))}<br>'
                f'{framing_note(model)}'
                f'Quantization: {e(quant)}<br>'
                f'Thinking: {e(think)} · Load: {load}<br>'
                f'{e(tokens_label(tokens, thinking_enabled))}<br>'
                f'Prompt processing: {seconds(timing.get("prompt_eval_duration_seconds"))} · '
                f'Generation: {seconds(timing.get("eval_duration_seconds"))}</p>'
                f'{error_html}<pre>{e(result["raw_content"]) or "No answer returned."}</pre>{thinking}</article>'
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
.ok{color:#3dd68c}.invalid,.error{color:#f07178}.pending{color:#8b9aab}summary{cursor:pointer}
@media(max-width:600px){body{padding:16px}.answers{grid-auto-flow:row;grid-template-columns:1fr}}
</style></head><body><main>'''
    html += (f'<h1>Threat hunt model comparison</h1><p>{e(manifest["created_at"])} · '
             f'{len(results)} / {total} runs recorded</p><p>Output validity checks format and cited IDs; '
             'it does not establish detection accuracy. Wall time includes model loading. '
             'Runs are sequential, with a fresh conversation for every case. '
             'Context used is prompt tokens plus generated tokens. Output tokens include thinking when thinking is enabled. '
             'A missing cached-token count is unknown, not zero.</p>'
             f'{declared_profiles_html(manifest)}'
             '<div class="scroll"><table><thead><tr><th>Case</th><th>Model</th><th>Verdict</th>'
             '<th>Output status</th><th>Unknown IDs</th><th>Wall time</th><th>Load time</th>'
             '<th>Quantization</th><th>Thinking</th><th>Context used</th><th>Context allocated</th>'
             '<th>Input tokens</th><th>Cached tokens</th><th>Output tokens</th>'
             f'</tr></thead><tbody>{"".join(rows)}</tbody></table></div>{"".join(sections)}</main></body></html>')
    temp = directory / "report.html.tmp"
    temp.write_text(html, encoding="utf-8")
    temp.replace(directory / "report.html")


def open_recorded_run(
    output_root: Path,
    profiles_directory: Path,
    slots: list[dict],
    cases: list[dict],
) -> tuple[Path, dict]:
    """Create the results directory and write profiles before any inference."""
    now = eastern_now()
    directory = allocate_results_directory(output_root, now)
    attached = [slot["declared_profile"] for slot in slots if slot.get("declared_profile")]
    write_declared_profiles(directory, attached)
    manifest = {
        "schema_version": 1,
        "created_at": now.isoformat(),
        "profiles_dir": (
            display_source(profiles_directory)
            if profiles_directory.exists()
            else str(profiles_directory)
        ),
        "profiles": attached,
        "request_settings": {
            "num_ctx": NUM_CTX,
            "num_predict": NUM_PREDICT,
            "think": THINK,
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
        "schema_version": 1,
        "case": case["name"],
        "log_path": case["path"],
        "events_sha256": case["events_sha256"],
        "model_digest": slot["digest"],
        "run_key": slot["run_key"],
        "declared_profile": slot.get("profile_source"),
    })
    return result


def save_result(directory: Path, manifest: dict, results: list[dict], result: dict) -> None:
    """Append one hunt and refresh the report. The profile is already on disk."""
    with (directory / "results.jsonl").open("a", encoding="utf-8") as output:
        output.write(json.dumps(result) + "\n")
        output.flush()
        os.fsync(output.fileno())
    results.append(result)
    write_report(directory, manifest, results)


def run_comparison(
    client: Client,
    models: list[str],
    logs: list[str],
    output_root: Path,
    profiles_dir: Path | None = None,
    profile_paths: list[Path] | None = None,
) -> Path:
    selected, cases = prepare_comparison(client, models, logs)
    profiles_directory = DEFAULT_PROFILES_DIR if profiles_dir is None else profiles_dir
    explicit = load_profile_paths(profile_paths) if profile_paths else []
    pinned = {canonical_model_name(profile["model"]) for profile in explicit}
    # Resolve profiles before creating the results directory or calling the model.
    loaded = []
    if any(canonical_model_name(model["name"]) not in pinned for model in selected):
        loaded = load_profiles(profiles_directory)
    slots = assign_run_slots(selected, loaded, explicit)
    directory, manifest = open_recorded_run(output_root, profiles_directory, slots, cases)
    results: list[dict] = []
    total = len(slots) * len(cases)
    for slot in slots:
        for index, case in enumerate(cases):
            print(
                f'[{len(results)+1}/{total}] {slot_label(slot)} · {case["name"]}',
                flush=True,
            )
            result = run_hunt(slot["name"], case["events"], client=client,
                              capabilities=slot["capabilities"],
                              chat_template=slot.get("chat_template"),
                              renderer=slot.get("renderer") or None,
                              model_max=slot.get("context_length"),
                              keep_alive=0 if index == len(cases)-1 else "5m")
            annotate_result(result, slot, case)
            save_result(directory, manifest, results, result)
            print(f'  {result["status"]} · {seconds(result["timing"]["wall_seconds"])}', flush=True)
    return directory


def positive_timeout(value: str) -> float:
    number = float(value)
    if not 0 < number < float("inf"):
        raise argparse.ArgumentTypeError("timeout must be a positive, finite number")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=MODELS, help="Installed Ollama names (default: MODELS in compare_models.py)")
    parser.add_argument("--logs", nargs="+", default=DEFAULT_SCENARIO_LOGS, help="JSONL paths relative to the script, or absolute paths")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT, help="Parent for a new US Eastern Time named results directory")
    parser.add_argument("--profiles-dir", type=Path, default=DEFAULT_PROFILES_DIR, help="Directory of declared *.profile files (default: profiles/ next to this script)")
    parser.add_argument("--profile", nargs="+", type=Path, default=None, help="Profile file for this run. Repeat to test one model with different settings.")
    parser.add_argument("--timeout", type=positive_timeout, default=300, help="HTTP operation timeout in seconds (default: 300)")
    args = parser.parse_args()
    try:
        directory = run_comparison(
            Client(timeout=args.timeout),
            args.models,
            args.logs,
            args.output_dir,
            args.profiles_dir,
            args.profile,
        )
    except (Exception, KeyboardInterrupt) as error:
        print(f"Comparison stopped: {str(error) or 'interrupted'}. Completed rows and report are retained if a run started.", file=sys.stderr)
        raise SystemExit(1)
    print(f"Report: {directory / 'report.html'}")
    with (directory / "results.jsonl").open(encoding="utf-8") as output:
        if any(json.loads(line)["status"] != "ok" for line in output):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
