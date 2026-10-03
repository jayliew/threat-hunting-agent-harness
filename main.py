"""Run one threat hunt using the shared harness and report writer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from ollama import Client

import shared.inference_configurations as inference_configurations
import shared.run_reports as run_reports
from shared.harness import (
    DEFAULT_LOG_FILE, DEFAULT_MODEL, KV_CACHE_TYPE, SEED,
    format_token_report, parse_timeout, preflight_models_and_logs, resolve_log_path, run_hunt,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the threat-hunting harness.")
    parser.add_argument(
        "log_file",
        nargs="?",
        default=DEFAULT_LOG_FILE,
        help=f"JSONL event file relative to this script (default: {DEFAULT_LOG_FILE})",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Ollama model already installed on this machine "
            f"(ollama list). Default: {DEFAULT_MODEL}. "
            "With --inference-configuration, this must match the inference configuration's model= value."
        ),
    )
    parser.add_argument(
        "--inference-configuration",
        type=Path,
        default=None,
        help=(
            "Declared inference configuration file for this run. Use another file on a later "
            "run to test the same model under different settings."
        ),
    )
    parser.add_argument(
        "--inference-configurations-dir",
        type=Path,
        default=None,
        help=(
            "Directory of declared *.conf files "
            "(default: inference-configurations/ next to this script)"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Parent for the results directory (default: results/ next to this script)",
    )
    parser.add_argument(
        "--timeout",
        type=parse_timeout,
        default=None,
        help=(
            "HTTP timeout in seconds for each Ollama request, or none to wait "
            "indefinitely (default: none)"
        ),
    )
    args = parser.parse_args()
    configurations_directory = (
        inference_configurations.DEFAULT_INFERENCE_CONFIGURATIONS_DIR
        if args.inference_configurations_dir is None
        else args.inference_configurations_dir
    )
    output_root = run_reports.OUTPUT_ROOT if args.output_dir is None else args.output_dir
    try:
        if args.inference_configuration is not None:
            explicit = inference_configurations.load_inference_configuration_paths(
                [args.inference_configuration]
            )
            chosen = explicit[0]
            if args.model and not inference_configurations.model_names_match(
                args.model, chosen["model"]
            ):
                raise ValueError(
                    f"Inference configuration {chosen['source']} declares {chosen['model']}, "
                    f"not {args.model}."
                )
            model_name = chosen["model"]
            loaded: list[dict] = []
        else:
            explicit = []
            model_name = args.model or DEFAULT_MODEL
            loaded = inference_configurations.load_inference_configurations(
                configurations_directory
            )
            # Choose the setup before Ollama is contacted. Several matches must be pinned.
            inference_configurations.unique_inference_configuration(loaded, model_name)
    except ValueError as error:
        print(error, file=sys.stderr)
        raise SystemExit(1)
    client = Client(timeout=args.timeout)
    log_path = resolve_log_path(args.log_file)
    try:
        selected, cases = preflight_models_and_logs(
            client, [model_name], [str(log_path)]
        )
        slots = inference_configurations.assign_run_slots(selected, loaded, explicit)
    except ValueError as error:
        print(error, file=sys.stderr)
        raise SystemExit(1)
    slot = slots[0]
    print(f"Model: {slot['name']}")
    renderer = slot.get("renderer") or ""
    if renderer:
        print(f"Chat framing: RENDERER {renderer}")
    print(f"Chat template:\n{slot.get('chat_template') or ''}")
    # resolve_log_path() accepts absolute paths, including files outside this repo.
    # relative_to() raises ValueError for those; print the absolute path instead.
    repo_root = Path(__file__).parent
    displayed_log = (
        log_path.relative_to(repo_root)
        if log_path.is_relative_to(repo_root)
        else log_path
    )
    event_list = cases[0]["events"]
    events = json.dumps(event_list, separators=(",", ":"))
    print(f"Log file: {displayed_log}")
    print(f"Security events supplied:\n{events}")
    directory, manifest = run_reports.open_recorded_run(
        output_root, configurations_directory, slots, cases
    )
    print("\n--- Analysis ---", flush=True)
    result = run_hunt(
        slot["name"],
        event_list,
        client=client,
        capabilities=slot["capabilities"],
        chat_template=slot.get("chat_template"),
        renderer=renderer or None,
        model_max=slot.get("context_length"),
        num_ctx=slot["num_ctx"],
        think=slot.get("think"),
        sampling=slot.get("sampling"),
    )
    run_reports.annotate_result(result, slot, cases[0])
    run_reports.save_result(directory, manifest, [], result)
    print(f"KV cache type (server OLLAMA_KV_CACHE_TYPE): {KV_CACHE_TYPE}")
    print(f"Seed: {SEED}")
    effective_think = result["request"].get("think")
    print(f"Thinking: {effective_think if effective_think is not None else 'unsupported or unavailable'}")
    if result["thinking"]:
        print(f"Thinking:\n{result['thinking']}")
    if result["raw_content"]:
        print(result["raw_content"])
    print(format_token_report(result))
    print(
        f"{result['status']} · {run_reports.seconds(result['timing'].get('evaluation_seconds'))}",
        flush=True,
    )
    for warning in result["warnings"]:
        print(warning, file=sys.stderr)
    errors = result["validation_errors"] + ([result["error"]] if result["error"] else [])
    for error in errors:
        print(error, file=sys.stderr)
    print(f"Report: {directory / 'report.html'}")
    if result["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
