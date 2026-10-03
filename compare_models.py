"""Compare already-installed Ollama models on the same synthetic hunt scenarios.

Runs each selected model against each JSONL log file, then writes a timestamped
directory under results/ with report.html, results.jsonl, and manifest.json.
Models are never downloaded; names must already appear in `ollama list`.
Comparisons use harness.run_hunt and its shared three-verdict prompt: suspicious,
benign, or inconclusive.

Declared inference configurations in inference-configurations/*.conf are copied
into that directory before inference. An uncommented num_ctx line is the context
window sent for that model. Uncommented temperature, top_p, and top_k lines are
the sampling options sent for that model. A line that starts with # is kept and
not applied.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from ollama import Client

from shared.harness import parse_timeout, preflight_models_and_logs, run_hunt
from shared.inference_configurations import (
    DEFAULT_INFERENCE_CONFIGURATIONS_DIR, assign_run_slots, canonical_model_name,
    load_inference_configuration_paths, load_inference_configurations,
)
from shared.run_reports import (
    OUTPUT_ROOT, annotate_result, open_recorded_run, save_result, seconds, slot_label,
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
    "logs/shared-vpn-logins.jsonl",
    "logs/managed-telemetry.jsonl",
    "logs/scheduled-discovery.jsonl",
    "logs/opaque-sync-transfers.jsonl",
]


def prepare_comparison(client: Client, models: list[str], logs: list[str]) -> tuple[list[dict], list[dict]]:
    """Resolve the complete matrix before generating anything."""
    return preflight_models_and_logs(client, models, logs)


def run_comparison(
    client: Client,
    models: list[str],
    logs: list[str],
    output_root: Path,
    configurations_dir: Path | None = None,
    configuration_paths: list[Path] | None = None,
) -> Path:
    selected, cases = prepare_comparison(client, models, logs)
    configurations_directory = (
        DEFAULT_INFERENCE_CONFIGURATIONS_DIR
        if configurations_dir is None
        else configurations_dir
    )
    explicit = load_inference_configuration_paths(configuration_paths) if configuration_paths else []
    pinned = {canonical_model_name(configuration["model"]) for configuration in explicit}
    # Resolve inference configurations before creating the results directory or calling the model.
    loaded = []
    if any(canonical_model_name(model["name"]) not in pinned for model in selected):
        loaded = load_inference_configurations(configurations_directory)
    slots = assign_run_slots(selected, loaded, explicit)
    directory, manifest = open_recorded_run(output_root, configurations_directory, slots, cases)
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
                              num_ctx=slot["num_ctx"],
                              think=slot.get("think"),
                              sampling=slot.get("sampling"),
                              keep_alive=0 if index == len(cases)-1 else "5m")
            annotate_result(result, slot, case)
            save_result(directory, manifest, results, result)
            print(
                f'  {result["status"]} · {seconds(result["timing"].get("evaluation_seconds"))}',
                flush=True,
            )
            for warning in result.get("warnings") or []:
                print(f"  {warning}", flush=True)
    return directory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=MODELS, help="Installed Ollama names (default: MODELS in compare_models.py)")
    parser.add_argument("--logs", nargs="+", default=DEFAULT_SCENARIO_LOGS, help="JSONL paths relative to the script, or absolute paths")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT, help="Parent for a new US Eastern Time named results directory")
    parser.add_argument("--inference-configurations-dir", type=Path, default=DEFAULT_INFERENCE_CONFIGURATIONS_DIR, help="Directory of declared *.conf files (default: inference-configurations/ next to this script)")
    parser.add_argument("--inference-configuration", nargs="+", type=Path, default=None, help="Inference configuration file for this run. Repeat to test one model with different settings.")
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
    try:
        directory = run_comparison(
            Client(timeout=args.timeout),
            args.models,
            args.logs,
            args.output_dir,
            args.inference_configurations_dir,
            args.inference_configuration,
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
