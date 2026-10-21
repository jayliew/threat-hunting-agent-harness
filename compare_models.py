"""Compare installed Ollama models and running Colibri models on the same hunts.

Runs each selected model against each JSONL log file, then writes a timestamped
directory under results/ with report.html, results.jsonl, and manifest.json.
Ollama models are never downloaded; names must already appear in `ollama list`.
A `colibri/<model-id>` name is served by an already-running `coli serve`.
The harness does not start Colibri. One coli serve process serves one model.
Comparisons use harness.run_hunt and a three-verdict prompt: suspicious, benign,
or inconclusive. Every model uses the shared prompt in shared/prompts.py unless
its canonical name is registered in MODEL_PROMPTS. A custom prompt still has to
request those three verdicts and the four output fields.

A single model and a single log file use this same command. Pass one name to
--models and one path to --logs.

Declared inference configurations in inference_config/*.conf are copied
into that directory before inference. An uncommented num_ctx line is the context
window recorded for that model. Uncommented temperature, top_p, and top_k lines
are the sampling options sent for an Ollama model. Colibri chat accepts
temperature and top_p; top_k is rejected. A line that starts with # is kept and
not applied. provider=colibri selects Colibri for a bare model name, and
base_url selects that model's coli serve when it is not the run default.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from ollama import Client

from shared.colibri import (
    DEFAULT_COLIBRI_BASE_URL, ColibriClient, parse_model_spec,
    resolve_colibri_api_key, resolve_colibri_base_url,
)
from shared.harness import (
    inspect_logs, inspect_ollama_models, parse_timeout, preflight_failure,
    preflight_models_and_logs, run_hunt,
)
from shared.prompts import ModelPrompt, prompt_for_model, prompt_instructions
from shared.inference_configurations import (
    DEFAULT_INFERENCE_CONFIG_DIR, assign_run_slots, canonical_model_name,
    inference_configuration_base_url, load_inference_configuration_paths,
    load_inference_configurations, model_names_match, planned_configurations,
    resolve_provider, validate_backend_configuration,
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


def _attach_system_prompts(slots: list[dict]) -> list[ModelPrompt]:
    """Record each slot's prompt name and instruction text, in slot order.

    The name and the instruction text are strings because slots are written to
    manifest.json. The prompt object itself is not stored on the slot. The text
    omits the security-event log payload appended at request time.
    """
    hunt_prompts = []
    for slot in slots:
        hunt_prompt = prompt_for_model(slot["name"])
        slot["system_prompt"] = hunt_prompt.name
        slot["prompt_text"] = prompt_instructions(hunt_prompt)
        hunt_prompts.append(hunt_prompt)
    return hunt_prompts


def prepare_comparison(client: Client, models: list[str], logs: list[str]) -> tuple[list[dict], list[dict]]:
    """Resolve an Ollama-only matrix before generating anything."""
    return preflight_models_and_logs(client, models, logs)


def _load_run_configurations(
    names: list[str],
    configurations_directory: Path,
    configuration_paths: list[Path] | None,
) -> tuple[list[dict], list[dict]]:
    explicit = load_inference_configuration_paths(configuration_paths) if configuration_paths else []
    pinned = {canonical_model_name(configuration["model"]) for configuration in explicit}
    loaded = []
    if any(canonical_model_name(name) not in pinned for name in names):
        loaded = load_inference_configurations(configurations_directory)
    return loaded, explicit


def _resolve_specs(
    specs: list[dict],
    loaded: list[dict],
    explicit: list[dict],
    default_base_url,
) -> list[dict]:
    """Provider and Colibri base URLs for each CLI model, before any server call.

    default_base_url is called only when a Colibri model has no base_url of its own,
    so an Ollama-only run does not read COLIBRI_BASE_URL.
    """
    problems = []
    resolved = []
    for spec in specs:
        try:
            chosen = planned_configurations(spec["name"], loaded, explicit)
            providers = []
            base_urls = []
            for configuration in chosen:
                provider = resolve_provider(spec["provider"], configuration)
                validate_backend_configuration(configuration, provider)
                providers.append(provider)
                if provider == "colibri":
                    base_urls.append(
                        inference_configuration_base_url(configuration) or default_base_url()
                    )
            if len(set(providers)) != 1:
                raise ValueError(
                    f"Inference configurations for {spec['name']} disagree on provider"
                )
            resolved.append({
                "name": spec["name"],
                "provider": providers[0],
                "base_urls": base_urls,
            })
        except ValueError as error:
            problems.append(str(error))
    if problems:
        raise ValueError("Inference configuration check failed:\n" + "\n".join(problems))
    return resolved


def _colibri_model(name: str, base_url: str) -> dict:
    return {
        "name": name,
        "provider": "colibri",
        "digest": None,
        "capabilities": ["completion", "thinking"],
        "chat_template": "",
        "renderer": "",
        "context_length": None,
        "quantization_level": None,
        "parameter_size": None,
        "format": None,
        "family": None,
        "base_url": base_url,
    }


def _preflight_colibri(
    resolved: list[dict],
    *,
    api_key: str | None,
    timeout: float | None,
    transport,
) -> tuple[dict[str, ColibriClient], list[str]]:
    """Require each base URL to be serving exactly the one requested model id."""
    by_url: dict[str, list[str]] = {}
    for item in resolved:
        if item["provider"] != "colibri":
            continue
        for url in item["base_urls"]:
            ids = by_url.setdefault(url, [])
            if item["name"] not in ids:
                ids.append(item["name"])
    problems = []
    clients: dict[str, ColibriClient] = {}
    for url, ids in by_url.items():
        if len(ids) > 1:
            problems.append(
                f"Colibri at {url} was asked for {', '.join(ids)}. "
                "One coli serve process serves one model. "
                "Point each model at its own base_url."
            )
            continue
        client = ColibriClient(url, api_key=api_key, timeout=timeout, transport=transport)
        clients[url] = client
        try:
            served = client.list_models()
        except Exception as error:
            problems.append(f"Cannot use {ids[0]}: {error}")
            continue
        model_id = ids[0]
        if model_id not in served:
            shown = ", ".join(served) or "(none)"
            problems.append(
                f"Colibri at {url} is not serving {model_id!r} (serving {shown})."
            )
    return clients, problems


def _selected_models(
    resolved: list[dict], ollama_selected: list[dict], default_base_url: str
) -> list[dict]:
    selected = []
    seen = set()
    for item in resolved:
        if item["provider"] == "ollama":
            key = ("ollama", canonical_model_name(item["name"]))
            if key in seen:
                continue
            match = next(
                (
                    model for model in ollama_selected
                    if model_names_match(model["name"], item["name"])
                ),
                None,
            )
            if match is None:
                continue
            seen.add(key)
            selected.append(match)
            continue
        key = ("colibri", item["name"])
        if key in seen:
            continue
        seen.add(key)
        selected.append(_colibri_model(item["name"], default_base_url))
    return selected


def run_comparison(
    client: Client,
    models: list[str],
    logs: list[str],
    output_root: Path,
    configurations_dir: Path | None = None,
    configuration_paths: list[Path] | None = None,
    *,
    colibri_base_url: str | None = None,
    colibri_api_key: str | None = None,
    colibri_timeout: float | None = None,
    colibri_transport=None,
) -> Path:
    if not models or not logs:
        raise ValueError("Configure at least one model and one log file.")
    configurations_directory = (
        DEFAULT_INFERENCE_CONFIG_DIR
        if configurations_dir is None
        else configurations_dir
    )
    specs = [parse_model_spec(name) for name in models]
    api_key = resolve_colibri_api_key(colibri_api_key)
    cached_base_url: list[str] = []

    def default_base_url() -> str:
        if not cached_base_url:
            cached_base_url.append(resolve_colibri_base_url(colibri_base_url))
        return cached_base_url[0]

    loaded, explicit = _load_run_configurations(
        [spec["name"] for spec in specs],
        configurations_directory,
        configuration_paths,
    )
    resolved = _resolve_specs(specs, loaded, explicit, default_base_url)
    fallback_base_url = cached_base_url[0] if cached_base_url else DEFAULT_COLIBRI_BASE_URL
    ollama_names = [item["name"] for item in resolved if item["provider"] == "ollama"]
    ollama_selected, model_problems = (
        inspect_ollama_models(client, ollama_names) if ollama_names else ([], [])
    )
    _colibri_clients, colibri_problems = _preflight_colibri(
        resolved,
        api_key=api_key,
        timeout=colibri_timeout,
        transport=colibri_transport,
    )
    cases, log_problems = inspect_logs(logs)
    preflight_failure(model_problems + colibri_problems + log_problems)
    selected = _selected_models(resolved, ollama_selected, fallback_base_url)
    slots = assign_run_slots(selected, loaded, explicit)
    hunt_prompts = _attach_system_prompts(slots)
    directory, manifest = open_recorded_run(output_root, configurations_directory, slots, cases)
    results: list[dict] = []
    total = len(slots) * len(cases)
    for slot, hunt_prompt in zip(slots, hunt_prompts, strict=True):
        for index, case in enumerate(cases):
            print(
                f'[{len(results)+1}/{total}] {slot_label(slot)} · {case["name"]}',
                flush=True,
            )
            result = run_hunt(
                slot["name"], case["events"], client=client,
                capabilities=slot["capabilities"],
                chat_template=slot.get("chat_template"),
                renderer=slot.get("renderer") or None,
                model_max=slot.get("context_length"),
                num_ctx=slot["num_ctx"],
                think=slot.get("think"),
                sampling=slot.get("sampling"),
                prompt=hunt_prompt,
                keep_alive=0 if index == len(cases)-1 else "5m",
                provider=slot.get("provider") or "ollama",
                colibri_client=_colibri_clients.get(slot.get("base_url")),
                base_url=slot.get("base_url"),
            )
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
    parser.add_argument(
        "--models", nargs="+", default=MODELS,
        help=(
            "Ollama names, or colibri/<model-id> for a running coli serve "
            "(default: MODELS in compare_models.py)"
        ),
    )
    parser.add_argument("--logs", nargs="+", default=DEFAULT_SCENARIO_LOGS, help="JSONL paths relative to the script, or absolute paths")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT, help="Parent for a new US Eastern Time named results directory")
    parser.add_argument("--inference-config-dir", type=Path, default=DEFAULT_INFERENCE_CONFIG_DIR, help="Directory of declared *.conf files (default: inference_config/ next to this script)")
    parser.add_argument("--inference-configuration", nargs="+", type=Path, default=None, help="Inference configuration file for this run. Repeat to test one model with different settings.")
    parser.add_argument(
        "--timeout",
        type=parse_timeout,
        default=None,
        help=(
            "HTTP timeout in seconds for each model request, or none to wait "
            "indefinitely (default: none)"
        ),
    )
    parser.add_argument(
        "--colibri-base-url",
        default=None,
        help=(
            "Default coli serve base URL for Colibri models "
            f"(default: COLIBRI_BASE_URL or {DEFAULT_COLIBRI_BASE_URL})"
        ),
    )
    parser.add_argument(
        "--colibri-api-key",
        default=None,
        help="Bearer token for coli serve. Falls back to COLIBRI_API_KEY. Not written to results.",
    )
    args = parser.parse_args()
    try:
        directory = run_comparison(
            Client(timeout=args.timeout),
            args.models,
            args.logs,
            args.output_dir,
            args.inference_config_dir,
            args.inference_configuration,
            colibri_base_url=args.colibri_base_url,
            colibri_api_key=args.colibri_api_key,
            colibri_timeout=args.timeout,
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
