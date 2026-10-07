# Threat Hunting Agent Harness

Open-weight models are advancing quickly, but security teams still lack a practical way to compare how well they investigate real threat-hunting patterns. To our knowledge, no public benchmark or evaluation currently focuses on open-weight model performance across both host and network security logs.

Threat Hunting Agent Harness is a local, reproducible test bed for that problem. It runs Ollama models against synthetic ECS event sets, requires a structured evidence-backed finding, validates the response, and produces side-by-side reports for human review. The current scenarios test suspicious activity, convincing benign lookalikes, and cases where the evidence should remain inconclusive.

This is an early proof of concept, and participants are welcome. Security practitioners, model researchers, and engineers can contribute model runs, new scenarios, answer keys, inference configurations, or improvements to the harness. Open an issue to discuss an experiment or submit a pull request with results and changes.

## What it evaluates

Each hunt starts a fresh conversation and asks the model for:

- `Verdict`: `suspicious`, `benign`, or `inconclusive`
- `Threat type`
- `Summary`
- `Evidence`, cited by `event.id`

The harness checks the output contract, cited IDs, completion status, and context use. It does **not** score detection accuracy automatically: compare each answer with [the human grading keys](evals/answer-keys.md). This version runs each model/scenario pair once, so results are observations rather than statistically stable performance estimates.

The seven synthetic scenarios include three suspicious patterns, three contextual benign lookalikes, and one deliberately ambiguous transfer. They are selected ECS excerpts, not full captures or authentic vendor exports. See [logs/README.md](logs/README.md) for field and collection semantics.

## Quick start

Requirements:

- Python 3.14+
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com), running locally
- At least one supported model already installed in `ollama list`

On macOS, set the recommended Ollama server environment before starting or restarting Ollama:

```bash
launchctl setenv OLLAMA_FLASH_ATTENTION 1
launchctl setenv OLLAMA_KV_CACHE_TYPE f16
launchctl setenv OLLAMA_NUM_PARALLEL 1
launchctl setenv OLLAMA_MAX_LOADED_MODELS 1
```

Install dependencies, install a model, and run one case:

```bash
uv sync
ollama run qwen3:32b
uv run python compare_models.py \
  --models qwen3:32b \
  --logs logs/http-beaconing.jsonl
```

Each invocation creates a timestamped results directory. Open its `report.html` to review the verdict, threat type, summary, evidence, context use, timings, and raw answer.

Useful variants:

```bash
# Use an explicit checked-in inference configuration
uv run python compare_models.py \
  --models qwen3:32b \
  --logs logs/http-beaconing.jsonl \
  --inference-configuration inference_config/qwen3-32b.conf

# Set a per-request timeout; the default is unlimited
uv run python compare_models.py \
  --models qwen3:32b \
  --logs logs/http-beaconing.jsonl \
  --timeout 600

# Run the default comparison matrix
uv run python compare_models.py
```

The default matrix compares Qwen3 32B, Mistral Small 3.2 24B, and Foundation-Sec 8B across all seven scenarios (21 runs). Edit `MODELS` and `DEFAULT_SCENARIO_LOGS` in `compare_models.py`, or override them:

```bash
uv run python compare_models.py \
  --models qwen3:32b mistral-small3.2:24b \
  --logs logs/password-spray.jsonl logs/http-beaconing.jsonl
```

Models are checked before inference and never downloaded by the harness. Runs are sequential and grouped by model; every case uses a fresh conversation. One failed inference does not stop the remaining cases.

## Scenarios

| Log                                | What it tests                                             |
| ---------------------------------- | --------------------------------------------------------- |
| `logs/http-beaconing.jsonl`        | Periodic HTTP check-ins among ordinary traffic            |
| `logs/password-spray.jsonl`        | A many-account authentication failure sequence            |
| `logs/internal-network-scan.jsonl` | Internal port probes among ordinary traffic               |
| `logs/shared-vpn-logins.jsonl`     | Shared-VPN logins that resemble spraying                  |
| `logs/managed-telemetry.jsonl`     | Managed check-ins that resemble beaconing                 |
| `logs/scheduled-discovery.jsonl`   | Authorized discovery that resembles an internal scan      |
| `logs/opaque-sync-transfers.jsonl` | Host and network evidence that should remain inconclusive |

Grade each case with [evals/answer-keys.md](evals/answer-keys.md), which defines three checks: interpretation, evidence, and restraint. Keep the answer keys out of the model prompt.

## Models

The suite currently recognizes these Ollama names:

- `command-r:latest`
- `cyberpal2-20b:latest`
- `deepseek-r1:32b`
- `foundation-sec-8b-instruct:latest`
- `gemma4:26b`
- `gemma4:31b`
- `glm-4.7-flash:q4_K_M`
- `gpt-oss:20b`
- `granite4.2:30b`
- `llama3.3:70b`
- `mistral-nemo:12b`
- `mistral-small3.2:24b`
- `mistral-small3.2:latest`
- `phi3:medium-128k`
- `qwen3:32b`

Most are installed directly with `ollama run NAME` or `ollama pull NAME`. For example:

```bash
ollama run deepseek-r1:32b
ollama run gemma4:31b
ollama run glm-4.7-flash:q4_K_M
ollama run gpt-oss:20b
ollama run granite4.2:30b
ollama run llama3.3:70b
ollama run mistral-nemo:12b
ollama run mistral-small3.2:24b
ollama run phi3:medium-128k
ollama run qwen3:32b
```

Three models require local wrappers so Ollama uses the expected chat format.

### Command R

The Ollama library template writes an extra `<|END_OF_TURN_TOKEN|>` before the chatbot turn. Create the suite name with the corrected template:

```bash
ollama create command-r -f modelfiles/Modelfile.command-r
```

### Foundation-Sec 8B

The raw GGUF import has no usable chat template. Pull the weights, then create the suite wrapper:

```bash
ollama pull hf.co/fdtn-ai/Foundation-Sec-8B-Instruct-Q8_0-GGUF
ollama create foundation-sec-8b-instruct \
  -f modelfiles/Modelfile.foundation-sec-8b-instruct
```

Re-run `ollama create` after changing the Modelfile; Ollama reuses the downloaded weights.

### CyberPal 2.0 20B

`cyberpal2-20b:latest` is a local BF16 conversion of [CyberPal2.0-20B](https://huggingface.co/cyber-pal-security/CyberPal2.0-20B) with a Harmony wrapper.

```bash
uv venv --python 3.12
source .venv/bin/activate
git clone https://github.com/ggml-org/llama.cpp.git
uv pip install --index https://pypi.org/simple \
  -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt \
  huggingface_hub
hf download cyber-pal-security/CyberPal2.0-20B \
  --local-dir ./CyberPal2.0-20B
```

If `CyberPal2.0-20B/tokenizer_config.json` sets `tokenizer_class` to `TokenizersBackend`, change it to `PreTrainedTokenizerFast`, then convert:

```bash
python llama.cpp/convert_hf_to_gguf.py \
  ./CyberPal2.0-20B \
  --outfile ./CyberPal2.0-20B-BF16.gguf \
  --outtype bf16
```

Create a base import with this local Modelfile:

```text
FROM ./CyberPal2.0-20B-BF16.gguf
PARAMETER num_ctx 32768
```

Then create the base model and suite wrapper:

```bash
ollama create cyberpal2:20b-bf16 -f Modelfile
ollama create cyberpal2-20b -f modelfiles/Modelfile.cyberpal2-20b
```

The wrapper sets `PARSER harmony` and stops `<|return|>` and `<|call|>`.

## Inference configurations

Each `inference_config/*.conf` file defines one reproducible eval setup. Lines use `key=value`; blank lines and `#` comments are ignored.

```ini
model=qwen3:32b
thinking=true
num_ctx=40960
# Qwen's thinking-mode recommendation
temperature=0.6
top_p=0.95
top_k=20
```

Supported keys are:

- `model`
- `num_ctx`
- `thinking`
- `temperature`
- `top_p`
- `top_k`
- `weight_precision`
- `weight_quant`
- `kv_cache`
- `repeat_penalty`

Uncommented values override Ollama or Modelfile defaults for that request. The runner validates names and values, prints the selected configuration, and copies it into the result directory before inference. If multiple configurations name the same model, select one or more with `--inference-configuration`.

Request defaults in `shared/harness.py` are `num_ctx=32768`, `thinking=false`, `temperature=0`, `seed=0`, `num_predict=-1`, and context shifting disabled. `top_p` and `top_k` are omitted unless configured. Thinking and final output share the context window. The harness sends `think` only to models that advertise the capability.

The checked-in configurations record model-specific context and thinking settings. Optional weight, KV-cache, and repeat-penalty fields document the environment; only request fields are sent to Ollama. The expected server-side KV cache is `f16`.

## Hunt and validation behavior

`compare_models.py` calls `run_hunt` in `shared/harness.py`:

1. Load a JSONL file and sort events by `@timestamp`.
2. Build standard `role`/`content` messages from the shared prompt and serialized events.
3. Ask for the three verdicts and four output fields.
4. Parse and validate the response.
5. Save the result and refresh the HTML report.

Log text is untrusted evidence. The prompt requires citations by `event.id` and prohibits inventing users, IPs, timestamps, or IDs absent from the file. Event values containing `<`, `>`, or `&` are Unicode-escaped so they cannot break the prompt's XML delimiters.

A reply is invalid when a required section is empty, the verdict is unsupported, a cited ID is absent, generation ends because of length, content is empty, or context use reaches/exceeds `num_ctx`. Use at or above 90% of the window produces a warning. Passing validation means only that the answer is structurally valid and grounded to known IDs—not that the diagnosis is correct.

Preflight stops before inference when a configuration is malformed, a model is missing or incompatible, a log is unreadable or empty, or configuration selection is ambiguous.

### Model-specific prompts

Most models use the shared system prompt and user task. Registered exceptions in `shared/prompts.py` adapt the same contract to model requirements:

- DeepSeek-R1 and Phi-3 move instructions into the user message.
- Command R uses Cohere preamble headings.
- CyberPal keeps the system text and adds a step-by-step instruction.

Each model uses one prompt across inference configurations.

## Results

Every run directory contains:

- `report.html`: summary table, full answers, prompt text, thinking traces, settings, timings, and token/context use.
- `results.jsonl`: one durable row per attempted model/case pair, including raw responses, parsed fields, validation errors, request data, hashes, model digest, timings, and token counts.
- `manifest.json`: selected models, capabilities, quantization, context, inputs, and pending/completed cases.
- `declared-inference-configurations/`: snapshots of matching configuration files.

Results are written after each attempt, so completed work survives interruption. Exit status is nonzero if preflight fails, execution is interrupted, a run fails validation, or a run exhausts its context window.

Wall time includes model loading and request overhead. Eval time covers prompt processing plus generation. Thinking/output token splits are exact when only one text is present and estimated by character length when both are present. Tokenizers and native limits can make identical context settings behave differently across models.

## Chat-template compatibility

The harness sends ordinary message text and relies on the installed Ollama template or renderer for native turn markers. Before any chat call, `shared/model_config.py` checks `/api/show` data against `shared/model_formats.toml`.

| Family            | Required framing                                                                          |
| ----------------- | ----------------------------------------------------------------------------------------- |
| Foundation-Sec    | `<\|system\|>`, `<\|user\|>`, `<\|assistant\|>`; no parser; only `<\|end_of_text\|>` stop |
| CyberPal 2.0      | Harmony markers and channel framing; `<\|return\|>` and `<\|call\|>` stops                |
| gpt-oss           | Harmony markers and channel framing; no library stop lines                                |
| Qwen3             | `<\|im_start\|>` and `<\|im_end\|>`                                                       |
| Llama 3           | `<\|start_header_id\|>` and `<\|eot_id\|>`                                                |
| Mistral Small     | `[SYSTEM_PROMPT]`, `[/SYSTEM_PROMPT]`, `[INST]`, `[/INST]`                                |
| Mistral Nemo      | `[INST]`, `[/INST]`, and `.System`                                                        |
| Gemma 4           | `RENDERER gemma4` or `RENDERER gemma4-large`                                              |
| GLM-4.7           | `RENDERER glm-4.7` and `PARSER glm-4.7`                                                   |
| Granite 4         | `<\|im_start\|>` and `<\|im_end\|>`                                                       |
| DeepSeek-R1       | `<｜User｜>` and `<｜Assistant｜>`                                                        |
| Command R         | Cohere turn, system, user, and chatbot tokens; no duplicate end token                     |
| Phi-3 Medium 128K | Role markers; exactly the end, user, and assistant stops                                  |

A missing family registration or mismatched template, renderer, parser, or stop sequence fails preflight. Inspect an installed package with:

```bash
ollama show --modelfile MODEL
```

### Adding a model

1. Add an ordered `[[formats]]` entry to `shared/model_formats.toml` using the checkpoint's published template and generation settings.
2. If the imported model lacks the correct framing, add a wrapper under `modelfiles/`.
3. Add an `inference_config/*.conf` file for the installed name.
4. Run the tests and a single scenario before adding the model to a comparison.

Example registry entry:

```toml
[[formats]]
label = "Example Instruct"
name_pattern = "example-instruct"
markers = ["<user>", "<assistant>"]
parsers = []
required_stops = ["<end>"]
hint = "Create example-instruct using modelfiles/Modelfile.example-instruct."
```

Name matching is case-insensitive, and the first regular-expression match wins. Use `markers` for templates or `renderers` for built-in renderers. The registry validates installed packages; it does not download models or rewrite templates.

## Repository layout

| Path                                 | Purpose                                                       |
| ------------------------------------ | ------------------------------------------------------------- |
| `compare_models.py`                  | Run model/scenario matrices and write reports                 |
| `shared/harness.py`                  | Log loading, prompting, inference, accounting, and validation |
| `shared/prompts.py`                  | Shared and model-specific prompts                             |
| `shared/model_config.py`             | Installed-model format validation                             |
| `shared/model_formats.toml`          | Ordered chat-format requirements                              |
| `shared/inference_configurations.py` | Configuration loading and selection                           |
| `shared/run_reports.py`              | Durable results and HTML reports                              |
| `inference_config/*.conf`            | Reproducible per-model settings                               |
| `modelfiles/`                        | Ollama wrappers for models that need them                     |
| `logs/*.jsonl`                       | Seven synthetic ECS scenarios                                 |
| `logs/README.md`                     | Scenario representation and field notes                       |
| `evals/answer-keys.md`               | Human grading keys; never loaded by the harness               |
| `tests/`                             | Unit tests for runs, reports, configuration, and validation   |

Run the test suite with:

```bash
uv run python -m unittest -v
```
