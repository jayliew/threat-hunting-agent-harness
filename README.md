# Threat Hunting Agent Harness

A bare-bones agentic harness for threat hunting on logs with open weight models. A proof-of-concept.

Give a local model some security events, set the inference config, and print a structured hunt result.

## TL;DR

You can download the open-weight models this suite already runs, or another open-weight model you want to try, and run them locally on these hunt scenarios. The report shows how that model's answers compare with the others. It would be *amazing* if you can share your results back with the community by opening an issue or a pull request on this repo.

I will help guide you through setup and a first run. Open an issue and say what you want to try.

This project needs cybersecurity professionals to evaluate and judge the models' responses. Contributions to the evaluation logs (`logs/`) and their answer keys ([evals/answer-keys.md](evals/answer-keys.md)) are welcome. I welcome all feedback to improve on any aspects of this—which I am sure there are many.

## Models

These Ollama names are the models currently in the suite:

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

## Run a hunt

You need Python 3.14+, [uv](https://docs.astral.sh/uv/), and [Ollama](https://ollama.com) running locally. Install weights yourself. The harness only calls a model already in `ollama list`.

On macOS, set the Ollama server environment before you start Ollama, or restart it afterward. These values enable flash attention, use an f16 KV cache, allow one parallel request, and keep one model loaded. They last until logout or reboot. Quit and reopen the Ollama app, or restart `ollama serve`, so the running server inherits them:

```bash
launchctl setenv OLLAMA_FLASH_ATTENTION 1
launchctl setenv OLLAMA_KV_CACHE_TYPE f16
launchctl setenv OLLAMA_NUM_PARALLEL 1
launchctl setenv OLLAMA_MAX_LOADED_MODELS 1
```

Install Ollama (`brew install ollama`, then `ollama serve` or the Ollama app) and see what is already installed:

```bash
ollama list
```

A single-file run below uses `foundation-sec-8b-instruct`. Pull the weights, then create that name from [modelfiles/Modelfile.foundation-sec-8b-instruct](modelfiles/Modelfile.foundation-sec-8b-instruct). The pull installs `hf.co/fdtn-ai/Foundation-Sec-8B-Instruct-Q8_0-GGUF:latest`, whose template is bare `{{ .Prompt }}` with no `RENDERER`. Preflight refuses that import. The `ollama create` step is what sends `<|system|>`, `<|user|>`, and `<|assistant|>`. Re-run it after a Modelfile change to refresh an existing wrapper; it reuses the downloaded weights:

```bash
ollama pull hf.co/fdtn-ai/Foundation-Sec-8B-Instruct-Q8_0-GGUF
ollama create foundation-sec-8b-instruct -f modelfiles/Modelfile.foundation-sec-8b-instruct
```

Another installed name works when preflight recognizes that family's chat template. Pass it with `--models`.

Foundation-Sec uses plain-text output and no Ollama `PARSER` directive. Its only configured stop must be `<|end_of_text|>`. `compare_models.py` checks that before inference. The wrapper appends that EOS token to completed assistant turns. The harness then parses the returned text into the four answer fields. That application parser is separate from Ollama's output parser.

From this repo, install dependencies and run hunts with `compare_models.py`. For a single model and a single log file, pass one `--models` name and one `--logs` path. This example is `logs/http-beaconing.jsonl` on `foundation-sec-8b-instruct`:

```bash
uv sync
uv run python compare_models.py --models foundation-sec-8b-instruct --logs logs/http-beaconing.jsonl
uv run python compare_models.py --models qwen3:32b --logs logs/http-beaconing.jsonl
uv run python compare_models.py --models qwen3:32b --logs logs/http-beaconing.jsonl --inference-configuration inference_config/qwen3-32b.conf
uv run python compare_models.py --models qwen3:32b --logs logs/http-beaconing.jsonl --timeout 600
```

`--timeout` is a positive number of seconds for one Ollama HTTP request. `--timeout none` is the same unlimited wait as the default. The limit applies to a single request.

The runner copies the inference configuration into a results directory before inference, then appends the hunt result. Open `report.html` in that directory for the verdict, threat type, summary, and evidence.

## What a hunt does

`compare_models.py` calls `run_hunt` from `shared/harness.py`. Each call starts a fresh conversation with the shared prompt and serialized events. A model uses a different prompt only when its canonical name is registered in `MODEL_PROMPTS` in `shared/prompts.py`. `deepseek-r1:32b` is registered: its instructions sit in the user message and its system text is empty, because that series reads the user turn and a system prompt makes it skip its thinking pattern. `phi3:medium-128k` is registered the same way, because Microsoft's Phi-3 family does not support a system message. `command-r` uses Cohere's preamble headings, `## Task and Context` then `## Style Guide`. `cyberpal2-20b` keeps the shared system text and adds "think step-by-step" to the user task, as its model card asks for harder questions. A custom prompt may replace the system text and the user-task sentence. It still has to ask for the same three verdicts and four output fields, because validation does not change per model. The same model uses one prompt across inference configurations.

1. Load a JSONL file and sort events by `@timestamp`.
2. Send standard `role` / `content` messages. The system message holds the analyst instructions, evidence rules, verdict definitions, and output contract. The user message holds the task and a JSON array of events. Markdown headings and lists are the instruction hierarchy. XML tags (`<verdicts>`, `<output_fields>`, `<task>`, `<security_events>`) separate metadata from supporting content.
3. Treat log text as untrusted evidence. Cite `event.id`. Leave users, IPs, timestamps, and event IDs that are absent from the file out of the answer.
4. Print `Verdict`, `Threat type`, `Summary`, and `Evidence`. Those four fields are this harness's output contract.
5. Machine-check the reply.

A reply fails when:

- a required section is missing or empty
- the verdict is outside `suspicious`, `benign`, and `inconclusive`
- a cited evidence ID is absent from the supplied events
- generation stops with `done_reason=length`, or the content is empty
- context used (prompt tokens plus generated tokens) is at or above `num_ctx`

Usage at or above 90% of `num_ctx`, while still under that window, prints a warning. Context shift is off for every model, including DeepSeek2: `SHIFT` in `shared/harness.py` is `False`, and every chat sends that value. Ollama 0.33 would otherwise slide older tokens out once `num_ctx` is full. A format-valid answer can still be the wrong diagnosis.

`<`, `>`, and `&` in event fields are written as JSON Unicode escapes, so a value containing `</security_events>` stays inside the evidence block and keeps its decoded value. The delimiters keep the block intact. Existing completion, output-format, and evidence-ID checks still apply. Prompt structure is tested offline; judge response quality per installed model and template.

The run stops before it can look like a successful hunt when:

- a file in `inference_config/` is empty, missing `model`, is not `key=value` lines, or uses a key that is not an exact inference configuration field name
- the model is missing from `ollama list`, lacks text completion, or its installed template, renderer, parser, or stops fail the format check (see [Chat templates](#chat-templates))
- the log file is missing, unreadable, malformed, or contains no events
- several inference configurations name the model and you omitted `--inference-configuration`

## Scenarios

Grade every case with the [answer keys](evals/answer-keys.md). Keep those keys out of the model prompt. Field notes for the lookalike packages are in [logs/README.md](logs/README.md).

| Log | Contents |
| --- | --- |
| `logs/http-beaconing.jsonl` | Periodic HTTP check-ins among ordinary traffic. The single-file example in [Run a hunt](#run-a-hunt) |
| `logs/password-spray.jsonl` | Authentication events with a many-account failure sequence |
| `logs/internal-network-scan.jsonl` | Internal port probes among ordinary traffic |
| `logs/shared-vpn-logins.jsonl` | Shared-VPN logins that resemble spraying |
| `logs/managed-telemetry.jsonl` | Managed check-ins that resemble beaconing |
| `logs/scheduled-discovery.jsonl` | Scheduled discovery that resembles an internal scan |
| `logs/opaque-sync-transfers.jsonl` | Host and network evidence for an inconclusive transfer |

## Inference configurations and request settings

Each file in `inference_config/` is one eval setup: the installed Ollama name, and the request values for that run. An uncommented value is sent on the chat request and overrides the Ollama or Modelfile default for that call. A `#` line is a note and is not sent. Blank lines are ignored.

Lines are `key=value`. The key must be exactly `model`, `num_ctx`, `thinking`, `temperature`, `top_p`, `top_k`, `weight_precision`, `weight_quant`, `kv_cache`, or `repeat_penalty`. A different case, hyphen, or space (`Num Ctx`, `num-ctx`) stops the run. `alpha` and `alpha:latest` are the same model. Two files may name the same model when the settings differ; those are different runs. Pass `--inference-configuration` with the file you want. If more than one file matches and you omit `--inference-configuration`, the run stops and lists them.

`inference_config/qwen3-32b.conf`:

```
model=qwen3:32b
thinking=true
num_ctx=40960
# basis: Qwen's explicit thinking-mode recommendation (https://huggingface.co/Qwen/Qwen3-32B)
temperature=0.6
top_p=0.95
top_k=20
```

Other recorded setups: `llama3.3:70b` (`num_ctx` 16384), `granite4.2:30b` (65536, `thinking=high`), `deepseek-r1:32b` (65536, `thinking=true`), `command-r:latest` (131072), `gemma4:31b` (32768, `thinking=true`), `gemma4:26b` (32768, `thinking=true`), `glm-4.7-flash:q4_K_M` (202752, `thinking=true`), `mistral-small3.2:24b` (131072), `mistral-nemo:12b` (131072), `phi3:medium-128k` (131072), `foundation-sec-8b-instruct` (131072), and `cyberpal2-20b:latest` (8192, `thinking=medium`). Models that do not support thinking omit that line. Granite 4.2 accepts `false`, `low`, `medium`, and `high`. CyberPal 2.0 uses `thinking=medium`. Qwen3, DeepSeek-R1, Gemma 4, and GLM-4.7-Flash are on or off (`thinking=true` or `false`).

`compare_models.py` loads the chosen inference configuration before the first inference call, prints it, and copies it into the results directory. A model with no file is reported as having no declared inference configuration. If an uncommented weight quant differs from the installed Ollama quantization, the recorded snapshot keeps both and the run prints the difference. A malformed inference configuration stops the run before any results directory is created. `num_ctx` must be one positive integer written as digits only (commas are rejected). `temperature` is a non-negative decimal, `top_p` is a decimal from 0 through 1, and `top_k` is a positive integer.

| Inference configuration key | Request field | If the line is absent |
| --- | --- | --- |
| `num_ctx` | `options.num_ctx` | `32768` from `NUM_CTX` in `shared/harness.py` |
| `thinking` | `think`: `true`, `false`, `low`, `medium`, or `high` | `THINK` in `shared/harness.py` (`False`) |
| `temperature` | `options.temperature` | `0` from `TEMPERATURE` |
| `top_p`, `top_k` | matching sampling options | omitted |
| — | `options.seed` | always `0` (`SEED`). An omitted seed becomes `-1`, and a negative seed picks a new seed each run |
| — | `options.num_predict` | always `-1` (no output-token limit) |
| — | `shift` | always off (`SHIFT = False`) |

The harness asks Ollama (`/api/show`) and sends `think` only when the model lists the `thinking` capability. Models without that capability reject the argument. Qwen3-class models think by default when the API omits `think`, so those models always receive an explicit value. An inference configuration that sets `thinking` for a model without the capability stops the run before any chat call. Set `THINK` in `shared/harness.py` to `True` or `False` before a run whose inference configuration has no `thinking` line. Thinking tokens and the final answer share `num_ctx`.

Weight precision, KV cache, and repeat penalty are optional and absent from the checked-in files. The expected KV cache type is `f16` (`OLLAMA_KV_CACHE_TYPE` on the Ollama server). It is recorded on each hunt and is not sent as a chat option. The saved inference configuration is what distinguishes two runs of the same model when those settings later differ.

## Compare models

`compare_models.py` is also the command for one model and one log file: pass a single `--models` name and a single `--logs` path, as in [Run a hunt](#run-a-hunt). Edit `MODELS` and `DEFAULT_SCENARIO_LOGS` at the top of `compare_models.py`, or override them on the command line. The defaults compare Qwen3 32B, Mistral Small 3.2 24B, and Foundation-Sec 8B across all seven scenarios: 21 runs. Use names from `ollama list`. A name without a tag resolves to `:latest` when that installed name exists. Models are checked before inference and are never downloaded. Missing models, unrecognized names, and wrong templates are reported together.

```bash
uv run python compare_models.py
uv run python compare_models.py --models qwen3:32b mistral-small3.2:24b --logs logs/password-spray.jsonl logs/http-beaconing.jsonl
uv run python compare_models.py --models mistral-small3.2:24b --timeout 600 --output-dir results
uv run python compare_models.py --inference-config-dir inference_config
uv run python compare_models.py --models qwen3:32b --inference-configuration inference_config/qwen3-32b.conf inference_config/qwen3-32b-other.conf
```

Log paths are relative to the script, or absolute. `--output-dir` and `--inference-config-dir` are relative to the current working directory. The default directory is `inference_config/` next to the script.

Each invocation creates a unique US Eastern Time subdirectory named with the day, month, year, weekday, and time, for example `20-Sep-2026-Sun_09-28am-ET`:

- `report.html` — dark summary table and full answers, grouped by case. A Prompts section lists each model's instruction text (system role and user task) and omits the security-event log payload. Each model card shows thinking mode (enabled, disabled, or unsupported), context used versus allocated with the percent of that window, and token counts (input, thinking, output). Run details on each card stay collapsed until opened. A near-full context window is a warning. Thinking traces expand when present. Output and error text are HTML-escaped.
- `declared-inference-configurations/` — a copy of each matching inference configuration, written before inference.
- `results.jsonl` — one row per attempted model/case pair, saved immediately. Includes the raw Ollama response, parsed sections, validation errors, unknown evidence IDs, exact request messages and options, prompt and input hashes, model digest, declared inference configuration path, timings, token counts, and context allocated and used.
- `manifest.json` — selected models (digest, capabilities, quantization, native context, and the `num_ctx` sent), declared inference configurations, shared request settings, and input file identities. Pending cases stay visible if the process is interrupted.

Runs are sequential and grouped by model to reduce repeated loading. Every case gets a fresh conversation. Sampling, context, and thinking come from the matched inference configuration, using the fallbacks above. The last case for each model requests unloading afterward. An individual inference error leaves the remaining cases running, and the report updates after every saved result.

Exit status is nonzero when any run is invalid or failed, any run meets or exceeds its `num_ctx`, preflight fails, or execution is interrupted. Completed results remain available.

`ok` means the answer passed the format and evidence-ID checks and stayed under `num_ctx`. Compare the full answers with the answer keys. This version runs each pair once and does not assign detection-accuracy scores.

Wall time includes model load and request overhead. Eval time is prompt processing plus generation for that log, and excludes load and unload. These are observed timings. Context allocated is the requested `num_ctx`. Context used is prompt tokens plus generated tokens. Thinking and output tokens divide that generated count: the split is exact when only one of those texts is present, and estimated from character length when both are. Identical context settings can still mean different effective context across tokenizers and native context limits. Use inputs that fit every selected model.

## Chat templates

`build_messages()` sends ordinary `role` / `content` text. Markdown headings and the XML tags are application text. Let the installed model's [Ollama chat template](https://docs.ollama.com/modelfile#template) supply its native role and turn tokens. Leave ChatML tokens, Llama headers, Mistral `[INST]` markers, and Gemma turn markers to that template. [Templates differ even between models from the same base](https://huggingface.co/docs/transformers/chat_templating). For a model whose native format has no separate system role, confirm the installed template still carries the analyst instructions. Inspect an import with `ollama show --modelfile MODEL`.

Before any chat call, `compare_models.py` requires the model to appear in `ollama list`, to support text completion when capabilities are advertised, and to pass the format check in [shared/model_formats.toml](shared/model_formats.toml). That check uses Ollama `/api/show` (the same data as `ollama show --template MODEL`) plus the Modelfile `RENDERER` line, parser, and stop sequences. Inference starts only when the installed package matches the family below. A name with no registered family fails. A `RENDERER` line fails for every family except Gemma 4 and GLM-4.7, because the renderer replaces the template. Qwen3.5 does not match the Qwen3 rule.

| Family | Required framing |
| --- | --- |
| Foundation-Sec | `<\|system\|>`, `<\|user\|>`, `<\|assistant\|>`. Fails on `<\|start_header_id\|>` or any `RENDERER`. No `PARSER`. Stop is only `<\|end_of_text\|>` |
| CyberPal 2.0 | `<\|start\|>system<\|message\|>`, `<\|start\|>developer<\|message\|>`, `<\|channel\|>`, `<\|end\|>`. Stops `<\|return\|>` and `<\|call\|>` |
| gpt-oss | `<\|start\|>system<\|message\|>`, `<\|start\|>developer<\|message\|>`, `<\|channel\|>`, `<\|end\|>`. The library package sets no stop lines |
| Qwen3 | `<\|im_start\|>` and `<\|im_end\|>` |
| Llama 3 | `<\|start_header_id\|>` and `<\|eot_id\|>` |
| Mistral Small | `[SYSTEM_PROMPT]`, `[/SYSTEM_PROMPT]`, `[INST]`, `[/INST]` |
| Mistral Nemo | `[INST]`, `[/INST]`, and `.System` |
| Gemma 4 | `RENDERER gemma4` or `RENDERER gemma4-large`. `{{ .Prompt }}` is fine only with that renderer |
| GLM-4.7 | `RENDERER glm-4.7` and `PARSER glm-4.7`. `{{ .Prompt }}` is fine only with that renderer |
| Granite 4 | `<\|im_start\|>` and `<\|im_end\|>` |
| DeepSeek-R1 | `<｜User｜>` and `<｜Assistant｜>` (fullwidth vertical bar, U+FF5C) |
| Command R | `<\|START_OF_TURN_TOKEN\|>`, `<\|SYSTEM_TOKEN\|>`, `<\|USER_TOKEN\|>`, `<\|CHATBOT_TOKEN\|>`. Fails when `<\|END_OF_TURN_TOKEN\|>` is written immediately before the chatbot turn |
| Phi-3 Medium 128K | `<\|system\|>`, `<\|user\|>`, `<\|assistant\|>`, `<\|end\|>`. Stops are exactly `<\|end\|>`, `<\|user\|>`, and `<\|assistant\|>` |

The Ollama library template for `command-r` closes the user turn with `<|END_OF_TURN_TOKEN|>`, then writes that token again before `<|CHATBOT_TOKEN|>`. Cohere's published template has one end token there. Recreate the installed name from [modelfiles/Modelfile.command-r](modelfiles/Modelfile.command-r):

```bash
ollama create command-r -f modelfiles/Modelfile.command-r
```

The raw `hf.co/fdtn-ai/Foundation-Sec-8B-Instruct-Q8_0-GGUF:latest` import is that bare `{{ .Prompt }}` case. The GGUF has no `tokenizer.chat_template`, the Hub repo has no Ollama `template` file, and the call sends only the user text: it omits `<|system|>`, `<|user|>`, and `<|assistant|>` from [`chat_template.jinja`](https://huggingface.co/fdtn-ai/Foundation-Sec-8B-Instruct/blob/main/chat_template.jinja) and drops the harness system message. The weights are Llama 3.1–based, but this checkpoint was not trained on `<|start_header_id|>` headers, so those tokenizer tokens are the wrong framing. A `RENDERER` on a different model does not make this name usable. Run `foundation-sec-8b-instruct` from `modelfiles/Modelfile.foundation-sec-8b-instruct`.

## Adding a model

Format rules live in `shared/model_formats.toml` and are applied by `shared/model_config.py`. Add an ordered `[[formats]]` entry from the checkpoint's published template and generation settings. Name matching is case-insensitive, and the first matching regular expression wins. Put local wrapper names in the pattern when they differ from the upstream name. Use `markers` for an Ollama template or `renderers` for a built-in renderer.

```toml
[[formats]]
label = "Example Instruct"
name_pattern = "example-instruct"
markers = ["<user>", "<assistant>"]
parsers = []
required_stops = ["<end>"]
hint = "Create example-instruct using modelfiles/Modelfile.example-instruct."
```

Replace those tokens with the model's own tokens. `parsers = []` means plain text and no `PARSER` directive. A nonempty `parsers` list names the accepted parsers. Omitting `parsers` leaves the parser unconstrained. `required_stops` must be configured. Extra stops are allowed unless `allowed_stops` is set, which limits stops to that list. Foundation-Sec's EOS-only policy is its registry entry. Marker presence does not prove every rendered conversation matches the training template.

If an imported GGUF lacks the right framing, add a Modelfile under `modelfiles/` and create a local wrapper, as with Foundation-Sec. The registry checks the installed package. It does not download models or rewrite installed templates. Add an `inference_config/*.conf` file for that wrapper. `compare_models.py` then uses that configuration without code changes.

## Layout

| Path | Role |
| --- | --- |
| `compare_models.py` | Run one or more installed models on one or more JSONL files and write `results/`. One model and one file is `--models NAME --logs PATH` |
| `shared/harness.py` | Defaults, log loading, preflight, inference, token accounting, and output validation |
| `shared/prompts.py` | Shared hunt prompt, and `MODEL_PROMPTS` for a model-specific prompt |
| `shared/model_config.py` | Validation of installed templates, renderers, parsers, and stop sequences |
| `shared/model_formats.toml` | Ordered model format requirements |
| `shared/inference_configurations.py` | Inference configuration loading, settings validation, and run selection |
| `shared/run_reports.py` | Run directories, inference configuration snapshots, result files, and HTML reports |
| `inference_config/*.conf` | Per-model eval setup. Uncommented values override Ollama and Modelfile defaults for that run |
| `modelfiles/Modelfile.foundation-sec-8b-instruct` | `<\|system\|>` / `<\|user\|>` / `<\|assistant\|>` template for the Foundation-Sec GGUF import |
| `modelfiles/Modelfile.command-r` | Command R template with one `<\|END_OF_TURN_TOKEN\|>` before the chatbot turn |
| `logs/*.jsonl` | Seven synthetic ECS scenarios. A single-file run passes one of these to `--logs` |
| `logs/README.md` | Field notes for the lookalike evidence packages |
| `evals/answer-keys.md` | Human grading keys for all seven cases. The harness does not load this file |
| `tests/test_main.py` | Tests for think-arg gating, incomplete replies, and hunt-output validation |
| `tests/test_compare_models.py` | Tests for the comparison matrix, HTML report, and the shared hunt runner |
| `tests/test_model_config.py` | Config-driven validation for model packages |
| `pyproject.toml` | Project metadata and the `ollama` client |

```bash
uv run python -m unittest -v
```
