# Threat Hunting Agent Harness

A bare-bones agentic harness for education and proof-of-concept work, not a production SOC product.

Give a local model some security events, constrain how it reasons, and print a structured hunt result. The lesson stays in a few files you can hold in your head.

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

The default model is `foundation-sec-8b-instruct`. Pull the weights, then create that name from [modelfiles/Modelfile.foundation-sec-8b-instruct](modelfiles/Modelfile.foundation-sec-8b-instruct). The pull installs `hf.co/fdtn-ai/Foundation-Sec-8B-Instruct-Q8_0-GGUF:latest`, whose template is bare `{{ .Prompt }}` with no `RENDERER`. Preflight refuses that import. The `ollama create` step is what sends `<|system|>`, `<|user|>`, and `<|assistant|>`. Re-run it after a Modelfile change to refresh an existing wrapper; it reuses the downloaded weights:

```bash
ollama pull hf.co/fdtn-ai/Foundation-Sec-8B-Instruct-Q8_0-GGUF
ollama create foundation-sec-8b-instruct -f modelfiles/Modelfile.foundation-sec-8b-instruct
```

Another installed name works when preflight recognizes that family's chat template. Pass it with `--model`.

Foundation-Sec uses plain-text output and no Ollama `PARSER` directive. Its only configured stop must be `<|end_of_text|>`. Both CLIs check that before inference. The wrapper appends that EOS token to completed assistant turns. The harness then parses the returned text into the four answer fields. That application parser is separate from Ollama's output parser.

From this repo, the default hunt is `logs/http-beaconing.jsonl` on `foundation-sec-8b-instruct`:

```bash
uv sync
uv run python main.py
uv run python main.py --model qwen3:32b
uv run python main.py logs/http-beaconing.jsonl --model qwen3:32b
uv run python main.py --profile profiles/qwen3-32b.profile
uv run python main.py --timeout 600
```

`--timeout` is a positive number of seconds for one Ollama HTTP request. `--timeout none` is the same unlimited wait as the default. The limit applies to a single request.

The script prints the model, the declared profile, the installed chat template, the log file, the events, then an `--- Analysis ---` block with `Verdict`, `Threat type`, `Summary`, and `Evidence`. It also prints input, thinking, and output tokens, how those generated tokens were split, and context used versus `num_ctx`. It writes a results directory the same way a comparison does: the profile is copied in before inference, then the hunt result is appended.

## What a hunt does

`main.py` and `compare_models.py` share `run_hunt` from `shared/harness.py`. Each call starts a fresh conversation with the same prompt and serialized events. Neither CLI imports the other.

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

- a file in `profiles/` is empty, missing `model`, is not `key=value` lines, or uses a key that is not an exact profile field name
- the model is missing from `ollama list`, lacks text completion, or its installed template, renderer, parser, or stops fail the format check (see [Chat templates](#chat-templates))
- the log file is missing, unreadable, malformed, or contains no events
- several profiles name the model and you omitted `--profile`

## Scenarios

Grade every case with the [answer keys](evals/answer-keys.md). Keep those keys out of the model prompt. Field notes for the lookalike packages are in [logs/README.md](logs/README.md).

| Log | Contents |
| --- | --- |
| `logs/http-beaconing.jsonl` | Default single-hunt file. Periodic HTTP check-ins among ordinary traffic |
| `logs/password-spray.jsonl` | Authentication events with a many-account failure sequence |
| `logs/internal-network-scan.jsonl` | Internal port probes among ordinary traffic |
| `logs/shared-vpn-logins.jsonl` | Shared-VPN logins that resemble spraying |
| `logs/managed-telemetry.jsonl` | Managed check-ins that resemble beaconing |
| `logs/scheduled-discovery.jsonl` | Scheduled discovery that resembles an internal scan |
| `logs/opaque-sync-transfers.jsonl` | Host and network evidence for an inconclusive transfer |

## Profiles and request settings

Each file in `profiles/` is one eval setup: the installed Ollama name, and the request values for that run. An uncommented value is sent on the chat request and overrides the Ollama or Modelfile default for that call. A `#` line stays in the file for a later change and leaves the default in place. Blank lines are ignored.

Lines are `key=value`. The key must be exactly `model`, `num_ctx`, `thinking`, `temperature`, `top_p`, `top_k`, `weight_precision`, `weight_quant`, `kv_cache`, or `repeat_penalty`. A different case, hyphen, or space (`Num Ctx`, `num-ctx`) stops the run. `alpha` and `alpha:latest` are the same model. Two files may name the same model when the settings differ; those are different runs. Pass `--profile` with the file you want. If more than one file matches and you omit `--profile`, the run stops and lists them.

`profiles/qwen3-32b.profile`:

```
model=qwen3:32b
# weight_precision=Developer Q8_0
# kv_cache=Q8_0
thinking=true
num_ctx=40960
# basis: Qwen's explicit thinking-mode recommendation (https://huggingface.co/Qwen/Qwen3-32B)
temperature=0.6
top_p=0.95
top_k=20
# repeat_penalty=1.0
```

Other recorded setups: `llama3.3:70b` (`num_ctx` 16384), `granite4.2:30b` (65536, `thinking=high`), `deepseek-r1:32b` (65536, `thinking=true`), `command-r:latest` (131072), `gemma4:31b` (32768, `thinking=true`), `mistral-small3.2:24b` (131072), `mistral-nemo:12b` (131072), and `foundation-sec-8b-instruct` (131072). `thinking` stays commented on models that do not support it. Granite 4.2 accepts `false`, `low`, `medium`, and `high`. Qwen3, DeepSeek-R1, and Gemma 4 are on or off (`thinking=true` or `false`).

Both CLIs load the chosen profile before the first inference call, print it, and copy it into the results directory. A model with no file is reported as having no declared profile. If an uncommented weight quant differs from the installed Ollama quantization, the report keeps both and notes the difference. A malformed profile stops the run before any results directory is created. `num_ctx` must be one positive integer written as digits only (commas are rejected). `temperature` is a non-negative decimal, `top_p` is a decimal from 0 through 1, and `top_k` is a positive integer.

| Profile key | Request field | If the line is absent |
| --- | --- | --- |
| `num_ctx` | `options.num_ctx` | `32768` from `NUM_CTX` in `shared/harness.py` |
| `thinking` | `think`: `true`, `false`, `low`, `medium`, or `high` | `THINK` in `shared/harness.py` (`False`) |
| `temperature` | `options.temperature` | `0` from `TEMPERATURE` |
| `top_p`, `top_k` | matching sampling options | omitted |
| — | `options.seed` | always `0` (`SEED`). An omitted seed becomes `-1`, and a negative seed picks a new seed each run |
| — | `options.num_predict` | always `-1` (no output-token limit) |
| — | `shift` | always off (`SHIFT = False`) |

The harness asks Ollama (`/api/show`) and sends `think` only when the model lists the `thinking` capability. Models without that capability reject the argument. Qwen3-class models think by default when the API omits `think`, so those models always receive an explicit value. A profile that sets `thinking` for a model without the capability stops the run before any chat call. Set `THINK` in `shared/harness.py` to `True` or `False` before a run whose profile has no `thinking` line. Thinking tokens and the final answer share `num_ctx`.

Weight precision, KV cache, and repeat penalty stay commented out. The expected KV cache type is `f16` (`OLLAMA_KV_CACHE_TYPE` on the Ollama server). It is recorded on each hunt and is not sent as a chat option. The saved profile is what distinguishes two runs of the same model when those settings later differ.

## Compare models

Edit `MODELS` and `DEFAULT_SCENARIO_LOGS` at the top of `compare_models.py`, or override them on the command line. The defaults compare Qwen3 32B, Mistral Small 3.2 24B, and Foundation-Sec 8B across all seven scenarios: 21 runs. Use names from `ollama list`. A name without a tag resolves to `:latest` when that installed name exists. Models are checked before inference and are never downloaded. Missing models, unrecognized names, and wrong templates are reported together.

```bash
uv run python compare_models.py
uv run python compare_models.py --models qwen3:32b mistral-small3.2:24b --logs logs/password-spray.jsonl logs/http-beaconing.jsonl
uv run python compare_models.py --models mistral-small3.2:24b --timeout 600 --output-dir results
uv run python compare_models.py --profiles-dir profiles
uv run python compare_models.py --models qwen3:32b --profile profiles/qwen3-32b.profile profiles/qwen3-32b-other.profile
```

Log paths are relative to the script, or absolute. `--output-dir` and `--profiles-dir` are relative to the current working directory. The default profiles directory is `profiles/` next to the script.

Each invocation creates a unique US Eastern Time subdirectory named with the day, month, year, weekday, and time, for example `20-Sep-2026-Sun_09-28am-ET`:

- `report.html` — dark summary table and full answers, grouped by case. Declared profiles sit above the table, including models with none recorded. Each model card shows quantization, thinking mode (enabled, disabled, or unsupported), context used versus allocated, and token counts (input, thinking, output). Run details on each card stay collapsed until opened. A near-full context window is a warning. Thinking traces expand when present. Output, profile, and error text are HTML-escaped.
- `declared-profiles/` — a copy of each matching profile, written before inference. The initial report already lists those profiles at `0 / N` runs recorded.
- `results.jsonl` — one row per attempted model/case pair, saved immediately. Includes the raw Ollama response, parsed sections, validation errors, unknown evidence IDs, exact request messages and options, prompt and input hashes, model digest, declared profile path, timings, token counts, and context allocated and used.
- `manifest.json` — selected models (digest, capabilities, quantization, native context, and the `num_ctx` sent), declared profiles, shared request settings, and input file identities. Pending cases stay visible if the process is interrupted.

Runs are sequential and grouped by model to reduce repeated loading. Every case gets a fresh conversation. Sampling, context, and thinking come from the matched profile, using the fallbacks above. The last case for each model requests unloading afterward. An individual inference error leaves the remaining cases running, and the report updates after every saved result.

Exit status is nonzero when any run is invalid or failed, any run meets or exceeds its `num_ctx`, preflight fails, or execution is interrupted. Completed results remain available.

`ok` means the answer passed the format and evidence-ID checks and stayed under `num_ctx`. Compare the full answers with the answer keys. This version runs each pair once and does not assign detection-accuracy scores.

Wall time includes model load and request overhead. Eval time is prompt processing plus generation for that log, and excludes load and unload. These are observed timings. Context allocated is the requested `num_ctx`. Context used is prompt tokens plus generated tokens. Thinking and output tokens divide that generated count: the split is exact when only one of those texts is present, and estimated from character length when both are. Identical context settings can still mean different effective context across tokenizers and native context limits. Use inputs that fit every selected model.

## Chat templates

`build_messages()` sends ordinary `role` / `content` text. Markdown headings and the XML tags are application text. Let the installed model's [Ollama chat template](https://docs.ollama.com/modelfile#template) supply its native role and turn tokens. Leave ChatML tokens, Llama headers, Mistral `[INST]` markers, and Gemma turn markers to that template. [Templates differ even between models from the same base](https://huggingface.co/docs/transformers/chat_templating). For a model whose native format has no separate system role, confirm the installed template still carries the analyst instructions. Inspect an import with `ollama show --modelfile MODEL`.

Before any chat call, both CLIs require the model to appear in `ollama list`, to support text completion when capabilities are advertised, and to pass the format check in [shared/model_formats.toml](shared/model_formats.toml). That check uses Ollama `/api/show` (the same data as `ollama show --template MODEL`) plus the Modelfile `RENDERER` line, parser, and stop sequences. Inference starts only when the installed package matches the family below. A name with no registered family fails. A `RENDERER` line fails for every family except Gemma 4, because the renderer replaces the template. Qwen3.5 does not match the Qwen3 rule.

| Family | Required framing |
| --- | --- |
| Foundation-Sec | `<\|system\|>`, `<\|user\|>`, `<\|assistant\|>`. Fails on `<\|start_header_id\|>` or any `RENDERER`. No `PARSER`. Stop is only `<\|end_of_text\|>` |
| CyberPal 2.0 | `<\|start\|>system<\|message\|>`, `<\|start\|>developer<\|message\|>`, `<\|channel\|>`, `<\|end\|>`. Stops `<\|return\|>` and `<\|call\|>` |
| Qwen3 | `<\|im_start\|>` and `<\|im_end\|>` |
| Llama 3 | `<\|start_header_id\|>` and `<\|eot_id\|>` |
| Mistral Small | `[SYSTEM_PROMPT]`, `[/SYSTEM_PROMPT]`, `[INST]`, `[/INST]` |
| Mistral Nemo | `[INST]`, `[/INST]`, and `.System` |
| Gemma 4 | `RENDERER gemma4` or `RENDERER gemma4-large`. `{{ .Prompt }}` is fine only with that renderer |
| Granite 4 | `<\|im_start\|>` and `<\|im_end\|>` |
| DeepSeek-R1 | `<｜User｜>` and `<｜Assistant｜>` (fullwidth vertical bar, U+FF5C) |
| Command R | `<\|START_OF_TURN_TOKEN\|>`, `<\|SYSTEM_TOKEN\|>`, `<\|USER_TOKEN\|>`, `<\|CHATBOT_TOKEN\|>` |

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

If an imported GGUF lacks the right framing, add a Modelfile under `modelfiles/` and create a local wrapper, as with Foundation-Sec. The registry checks the installed package. It does not download models or rewrite installed templates. Add a `profiles/*.profile` file for that wrapper. Both CLIs then use the same configuration without code changes.

## Layout

| Path | Role |
| --- | --- |
| `main.py` | Single-hunt CLI. Uses the shared harness, profiles, and report writer |
| `compare_models.py` | Compare installed models across the JSONL scenarios and write `results/` |
| `shared/harness.py` | Defaults, log loading, preflight, prompts, inference, token accounting, and output validation |
| `shared/model_config.py` | Validation of installed templates, renderers, parsers, and stop sequences |
| `shared/model_formats.toml` | Ordered model format requirements |
| `shared/model_profiles.py` | Profile loading, settings validation, and run selection |
| `shared/run_reports.py` | Run directories, profile snapshots, result files, and HTML reports |
| `profiles/*.profile` | Per-model eval setup. Uncommented values override Ollama and Modelfile defaults for that run |
| `modelfiles/Modelfile.foundation-sec-8b-instruct` | `<\|system\|>` / `<\|user\|>` / `<\|assistant\|>` template for the Foundation-Sec GGUF import |
| `logs/*.jsonl` | Seven synthetic ECS scenarios. The default hunt file is `logs/http-beaconing.jsonl` |
| `logs/README.md` | Field notes for the lookalike evidence packages |
| `evals/answer-keys.md` | Human grading keys for all seven cases. The harness does not load this file |
| `tests/test_main.py` | Tests for think-arg gating, incomplete replies, and hunt-output validation |
| `tests/test_compare_models.py` | Tests for the comparison matrix, HTML report, and the shared hunt runner |
| `tests/test_model_config.py` | Config-driven validation for model packages |
| `pyproject.toml` | Project metadata and the `ollama` client |

```bash
uv run python -m unittest -v
```
