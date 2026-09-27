# Threat Hunting Agent Harness

A **bare-bones agentic harness** for education and proof-of-concept work.

The point is not a production SOC product. It is a small, readable loop you can inspect, change, and demonstrate: give a local model some security events, constrain how it reasons, and print a structured hunt result.

## What this is for

- Teaching the shape of an agentic workflow without a large framework
- Showing a local, defensive security model reading logs
- Giving a canned scenario so demos are repeatable
- Leaving room to grow into tools, multi-step investigation, and evaluation

It is intentionally small. Prefer a few files you can hold in your head over a platform.

## What it does today

`main.py` is a single-pass harness:

1. Load events from a JSONL file (`logs/http-beaconing.jsonl` by default)
2. Sort them by timestamp
3. Send them to a local [Ollama](https://ollama.com) model with a defensive analyst prompt
4. Print a fixed-format answer: verdict, threat type, summary, and evidence IDs
5. Reject replies that are incomplete, missing required sections, use an illegal verdict, or cite event IDs that are not in the supplied events

The prompt treats log contents as untrusted evidence, not instructions, and forbids inventing users, IPs, timestamps, or event IDs. Cite `event.id`.

The harness then machine-checks that contract: `Verdict` must be `suspicious`, `benign`, or `inconclusive`; all four sections must be present; every cited evidence ID must appear on a supplied event. Valid IDs do not prove the summary or threat type is correct.

The single-hunt command and comparison runner share `run_hunt(model, events)`. Each call starts a fresh conversation with the same prompt and serialized events. The loader could become a tool later if you want the model to request data instead of receiving the full dump up front.

`compare_models.py` runs that same hunt across several installed Ollama models and writes a side-by-side report under `results/`.

## Prompt portability

`build_messages()` sends standard `role`/`content` dictionaries: the system message contains the analyst instructions, evidence rules, verdict definitions, and output contract; the user message contains the task and a JSON array inside `<security_events>` delimiters. Markdown headings and these XML-style delimiters are ordinary application text, not a universal model-specific prompt format. The four answer fields (`Verdict`, `Threat type`, `Summary`, `Evidence`) are this harness's output contract, not reserved LLM fields.

Let the installed model's [Ollama chat template](https://docs.ollama.com/modelfile#template) supply its native role/turn tokens. Do not manually add ChatML tokens, Llama headers, Mistral `[INST]` markers, or Gemma turn markers to these prompts. [Chat templates differ even between models derived from the same base](https://huggingface.co/docs/transformers/chat_templating). For models whose native format lacks a separate system role, verify that the installed template incorporates those instructions appropriately; a portable message API does not imply identical role support. Inspect imported GGUF models with `ollama show --modelfile MODEL`. A bare `{{ .Prompt }}` template does not provide native chat framing when that Modelfile has no `RENDERER` line. Gemma 4 keeps that placeholder and is framed by `RENDERER gemma4`, which writes `<|turn>system`, `<|turn>user`, and `<|turn>model`. DeepSeek-R1 is framed by its installed template's `<｜User｜>` and `<｜Assistant｜>` markers (a fullwidth vertical bar, not ASCII `|`).

The Hugging Face GGUF `hf.co/fdtn-ai/Foundation-Sec-8B-Instruct-Q8_0-GGUF:latest` is a real bare-prompt import: `TEMPLATE {{ .Prompt }}` and no `RENDERER`. The GGUF has no `tokenizer.chat_template`, the Hub repo has no Ollama `template` file, and that path sends only the user text. It omits `<|system|>`, `<|user|>`, and `<|assistant|>` from [`chat_template.jinja`](https://huggingface.co/fdtn-ai/Foundation-Sec-8B-Instruct/blob/main/chat_template.jinja), and it does not interpolate the harness system message. The weights are Llama 3.1–based, but this instruct checkpoint was not trained on Llama `<|start_header_id|>` headers; those tokens are still in the tokenizer and are the wrong framing. A `RENDERER` on some other model does not make this name usable. Run `foundation-sec-8b-instruct`, created from `Modelfile.foundation-sec-8b-instruct`.

Before any chat call, both CLIs share the same preflight: the model must appear in `ollama list`, support text completion when capabilities are advertised, and pass the installed-template check via Ollama `/api/show` (the same data as `ollama show --template MODEL`) plus the Modelfile `RENDERER` line. Log files must exist, parse as JSONL, and contain at least one event. Inference does not start unless that selected template is the native framing for the model family. Foundation-Sec names need `<|system|>`, `<|user|>`, and `<|assistant|>`, and fail on `<|start_header_id|>` or any `RENDERER` line. Qwen3 needs `<|im_start|>` and `<|im_end|>`. Llama 3 needs `<|start_header_id|>` and `<|eot_id|>`. Mistral Small needs `[SYSTEM_PROMPT]`, `[/SYSTEM_PROMPT]`, `[INST]`, and `[/INST]`. Mistral Nemo needs `[INST]`, `[/INST]`, and `.System`. Gemma 4 needs `RENDERER gemma4` or `RENDERER gemma4-large` (its `{{ .Prompt }}` placeholder is fine only with that renderer). Granite 4 needs `<|im_start|>` and `<|im_end|>`. DeepSeek-R1 needs `<｜User｜>` and `<｜Assistant｜>`. Command R needs `<|START_OF_TURN_TOKEN|>`, `<|SYSTEM_TOKEN|>`, `<|USER_TOKEN|>`, and `<|CHATBOT_TOKEN|>`. A `RENDERER` line fails for every family except Gemma 4, because the renderer replaces the template. A name with no registered family fails closed. Passing the raw `hf.co/...` Foundation-Sec import exits with that error instead of hunting.

The evidence serializer escapes `<`, `>`, and `&` using JSON Unicode escapes. This prevents a field containing `</security_events>` from literally ending the outer evidence block while preserving its exact decoded value. These delimiters are not a security boundary or a guarantee of instruction-following. Existing completion, output-format, and evidence-ID checks still apply. Prompt structure is tested offline; response quality and compatibility must be evaluated per installed model/template.

## Demo scenarios

Use the [private MVP answer keys](evals/answer-keys.md) to grade all six cases manually. They define expected assessments, supporting evidence, and unsupported claims; keep them out of the model prompt.

### Password spray (`logs/password-spray.jsonl`)

Fourteen synthetic [Elastic Common Schema](https://www.elastic.co/docs/reference/ecs) authentication documents from `auth-01.corp.internal`, one nested JSON object per line. `@timestamp` is ISO-8601 UTC. Cite `event.id` (`c04-e001` … `c04-e014`).

All events concern password authentication to the same `employee-portal` service. The single suspicious sequence is six failures from `198.51.100.7` (a documentation IP) against six different accounts in 150 seconds (`c04-e005` … `c04-e010`), followed by a success for `bob` from that same source (`c04-e011`). The seven benign events are ordinary internal logins and two isolated failure-then-success pairs (`c04-e003`/`c04-e004` and `c04-e013`/`c04-e014`). The original times, users, source IPs, and authentication outcomes are preserved.

Ground truth for instructors (not present in the logs): the intended scenario is one password spray; verdict `suspicious`; attack-sequence evidence = `c04-e005` … `c04-e011`. The observable pattern is consistent with spraying and possible compromise. Because the logs do not record attempted passwords or credential provenance, they cannot prove password reuse, distinguish spraying conclusively from credential stuffing, or establish account takeover from the subsequent success alone. No passwords or attack labels are embedded in the events.

Use it to check whether the model:

- recognizes the many-account, low-attempt-count pattern and treats the later success as suspicious
- cites real `event.id` values and distinguishes observations from hypotheses
- ignores the benign typo-and-retry noise

### HTTP beaconing (`logs/http-beaconing.jsonl`)

Default single-hunt file for `main.py`. Synthetic [ECS](https://www.elastic.co/docs/reference/ecs) JSONL (the nested document shape Filebeat/Logstash write into Elasticsearch). Each line is one firewall/proxy event. Cite `event.id` (`c01-e001` … `c01-e043`).

The hour of traffic mixes ordinary work with a low-and-slow HTTP check-in:

- `jlee` on `ws-014.corp.internal` (`10.47.12.88`) browses Office, GitHub, and LinkedIn, and syncs Outlook (`outlook.office365.com` / `198.51.100.30`) on an irregular schedule with varying payload sizes.
- The same host also issues `GET /api/heartbeat` to `203.0.113.77:443` (a documentation IP, no hostname) about every 300 seconds with a few seconds of jitter, tiny stable byte counts, and an identical short user-agent. Those twelve events are `c01-e006`, `c01-e011`, `c01-e016`, `c01-e020`, `c01-e023`, `c01-e024`, `c01-e027`, `c01-e031`, `c01-e035`, `c01-e038`, `c01-e041`, and `c01-e043`.
- Cover traffic that can look periodic if you only glance at timestamps: Windows Update from `asmith` / `ws-022` (large, variable bodies), Slack presence polls from `bnguyen` / `ws-008` (~15 minutes apart with high jitter and changing sizes), plus DNS, NTP, and SMB.

Ground truth for instructors (not present in the logs): verdict `suspicious`, threat type HTTP/C2 beaconing, evidence = the twelve `event.id`s to `203.0.113.77`.

Use it to check whether the model:

- names beaconing (regular interval, low jitter, consistent small payloads, odd destination) rather than “lots of HTTPS”
- cites those real `event.id` values
- ignores Update, Outlook, and Slack lookalikes

## How to run

You need Python 3.14+, [uv](https://docs.astral.sh/uv/), and [Ollama](https://ollama.com) running locally. This harness does not download weights. It talks to Ollama on your machine and will fail if the chosen model is not installed.

1. On macOS, set the Ollama server environment before you start Ollama (or restart it afterward). These `launchctl` values enable flash attention, use an f16 KV cache, allow one parallel request, and keep one model loaded. They last until logout or reboot. Quit and reopen the Ollama app, or restart `ollama serve`, so the running server inherits them:

```bash
launchctl setenv OLLAMA_FLASH_ATTENTION 1
launchctl setenv OLLAMA_KV_CACHE_TYPE f16
launchctl setenv OLLAMA_NUM_PARALLEL 1
launchctl setenv OLLAMA_MAX_LOADED_MODELS 1
```

2. Install Ollama and start it. On macOS that is typically `brew install ollama` then `ollama serve` (or open the Ollama app). Confirm it is up and see which models you already have:

```bash
ollama list
```

3. Pick a model from that list. If the list is empty (or you want the suggested defensive model), pull it once (several GB):

```bash
ollama pull hf.co/fdtn-ai/Foundation-Sec-8B-Instruct-Q8_0-GGUF
ollama create foundation-sec-8b-instruct -f Modelfile.foundation-sec-8b-instruct
```

The pull installs `hf.co/fdtn-ai/Foundation-Sec-8B-Instruct-Q8_0-GGUF:latest`. That name has no native chat template (`{{ .Prompt }}`). The `ollama create` step is required so the script default `foundation-sec-8b-instruct` sends `<|system|>`, `<|user|>`, and `<|assistant|>`. Another installed name is usable when preflight recognizes that family's chat template; pass it with `--model`.

4. From this repo, install Python deps and run a hunt. Default is the HTTP-beaconing file and `foundation-sec-8b-instruct`:

```bash
uv sync
uv run python main.py
```

Pass `--model` with a name from `ollama list`, and optionally another JSONL file:

```bash
uv run python main.py --model qwen3:32b
uv run python main.py logs/http-beaconing.jsonl --model qwen3:32b
uv run python main.py --profile profiles/qwen3-32b.profile
uv run python main.py --timeout 600
```

Each Ollama HTTP request waits until it finishes. Pass `--timeout` with a positive number of seconds to stop a single request, or `--timeout none` for the same unlimited wait as the default. That limit is per request, not a deadline for the whole run.

A second profile for the same model is a different setup. Run it on its own so its result stays separate:

```bash
uv run python main.py --profile profiles/qwen3-32b-other.profile
```

The script prints the model name, the declared profile, the installed chat template, the log file, the events it is sending, then an `--- Analysis ---` block with `Verdict`, `Threat type`, `Summary`, and `Evidence`. It also prints input, thinking, and output tokens, how those generated tokens were split, and context used versus the configured `num_ctx`. It writes a results directory, the same kind of record as a comparison: the profile is copied in before inference, then the hunt result is appended. Pass `--profile` to choose the setup. If several profiles name the model and you do not pass `--profile`, the hunt stops and lists them. It exits with an error instead of looking like a successful hunt when:

- a file in `profiles/` is empty, missing `model`, or is not `key=value` lines
- the model is not installed (`ollama list`), lacks text completion, or its installed template is not the native framing for that family (wrong role markers, a `RENDERER` on a non-Gemma model, Gemma 4 without `RENDERER gemma4` or `gemma4-large`, or a name with no registered template)
- the log file is missing, unreadable, malformed, or contains no events
- generation hits the token limit (`done_reason=length`) or returns empty content
- a required section is missing or empty
- the verdict is not `suspicious`, `benign`, or `inconclusive`
- a cited evidence ID is not in the supplied events (for example `nonexistent-id`)
- context used (prompt tokens plus generated tokens) is at least the configured `num_ctx`

Usage at or above 90% of `num_ctx`, while still under that window, prints a warning and does not by itself fail the run. Valid citations do not mean the explanation is right.

Context shift is off for every model, including DeepSeek2. `SHIFT` in `main.py` defaults to `False`, and every chat request sends that value as `shift`. Ollama 0.33 enables context shift when the field is omitted, except for DeepSeek2, and slides older tokens out once `num_ctx` is full. With shift off, a full window stays an error.

## Thinking mode

Before you run, set `THINK` in `main.py` to `True` (on) or `False` (off) for models that support thinking. Do not leave the choice implicit.

The harness queries Ollama (`/api/show`) and sends `think` only when the model lists the `thinking` capability. Models without that capability reject the argument (`does not support thinking`), so it is omitted. Qwen3-class models enable thinking by default when the API omits `think`; for those models the harness always sends `think` explicitly.

Both `main.py` and `compare_models.py` always send `num_predict=-1` (no output-token limit). Thinking tokens and the final answer still share the configured `num_ctx`. A full context window is an error. A reply that stops with `done_reason=length` is still a failed hunt.

## Compare models across scenarios

Each model used in a test or eval has its own file in `profiles/`. The file is that model's setup for the run: which installed Ollama name to call, and the request values to use for it. Ollama and the model's Modelfile already have defaults, such as `PARAMETER num_ctx`. An uncommented profile value is sent on the chat request and may override those defaults for that call. `num_ctx` is the eval context window. The harness sends it as `options.num_ctx`, which replaces the Modelfile context size and Ollama's default for the request. A `#` line stays in the file for a later change and leaves the Ollama or Modelfile default in place. Blank lines are ignored.

Each line is `key=value`, the same shape as a `.env` file. `model` names the installed model (`alpha` and `alpha:latest` are the same model). Two files may name the same model when the settings differ; those are different runs and can produce different results. Pass `--profile` with the file for the run you want. If more than one file matches a model and you do not pass `--profile`, the run stops and lists the files.

`profiles/qwen3-32b.profile` records this setup:

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

The other recorded setups are `llama3.3:70b` (`num_ctx` 16384), `granite4.2:30b` (65536, `thinking=high`), `deepseek-r1:32b` (65536, `thinking=true`), `command-r:latest` (131072), `gemma4:31b` (32768, `thinking=true`), `mistral-small3.2:24b` (131072), `mistral-nemo:12b` (131072), and `foundation-sec-8b-instruct` (131072). `thinking` stays commented on models that do not support it. Granite 4.2 accepts several reasoning levels (`false`, `low`, `medium`, `high`); its profile comments those values and sets the highest, `high`. Qwen3, DeepSeek-R1, and Gemma 4 are on or off, so their profiles use `thinking=true`.

Both `main.py` and `compare_models.py` load the chosen profile before the first inference call, print it, and copy it into the results directory. A model with no file is reported as having no declared profile. If an uncommented weight quant differs from the installed Ollama quantization, the report keeps both and notes the difference. A malformed profile, including a `num_ctx` that is not one positive integer written as digits only, stops the run before any results directory is created. Commas are not allowed in that value.

A matched profile's `num_ctx` is the context window sent for that hunt, overriding the model's usual context size. A model with no profile, or a profile with no `num_ctx` line, uses 32768 from `NUM_CTX` in `main.py`. An uncommented `thinking` line is resolved before inference and sent as the Ollama `think` argument (`true`, `false`, `low`, `medium`, or `high`) when the installed model lists the thinking capability. A profile that sets `thinking` for a model without that capability stops the run before any chat call. A profile with no `thinking` line uses `THINK` in `main.py`. An uncommented `temperature`, `top_p`, or `top_k` line is sent as the matching Ollama sampling option. A missing `temperature` line uses `0` from `TEMPERATURE` in `main.py`. Missing `top_p` and `top_k` lines are left off the request. `temperature` is a non-negative decimal, `top_p` is a decimal from 0 through 1, and `top_k` is a positive integer; commas and other text stop the run before any results directory is created. Weight precision, KV cache, and repeat penalty stay commented out. The saved profile is what distinguishes two runs of the same model when those settings later differ.

Edit the `MODELS` and `DEFAULT_SCENARIO_LOGS` lists at the top of `compare_models.py`, then run:

```bash
uv run python compare_models.py
```

The defaults compare Qwen3 32B, Mistral Small 3.2 24B, and Foundation-Sec 8B across all six scenarios: password spray, HTTP beaconing, internal network scanning, shared VPN logins (`logs/shared-vpn-logins.jsonl`), managed telemetry (`logs/managed-telemetry.jsonl`), and scheduled discovery (`logs/scheduled-discovery.jsonl`): 18 runs. Use names from `ollama list`. All configured models and input files are checked before inference starts, including whether the installed Ollama chat template is the native framing for that model family; missing models, unrecognized names, and models with the wrong template are reported together and are never downloaded automatically. Names without a tag resolve to `:latest` when that installed name exists.

You can override the lists without editing code:

```bash
uv run python compare_models.py --models qwen3:32b mistral-small3.2:24b --logs logs/password-spray.jsonl logs/http-beaconing.jsonl
uv run python compare_models.py --models mistral-small3.2:24b --timeout 600 --output-dir results
uv run python compare_models.py --profiles-dir profiles
uv run python compare_models.py --models qwen3:32b --profile profiles/qwen3-32b.profile profiles/qwen3-32b-other.profile
```

Log paths are relative to the script (or absolute); a supplied output directory or `--profiles-dir` is relative to your current working directory. The default profiles directory is `profiles/` next to the script. Each invocation creates a unique US Eastern Time subdirectory named with dd-Mon-yyyy, weekday, and hh-mm am/pm (for example `20-Sep-2026-Sun_09-28am-ET`) containing:

- `report.html`: open in a browser for a dark-themed summary table and full answers side by side, grouped by case. Declared profiles appear above the table, including models with none recorded. Each model card and the summary table include quantization, thinking mode (enabled / disabled / not supported), context used vs allocated, and token counts (input, thinking, output, cached, uncached). A near-full context window is shown as a warning. Thinking traces can be expanded when present. Output, profile, and error text are HTML-escaped.
- `declared-profiles/`: a copy of each matching profile, written before inference. The initial report already lists those profiles at `0 / N` runs recorded.
- `results.jsonl`: one row per attempted model/case pair, saved immediately. Includes full raw Ollama response, parsed sections, validation errors, unknown evidence IDs, exact request messages/options, prompt and input hashes, model digest, the declared profile path when one matched, timings, token counts, and context allocated/used.
- `manifest.json`: selected models (digest, capabilities, quantization, native context, and the `num_ctx` sent for that model), declared profiles, plus shared request settings (`num_ctx` as the fallback when a profile does not set one, `num_predict`, `think`, `shift`, `kv_cache_type`) and input file identities. Pending cases remain visible in the report if the process is interrupted.

Runs are sequential and grouped by model to reduce repeated loading. Every case receives a fresh conversation. Sampling comes from the matched profile's `temperature`, `top_p`, and `top_k`. A missing `temperature` line uses `0` from `TEMPERATURE` in `main.py`, and missing `top_p` and `top_k` lines are omitted. `num_predict` (`-1`, no output-token limit) and context shift come from `main.py`. The context window is the matched profile's `num_ctx`, or `NUM_CTX` when that line is absent. The thinking mode is the matched profile's `thinking` line, or `THINK` when that line is absent, and it is placed on the request before the chat call when the model lists the thinking capability. Context shift is sent for every model. The last case for each model requests unloading afterward. Each Ollama HTTP request waits until it finishes. Pass `--timeout` with a positive number of seconds to limit one request; `--timeout none` is the same unlimited wait as the default. It is not a total batch deadline.

Invalid, empty, truncated, and failed responses are retained. An individual inference error does not stop the remaining cases. The report updates after every saved result. Exit status is nonzero if any run is invalid or failed, if any run meets or exceeds the configured context window, if preflight fails, or if execution is interrupted; completed results remain available.

`ok` means the answer passed format and evidence-ID checks and stayed under the configured context window, not that its diagnosis is correct. Compare the full answers against each scenario's instructor notes. This first version performs one run per pair and does not assign detection-accuracy scores. Timing separates model loading, prompt processing, and generation; wall time includes loading and request overhead. Eval time is prompt processing plus generation for that log and excludes model load and unload. These are observed timings, not a controlled cold/warm performance benchmark. Context allocated is the requested `num_ctx`. Context used is prompt tokens plus generated tokens for that turn. Thinking and output tokens divide that generated count: the split is exact when only one of those texts is present, and estimated from character length when both are present. A run warns when used tokens reach 90% of `num_ctx` and is an error when used tokens are at least `num_ctx`. Context shift is disabled for every model. The expected KV cache type is `f16` (`OLLAMA_KV_CACHE_TYPE` on the Ollama server, recorded on each hunt and not sent as a chat option). A missing cached-token count is unknown, not zero. Identical context settings do not guarantee that different model tokenizers or native context limits see identical effective context; use inputs that fit every selected model.

Run offline checks with:

```bash
uv run python -m unittest -v
```

## Layout

| Path | Role |
| --- | --- |
| `main.py` | Single-hunt CLI: prompt, log loading, one Ollama chat call (think only if supported), output and evidence-ID validation |
| `compare_models.py` | Compare installed Ollama models across the same JSONL scenarios and write a report under `results/` |
| `profiles/*.profile` | Per-model eval setup. Uncommented values may override Ollama and Modelfile defaults for that run |
| `Modelfile.foundation-sec-8b-instruct` | Native `<|system|>/<|user|>/<|assistant|>` template for the Foundation-Sec GGUF import |
| `logs/password-spray.jsonl` | Optional demo: ECS login / password-spray events |
| `logs/http-beaconing.jsonl` | Default demo: ECS HTTP beaconing among legitimate traffic |
| `logs/internal-network-scan.jsonl` | Optional demo: ECS internal scanning among legitimate traffic |
| `logs/shared-vpn-logins.jsonl` | Optional demo: ECS shared-VPN logins that look like spraying |
| `logs/managed-telemetry.jsonl` | Optional demo: ECS managed check-ins that look like beaconing |
| `logs/scheduled-discovery.jsonl` | Optional demo: ECS authorized scanning that looks like an internal scan |
| `test_main.py` | Unit tests for think-arg gating, incomplete replies, and hunt-output validation |
| `test_compare_models.py` | Unit tests for the comparison matrix, HTML report, and the shared hunt runner |
| `pyproject.toml` | Project metadata and the `ollama` client |

Keep the harness thin so the lesson stays in the open.
