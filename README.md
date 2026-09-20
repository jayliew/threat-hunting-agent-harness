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

1. Load events from a JSONL file (`logs/password-spray.jsonl` by default)
2. Sort them by timestamp
3. Send them to a local [Ollama](https://ollama.com) model with a defensive analyst prompt
4. Print a fixed-format answer: verdict, threat type, summary, and evidence IDs
5. Reject replies that are incomplete, missing required sections, use an illegal verdict, or cite event IDs that are not in the supplied events

The prompt treats log contents as untrusted evidence, not instructions, and forbids inventing users, IPs, timestamps, or event IDs. Cite `id` or `_event_id`, depending on the file.

The harness then machine-checks that contract: `Verdict` must be `suspicious`, `benign`, or `inconclusive`; all four sections must be present; every cited evidence ID must appear on a supplied event. Valid IDs do not prove the summary or threat type is correct.

The current script calls `get_security_events()` itself. That function is the obvious next “tool” if you want the model to request data instead of receiving the full dump up front.

## Demo scenarios

### Password spray (`logs/password-spray.jsonl`)

Default hunt. Fourteen synthetic [GELF 1.1](https://go2docs.graylog.org/current/getting_in_log_data/gelf_format.html) authentication messages from `auth-01.corp.internal`, one JSON object per line. Timestamps are UNIX seconds in UTC; custom fields are underscore-prefixed. Cite `_event_id` (`e1` … `e14`), not GELF's reserved `_id`.

All events concern password authentication to the same `employee-portal` service. The single suspicious sequence is six failures from `198.51.100.7` (a documentation IP) against six different accounts in 150 seconds (`e5` … `e10`), followed by a success for `bob` from that same source (`e11`). The seven benign events are ordinary internal logins and two isolated failure-then-success pairs (`e3`/`e4` and `e13`/`e14`). The original IDs, times, users, source IPs, and outcomes are preserved.

Ground truth for instructors (not present in the logs): the intended scenario is one password spray; verdict `suspicious`; attack-sequence evidence = `e5` … `e11`. The observable pattern is consistent with spraying and possible compromise. Because the logs do not record attempted passwords or credential provenance, they cannot prove password reuse, distinguish spraying conclusively from credential stuffing, or establish account takeover from the subsequent success alone. No passwords or attack labels are embedded in the events.

Use it to check whether the model:

- recognizes the many-account, low-attempt-count pattern and treats the later success as suspicious
- cites real `_event_id` values and distinguishes observations from hypotheses
- ignores the benign typo-and-retry noise

### HTTP beaconing (`logs/http-beaconing.jsonl`)

Synthetic [GELF 1.1](https://go2docs.graylog.org/current/getting_in_log_data/gelf.html) JSONL (Graylog’s native ingest format). Each line is one firewall/proxy message with underscore-prefixed extra fields. Cite `_event_id` (`e1` … `e43`); GELF reserves `_id`, so it is not used.

The hour of traffic mixes ordinary work with a low-and-slow HTTP check-in:

- `jlee` on `ws-014.corp.internal` (`10.47.12.88`) browses Office, GitHub, and LinkedIn, and syncs Outlook (`outlook.office365.com` / `198.51.100.30`) on an irregular schedule with varying payload sizes.
- The same host also issues `GET /api/heartbeat` to `203.0.113.77:443` (a documentation IP, no hostname) about every 300 seconds with a few seconds of jitter, tiny stable byte counts, and an identical short user-agent. Those twelve events are `e6`, `e11`, `e16`, `e20`, `e23`, `e24`, `e27`, `e31`, `e35`, `e38`, `e41`, and `e43`.
- Cover traffic that can look periodic if you only glance at timestamps: Windows Update from `asmith` / `ws-022` (large, variable bodies), Slack presence polls from `bnguyen` / `ws-008` (~15 minutes apart with high jitter and changing sizes), plus DNS, NTP, and SMB.

Ground truth for instructors (not present in the logs): verdict `suspicious`, threat type HTTP/C2 beaconing, evidence = the twelve `_event_id`s to `203.0.113.77`.

Use it to check whether the model:

- names beaconing (regular interval, low jitter, consistent small payloads, odd destination) rather than “lots of HTTPS”
- cites those real `_event_id` values
- ignores Update, Outlook, and Slack lookalikes

## How to run

You need Python 3.14+, [uv](https://docs.astral.sh/uv/), and [Ollama](https://ollama.com) running locally. This harness does not download weights. It talks to Ollama on your machine and will fail if the chosen model is not installed.

1. Install Ollama and start it. On macOS that is typically `brew install ollama` then `ollama serve` (or open the Ollama app). Confirm it is up and see which models you already have:

```bash
ollama list
```

2. Pick a model from that list. If the list is empty (or you want the suggested defensive model), pull it once (several GB):

```bash
ollama pull hf.co/fdtn-ai/Foundation-Sec-8B-Instruct-Q8_0-GGUF
```

That name is the script default. Any other installed name is fine; pass it with `--model`.

3. From this repo, install Python deps and run a hunt. Default is the password-spray file and the suggested model:

```bash
uv sync
uv run python main.py
```

Pass `--model` with a name from `ollama list`, and optionally another JSONL file:

```bash
uv run python main.py --model qwen3:32b
uv run python main.py logs/http-beaconing.jsonl --model qwen3:32b
```

The script prints the model name, thinking on/off/unsupported, the log file, the events it is sending, then an `--- Analysis ---` block with `Verdict`, `Threat type`, `Summary`, and `Evidence`. It exits with an error instead of looking like a successful hunt when:

- generation hits the token limit (`done_reason=length`) or returns empty content
- a required section is missing or empty
- the verdict is not `suspicious`, `benign`, or `inconclusive`
- a cited evidence ID is not in the supplied events (for example `nonexistent-id`)

Valid citations do not mean the explanation is right.

## Thinking mode

Before you run, set `THINK` in `main.py` to `True` (on) or `False` (off) for models that support thinking. Do not leave the choice implicit.

The harness queries Ollama (`/api/show`) and sends `think` only when the model lists the `thinking` capability. Models without that capability reject the argument (`does not support thinking`), so it is omitted. Qwen3-class models enable thinking by default when the API omits `think`; for those models the harness always sends `think` explicitly.

Thinking tokens and the final answer share the 1,024-token `num_predict` budget; with thinking on, the model can hit that limit mid-trace and return an empty or truncated analysis.

## Layout

| Path | Role |
| --- | --- |
| `main.py` | Prompt, log loading, one Ollama chat call (think only if supported), output and evidence-ID validation |
| `logs/password-spray.jsonl` | Default demo: synthetic login / password-spray events |
| `logs/http-beaconing.jsonl` | Optional demo: GELF 1.1 HTTP beaconing among legitimate traffic |
| `test_main.py` | Unit tests for think-arg gating, incomplete replies, and hunt-output validation |
| `pyproject.toml` | Project metadata and the `ollama` client |

Keep the harness thin so the lesson stays in the open.
