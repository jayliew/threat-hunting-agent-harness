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

1. Load GELF 1.1 events from `logs.jsonl`
2. Sort them by timestamp
3. Send them to a local [Ollama](https://ollama.com) model with a defensive analyst prompt
4. Print a fixed-format answer: verdict, threat type, summary, and `_event_id` values

The prompt treats log contents as untrusted evidence, not instructions, and forbids inventing users, IPs, timestamps, or event IDs.

The current script calls `get_security_events()` itself. That function is the obvious next “tool” if you want the model to request data instead of receiving the full dump up front.

## Demo scenario

`logs.jsonl` is synthetic [GELF 1.1](https://go2docs.graylog.org/current/getting_in_log_data/gelf.html) JSONL (Graylog’s native ingest format). Each line is one firewall/proxy message with underscore-prefixed extra fields. Cite `_event_id` (`e1` … `e43`); GELF reserves `_id`, so it is not used.

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

You need Python 3.14+, [uv](https://docs.astral.sh/uv/), and [Ollama](https://ollama.com) running locally.

1. Install Ollama and start it. On macOS that is typically `brew install ollama` then `ollama serve` (or open the Ollama app). Confirm it is up:

```bash
ollama list
```

2. Pull the local defensive model once (several GB; this is the name `main.py` already uses):

```bash
ollama pull hf.co/fdtn-ai/Foundation-Sec-8B-Instruct-Q8_0-GGUF
```

3. From this repo, install Python deps and run the hunt:

```bash
uv sync
uv run python main.py
```

The script prints the model name, the GELF events it is sending, then an `--- Analysis ---` block with `Verdict`, `Threat type`, `Summary`, and `Evidence` (`_event_id` values).

To use a different Ollama model, change `MODEL` in `main.py` and pull that name instead. Swap `logs.jsonl` to try another hunt.

## Layout

| Path | Role |
| --- | --- |
| `main.py` | Prompt, log loading, one Ollama chat call |
| `logs.jsonl` | Canned GELF 1.1 security events for the demo |
| `pyproject.toml` | Project metadata and the `ollama` client |

Keep the harness thin so the lesson stays in the open.
