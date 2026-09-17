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

1. Load login events from `logs.jsonl`
2. Sort them by timestamp
3. Send them to a local [Ollama](https://ollama.com) model with a defensive analyst prompt
4. Print a fixed-format answer: verdict, threat type, summary, and evidence IDs

The prompt treats log contents as untrusted evidence, not instructions, and forbids inventing users, IPs, timestamps, or event IDs.

The current script calls `get_security_events()` itself. That function is the obvious next “tool” if you want the model to request data instead of receiving the full dump up front.

## Demo scenario

`logs.jsonl` is synthetic. It mixes ordinary internal logins with a short spray from `198.51.100.7` (a documentation IP) across several accounts, then a success for `bob` from that same source.

Use it to check whether the model:

- names the right pattern (password spray / credential stuffing, then likely account takeover)
- cites real event IDs
- ignores the benign typo-and-retry noise

## Requirements

- Python 3.14+
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com) running locally
- The model pulled once:

```bash
ollama pull hf.co/fdtn-ai/Foundation-Sec-8B-Instruct-Q8_0-GGUF
```

## Run

```bash
uv sync
uv run python main.py
```

## Layout

| Path | Role |
| --- | --- |
| `main.py` | Prompt, log loading, one Ollama chat call |
| `logs.jsonl` | Canned security events for the demo |
| `pyproject.toml` | Project metadata and the `ollama` client |

Swap the model name in `main.py`, edit the prompt, or replace `logs.jsonl` to try a different hunt. Keep the harness thin so the lesson stays in the open.
