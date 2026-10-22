"""A short hunt loop: one equality search, then the four-field answer.

The model does not receive the events. A reply that starts with ``SEARCH ``
filters the case. Any other reply is the answer.
"""
from __future__ import annotations

import json
import re

from .harness import allowed_evidence_ids, invalid_hunt_output_message


MAX_TURNS = 4
MAX_HITS = 5
BAD_SEARCH = "Bad search. Use SEARCH field=value"

SYSTEM_PROMPT = """
You are a defensive security analyst.
You have not been given the events. To see some, print exactly one line:
SEARCH field=value
Fields you can filter on include source.ip, user.name, and event.outcome.
One search per reply. When you are ready, stop searching and return exactly these four fields:
Verdict: suspicious, benign, or inconclusive
Threat type: specific threat name, or none
Summary: one short paragraph
Evidence: comma-separated event IDs, or none
""".strip()

USER_TASK = "Assess the security events. Search, then return the four fields."

_SEARCH_RE = re.compile(
    r"^SEARCH[ \t]+([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)=(.*\S)\s*$"
)


def _lookup(event: dict, field: str) -> tuple[bool, object]:
    current = event
    for part in field.split("."):
        if not isinstance(current, dict) or part not in current:
            return False, None
        current = current[part]
    return True, current


def search_events(events: list[dict], line: str) -> str:
    """Return up to five matching events as JSON, or one short error."""
    match = _SEARCH_RE.match(line.strip())
    if not match:
        return BAD_SEARCH
    field, expected = match.group(1), match.group(2).strip()
    if not any(_lookup(event, field)[0] for event in events):
        return f"Unknown field: {field}"
    hits = []
    for event in events:
        found, value = _lookup(event, field)
        if found and str(value) == expected:
            hits.append(event)
            if len(hits) == MAX_HITS:
                break
    return json.dumps(hits, separators=(",", ":"))


def _first_line(reply: str) -> str:
    stripped = reply.lstrip()
    if not stripped:
        return ""
    return stripped.splitlines()[0].strip()


def run_search_loop(events: list[dict], chat) -> dict:
    """Ask chat up to four times. A SEARCH line returns hits; anything else is the answer.

    chat(messages) returns the next reply as text. The returned messages include
    each reply and, after a search, the tool result that followed it.
    """
    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TASK},
    ]
    answer = ""
    for turn in range(1, MAX_TURNS + 1):
        reply = chat(messages)
        answer = reply or ""
        line = _first_line(answer)
        print(f"turn {turn}: {line or '(empty)'}")
        messages.append({"role": "assistant", "content": answer})
        if not line.startswith("SEARCH "):
            break
        result = search_events(events, line)
        messages.append({"role": "user", "content": result})
        if result.startswith("["):
            print(f"  {len(json.loads(result))} events")
        else:
            print(f"  {result}")
    error = invalid_hunt_output_message(answer, allowed_evidence_ids(events))
    return {
        "messages": messages,
        "answer": answer,
        "validation_error": error,
    }
