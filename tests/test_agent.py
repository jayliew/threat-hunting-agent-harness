"""Scripted checks for the one-search hunt loop. No model is called."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from shared.agent import BAD_SEARCH, run_search_loop, search_events
from shared.harness import load_security_events


PASSWORD_SPRAY = Path(__file__).resolve().parent.parent / "logs" / "password-spray.jsonl"
ANSWER = """
Verdict: suspicious
Threat type: password spraying
Summary: Repeated authentication failures from one source.
Evidence: c04-e005
""".strip()


class SearchTests(unittest.TestCase):
    def test_equality_returns_the_first_five_failures(self) -> None:
        events = load_security_events(PASSWORD_SPRAY)
        hits = json.loads(search_events(events, "SEARCH event.outcome=failure"))
        ids = [hit["event"]["id"] for hit in hits]
        self.assertEqual(
            ids, ["c04-e003", "c04-e005", "c04-e006", "c04-e007", "c04-e008"]
        )

    def test_bad_line_and_unknown_field_are_short_errors(self) -> None:
        events = load_security_events(PASSWORD_SPRAY)
        self.assertEqual(search_events(events, "SEARCH nope"), BAD_SEARCH)
        self.assertEqual(
            search_events(events, "SEARCH missing.field=1"),
            "Unknown field: missing.field",
        )

    def test_known_field_with_no_match_is_an_empty_list(self) -> None:
        events = load_security_events(PASSWORD_SPRAY)
        self.assertEqual(search_events(events, "SEARCH user.name=nobody"), "[]")


class SearchLoopTests(unittest.TestCase):
    def test_scripted_search_then_valid_answer(self) -> None:
        events = load_security_events(PASSWORD_SPRAY)
        replies = iter(["SEARCH event.outcome=failure", ANSWER])
        seen: list[list[dict[str, str]]] = []

        def chat(messages):
            seen.append([dict(message) for message in messages])
            return next(replies)

        result = run_search_loop(events, chat)

        self.assertIsNone(result["validation_error"])
        self.assertEqual(len(seen), 2)
        opening = seen[0][0]["content"] + seen[0][1]["content"]
        self.assertNotIn("password-spray", opening)
        self.assertNotIn("c04-", opening)
        tool = seen[1][-1]
        self.assertEqual(tool["role"], "user")
        ids = [hit["event"]["id"] for hit in json.loads(tool["content"])]
        self.assertIn("c04-e005", ids)
        self.assertEqual(result["messages"][-2]["content"], tool["content"])
        self.assertEqual(result["messages"][-1]["content"], ANSWER)

    def test_bad_search_continues_the_loop(self) -> None:
        events = load_security_events(PASSWORD_SPRAY)
        replies = iter(["SEARCH nope", ANSWER])

        def chat(messages):
            return next(replies)

        result = run_search_loop(events, chat)

        self.assertEqual(result["messages"][-2]["content"], BAD_SEARCH)
        self.assertIsNone(result["validation_error"])
