from __future__ import annotations

import unittest
from pathlib import Path

from ollama import ChatResponse, Message

from main import (
    allowed_evidence_ids,
    incomplete_response_message,
    invalid_hunt_output_message,
    load_security_events,
)


NUM_PREDICT = 1024
REPO_ROOT = Path(__file__).resolve().parent
PASSWORD_SPRAY = REPO_ROOT / "logs" / "password-spray.jsonl"
HTTP_BEACONING = REPO_ROOT / "logs" / "http-beaconing.jsonl"


def make_response(
    content: str | None,
    *,
    done_reason: str | None = None,
    eval_count: int | None = None,
    thinking: str | None = None,
) -> ChatResponse:
    return ChatResponse(
        message=Message(role="assistant", content=content, thinking=thinking),
        done_reason=done_reason,
        eval_count=eval_count,
    )


def hunt_output(
    *,
    verdict: str = "suspicious",
    threat_type: str = "password spray",
    summary: str = "External source sprayed several accounts then succeeded.",
    evidence: str = "e5, e11",
) -> str:
    return (
        f"Verdict: {verdict}\n"
        f"Threat type: {threat_type}\n"
        f"Summary: {summary}\n"
        f"Evidence: {evidence}"
    )


class IncompleteResponseMessageTests(unittest.TestCase):
    def test_length_with_empty_content(self) -> None:
        response = make_response("", done_reason="length", eval_count=1024)
        message = incomplete_response_message(response, NUM_PREDICT)
        self.assertIsNotNone(message)
        self.assertIn("token limit", message)
        self.assertIn("done_reason=length", message)
        self.assertIn("eval_count=1024/1024", message)

    def test_length_with_partial_content(self) -> None:
        response = make_response(
            "Verdict: suspicious",
            done_reason="length",
            eval_count=1024,
        )
        message = incomplete_response_message(response, NUM_PREDICT)
        self.assertIsNotNone(message)
        self.assertIn("token limit", message)

    def test_stop_with_empty_content(self) -> None:
        response = make_response("   ", done_reason="stop", eval_count=12)
        message = incomplete_response_message(response, NUM_PREDICT)
        self.assertIsNotNone(message)
        self.assertIn("empty answer", message)
        self.assertIn("done_reason=stop", message)

    def test_stop_with_normal_content(self) -> None:
        response = make_response(
            "Verdict: suspicious\nThreat type: password spray\n",
            done_reason="stop",
            eval_count=187,
        )
        self.assertIsNone(incomplete_response_message(response, NUM_PREDICT))

    def test_eval_count_at_limit_without_done_reason(self) -> None:
        response = make_response(
            "Verdict: inconclusive",
            done_reason=None,
            eval_count=NUM_PREDICT,
        )
        message = incomplete_response_message(response, NUM_PREDICT)
        self.assertIsNotNone(message)
        self.assertIn("token limit", message)
        self.assertIn("done_reason=unknown", message)


class HuntOutputValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.password_spray_ids = allowed_evidence_ids(
            load_security_events(PASSWORD_SPRAY)
        )
        cls.beaconing_ids = allowed_evidence_ids(load_security_events(HTTP_BEACONING))

    def test_allowed_ids_use_id_from_password_spray(self) -> None:
        self.assertIn("e1", self.password_spray_ids)
        self.assertIn("e14", self.password_spray_ids)
        self.assertNotIn("e999", self.password_spray_ids)
        self.assertNotIn("nonexistent-id", self.password_spray_ids)

    def test_allowed_ids_use_event_id_from_beaconing(self) -> None:
        self.assertIn("e6", self.beaconing_ids)
        self.assertIn("e43", self.beaconing_ids)
        self.assertNotIn("e999", self.beaconing_ids)

    def test_valid_password_spray_sections(self) -> None:
        content = hunt_output(evidence="e5, e11")
        self.assertIsNone(
            invalid_hunt_output_message(content, self.password_spray_ids)
        )

    def test_valid_beaconing_event_ids(self) -> None:
        content = hunt_output(
            threat_type="HTTP beaconing",
            summary="Periodic heartbeats to 203.0.113.77.",
            evidence="e6, e43",
        )
        self.assertIsNone(invalid_hunt_output_message(content, self.beaconing_ids))

    def test_wrapped_summary_is_accepted(self) -> None:
        content = (
            "Verdict: suspicious\n"
            "Threat type: HTTP C2 beaconing\n"
            "Summary: Host ws-014 sent periodic GET /api/heartbeat requests\n"
            "to 203.0.113.77 with low jitter.\n"
            "Evidence: e6, e43"
        )
        self.assertIsNone(invalid_hunt_output_message(content, self.beaconing_ids))

    def test_evidence_none_is_accepted(self) -> None:
        content = hunt_output(verdict="inconclusive", evidence="none")
        self.assertIsNone(
            invalid_hunt_output_message(content, self.password_spray_ids)
        )

    def test_missing_section_is_rejected(self) -> None:
        content = (
            "Verdict: suspicious\n"
            "Threat type: password spray\n"
            "Evidence: e5"
        )
        message = invalid_hunt_output_message(content, self.password_spray_ids)
        self.assertIsNotNone(message)
        self.assertIn("missing required section", message)
        self.assertIn("Summary", message)

    def test_empty_section_is_rejected(self) -> None:
        content = hunt_output(summary="")
        message = invalid_hunt_output_message(content, self.password_spray_ids)
        self.assertIsNotNone(message)
        self.assertIn("empty required section", message)
        self.assertIn("Summary", message)

    def test_illegal_verdict_is_rejected(self) -> None:
        content = hunt_output(verdict="compromised")
        message = invalid_hunt_output_message(content, self.password_spray_ids)
        self.assertIsNotNone(message)
        self.assertIn("verdict", message)
        self.assertIn("compromised", message)

    def test_nonexistent_evidence_id_is_rejected(self) -> None:
        content = hunt_output(evidence="nonexistent-id")
        message = invalid_hunt_output_message(content, self.password_spray_ids)
        self.assertIsNotNone(message)
        self.assertIn("Unknown evidence IDs", message)
        self.assertIn("nonexistent-id", message)

    def test_mixed_unknown_evidence_id_is_rejected(self) -> None:
        content = hunt_output(evidence="e5, nonexistent-id")
        message = invalid_hunt_output_message(content, self.password_spray_ids)
        self.assertIsNotNone(message)
        self.assertIn("nonexistent-id", message)
        self.assertNotIn("e5,", message)

    def test_truncated_valid_sections_report_token_limit_first(self) -> None:
        content = hunt_output()
        response = make_response(content, done_reason="length", eval_count=NUM_PREDICT)
        message = incomplete_response_message(response, NUM_PREDICT)
        self.assertIsNotNone(message)
        self.assertIn("token limit", message)
        self.assertIsNone(
            invalid_hunt_output_message(content, self.password_spray_ids)
        )


if __name__ == "__main__":
    unittest.main()
