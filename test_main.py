from __future__ import annotations

import unittest

from ollama import ChatResponse, Message

from main import incomplete_response_message


NUM_PREDICT = 1024


def make_response(
    content: str | None,
    *,
    done_reason: str | None = None,
    eval_count: int | None = None,
    prompt_eval_count: int | None = None,
    thinking: str | None = None,
) -> ChatResponse:
    return ChatResponse(
        message=Message(role="assistant", content=content, thinking=thinking),
        done_reason=done_reason,
        eval_count=eval_count,
        prompt_eval_count=prompt_eval_count,
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

    def test_prompt_eval_count_at_context_limit(self) -> None:
        response = make_response(
            '{"verdict":"benign","threat_type":"none","summary":"ok","evidence":[]}',
            done_reason="stop",
            eval_count=50,
            prompt_eval_count=32768,
        )
        message = incomplete_response_message(response, NUM_PREDICT, num_ctx=32768)
        self.assertIsNotNone(message)
        self.assertIn("context window", message)
        self.assertIn("prompt_eval_count=32768/32768", message)

    def test_prompt_eval_count_below_context_limit(self) -> None:
        response = make_response(
            '{"verdict":"benign","threat_type":"none","summary":"ok","evidence":[]}',
            done_reason="stop",
            eval_count=50,
            prompt_eval_count=400,
        )
        self.assertIsNone(
            incomplete_response_message(response, NUM_PREDICT, num_ctx=32768)
        )

    def test_length_takes_priority_over_context_overflow(self) -> None:
        response = make_response(
            '{"verdict":"benign","threat_type":"none","summary":"ok","evidence":[]}',
            done_reason="length",
            eval_count=NUM_PREDICT,
            prompt_eval_count=32768,
        )
        message = incomplete_response_message(response, NUM_PREDICT, num_ctx=32768)
        self.assertIsNotNone(message)
        self.assertIn("token limit", message)
        self.assertNotIn("context window", message)


if __name__ == "__main__":
    unittest.main()
