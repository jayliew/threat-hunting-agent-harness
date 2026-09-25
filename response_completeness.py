"""Shared checks that an Ollama chat reply finished without premature cutoff.

Used by main.py and compare_models.py (via run_hunt) so truncated or
interrupted final answers cannot pass as successful hunts.
"""
from __future__ import annotations

from ollama import ChatResponse


def incomplete_response_message(
    response: ChatResponse, num_predict: int
) -> str | None:
    """Return an error if generation did not finish a usable final answer."""
    content = (response.message.content or "").strip()
    done_reason = response.done_reason or ""
    eval_count = response.eval_count
    if response.done is False:
        return (
            "Incomplete response: generation ended before completion "
            f"(done=false, done_reason={done_reason or 'unknown'})."
        )
    hit_limit = done_reason == "length" or (
        eval_count is not None and eval_count >= num_predict
    )
    if hit_limit:
        return (
            f"Incomplete response: generation stopped at the token limit "
            f"(done_reason={done_reason or 'unknown'}, "
            f"eval_count={eval_count}/{num_predict})."
        )
    if not content:
        return (
            f"Incomplete response: model returned an empty answer "
            f"(done_reason={done_reason or 'unknown'})."
        )
    return None
