from __future__ import annotations

import json
from pathlib import Path

import pytest
from ollama import ChatResponse, Message
from pydantic import ValidationError

from main import (
    HuntResult,
    HuntResultError,
    allowed_evidence_ids,
    format_hunt_result,
    load_security_events,
    parse_hunt_result,
    validate_evidence_ids,
    validate_model_response,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
PASSWORD_SPRAY = REPO_ROOT / "logs" / "password-spray.jsonl"
HTTP_BEACONING = REPO_ROOT / "logs" / "http-beaconing.jsonl"


def make_response(
    content: str | None,
    *,
    done_reason: str | None = "stop",
    eval_count: int | None = 80,
    prompt_eval_count: int | None = 400,
    thinking: str | None = None,
) -> ChatResponse:
    return ChatResponse(
        message=Message(role="assistant", content=content, thinking=thinking),
        done_reason=done_reason,
        eval_count=eval_count,
        prompt_eval_count=prompt_eval_count,
    )


def valid_payload(
    *,
    verdict: str = "suspicious",
    threat_type: str = "password spray",
    summary: str = "External source sprayed several accounts then succeeded.",
    evidence: list[str] | None = None,
) -> str:
    if evidence is None:
        evidence = ["e5", "e6", "e11"]
    return json.dumps(
        {
            "verdict": verdict,
            "threat_type": threat_type,
            "summary": summary,
            "evidence": evidence,
        }
    )


@pytest.fixture
def password_spray_ids() -> set[str]:
    return allowed_evidence_ids(load_security_events(PASSWORD_SPRAY))


@pytest.fixture
def beaconing_ids() -> set[str]:
    return allowed_evidence_ids(load_security_events(HTTP_BEACONING))


def test_allowed_ids_use_id_from_password_spray(password_spray_ids: set[str]) -> None:
    assert "e1" in password_spray_ids
    assert "e14" in password_spray_ids
    assert "e999" not in password_spray_ids


def test_allowed_ids_use_event_id_from_beaconing(beaconing_ids: set[str]) -> None:
    assert "e6" in beaconing_ids
    assert "e43" in beaconing_ids
    assert "e999" not in beaconing_ids


def test_parse_valid_result(password_spray_ids: set[str]) -> None:
    result = parse_hunt_result(valid_payload())
    validate_evidence_ids(result, password_spray_ids)
    assert result.verdict == "suspicious"
    assert result.evidence == ["e5", "e6", "e11"]


def test_parse_valid_beaconing_ids(beaconing_ids: set[str]) -> None:
    raw = valid_payload(
        threat_type="HTTP beaconing",
        summary="Periodic heartbeats to 203.0.113.77.",
        evidence=["e6", "e11", "e43"],
    )
    result = parse_hunt_result(raw)
    validate_evidence_ids(result, beaconing_ids)
    assert result.evidence == ["e6", "e11", "e43"]


def test_empty_evidence_is_allowed(password_spray_ids: set[str]) -> None:
    result = parse_hunt_result(valid_payload(verdict="inconclusive", evidence=[]))
    validate_evidence_ids(result, password_spray_ids)
    assert format_hunt_result(result).endswith("Evidence: none")


def test_invalid_verdict_is_rejected() -> None:
    raw = valid_payload(verdict="compromised")
    with pytest.raises(ValidationError):
        parse_hunt_result(raw)


def test_missing_field_is_rejected() -> None:
    raw = json.dumps(
        {
            "verdict": "suspicious",
            "threat_type": "password spray",
            "evidence": ["e5"],
        }
    )
    with pytest.raises(ValidationError):
        parse_hunt_result(raw)


def test_unknown_evidence_id_is_rejected(password_spray_ids: set[str]) -> None:
    result = parse_hunt_result(valid_payload(evidence=["e5", "e999"]))
    with pytest.raises(HuntResultError, match="e999"):
        validate_evidence_ids(result, password_spray_ids)


def test_validate_model_response_accepts_valid_json(
    password_spray_ids: set[str],
) -> None:
    response = make_response(valid_payload())
    result = validate_model_response(response, password_spray_ids)
    assert result.verdict == "suspicious"


def test_validate_model_response_rejects_unknown_id(
    password_spray_ids: set[str],
) -> None:
    response = make_response(valid_payload(evidence=["e999"]))
    with pytest.raises(HuntResultError, match="e999"):
        validate_model_response(response, password_spray_ids)


def test_validate_model_response_rejects_schema_mismatch(
    password_spray_ids: set[str],
) -> None:
    response = make_response(valid_payload(verdict="maybe"))
    with pytest.raises(HuntResultError, match="schema"):
        validate_model_response(response, password_spray_ids)


def test_format_hunt_result_uses_section_layout() -> None:
    result = HuntResult(
        verdict="benign",
        threat_type="none",
        summary="Only internal retries.",
        evidence=["e3", "e4"],
    )
    assert format_hunt_result(result) == (
        "Verdict: benign\n"
        "Threat type: none\n"
        "Summary: Only internal retries.\n"
        "Evidence: e3, e4"
    )
