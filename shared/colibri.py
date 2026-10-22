"""Colibri chat client for an already-running `coli serve`.

The harness does not install Colibri or start the server. One process serves
one model id on an OpenAI-compatible `/v1/chat/completions` endpoint.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request


COLIBRI_PREFIX = "colibri/"
DEFAULT_COLIBRI_BASE_URL = "http://127.0.0.1:8000/v1"
# Recorded on the hunt request and not sent. Colibri accepts seed and ignores it.
RECORDED_REQUEST_FIELDS = frozenset({
    "provider",
    "base_url",
    "seed_applied",
    "context_control",
})


class ColibriError(Exception):
    """A Colibri HTTP call failed before a hunt answer was available."""


def parse_model_spec(raw: str) -> dict:
    """Split a CLI model into a provider hint and the name used for lookup.

    `colibri/<model-id>` selects Colibri. Any other string stays an Ollama
    name; a matching inference configuration may still select Colibri.
    """
    if raw.startswith(COLIBRI_PREFIX):
        model_id = raw[len(COLIBRI_PREFIX):]
        if not model_id or model_id.strip() != model_id or "/" in model_id:
            raise ValueError(
                f"Invalid Colibri model spec {raw!r}. "
                "Use colibri/<model-id> with the id `coli serve` publishes."
            )
        return {"provider": "colibri", "name": model_id}
    return {"provider": None, "name": raw}


def parse_base_url(value: str, source: str) -> str:
    """Return a credential-free http(s) base URL without a trailing slash."""
    text = value.strip().rstrip("/")
    parsed = urllib.parse.urlsplit(text)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(
            f"{source}: base_url must be an http(s) URL (got {value!r})"
        )
    if parsed.username or parsed.password:
        raise ValueError(f"{source}: base_url must not include credentials")
    if parsed.query or parsed.fragment:
        raise ValueError(
            f"{source}: base_url must not include a query or fragment"
        )
    return text


def resolve_colibri_base_url(override: str | None) -> str:
    """Flag, then COLIBRI_BASE_URL, then the local coli serve default."""
    if override is not None and override.strip():
        return parse_base_url(override, "--colibri-base-url")
    env = os.environ.get("COLIBRI_BASE_URL", "").strip()
    if env:
        return parse_base_url(env, "COLIBRI_BASE_URL")
    return DEFAULT_COLIBRI_BASE_URL


def resolve_colibri_api_key(override: str | None) -> str | None:
    """Flag, then COLIBRI_API_KEY. Empty means the server is not checking a key."""
    if override is not None and override.strip():
        return override.strip()
    env = os.environ.get("COLIBRI_API_KEY", "").strip()
    return env or None


def wire_body(request: dict) -> dict:
    """Chat-completions JSON. Harness notes stay on the recorded request."""
    return {
        key: value
        for key, value in request.items()
        if key not in RECORDED_REQUEST_FIELDS
    }


class NormalizedMessage:
    def __init__(self, content: str, thinking: str) -> None:
        self.content = content
        self.thinking = thinking


class NormalizedChat:
    """Ollama-shaped view of one Colibri chat completion."""

    def __init__(self, payload: dict) -> None:
        choices = payload.get("choices") if isinstance(payload, dict) else None
        choice = choices[0] if isinstance(choices, list) and choices else {}
        if not isinstance(choice, dict):
            choice = {}
        message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
        content = message.get("content")
        thinking = message.get("reasoning_content")
        self.message = NormalizedMessage(
            content if isinstance(content, str) else "",
            thinking if isinstance(thinking, str) else "",
        )
        finish = choice.get("finish_reason")
        self.done_reason = finish if isinstance(finish, str) else ""
        usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
        self.prompt_eval_count = _optional_int(usage.get("prompt_tokens"))
        self.eval_count = _optional_int(usage.get("completion_tokens"))
        details = usage.get("prompt_tokens_details")
        cached = None
        if isinstance(details, dict):
            cached = _optional_int(details.get("cached_tokens"))
        self.prompt_eval_cached_count = cached
        self.total_duration = None
        self.load_duration = None
        self.prompt_eval_duration = None
        self.eval_duration = None
        self._payload = payload

    def model_dump(self, mode: str = "json") -> dict:
        return self._payload


def _optional_int(value) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


class ColibriClient:
    """Minimal OpenAI-compatible client for one `coli serve` base URL."""

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        timeout: float | None = None,
        transport=None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.transport = transport

    def list_models(self) -> list[str]:
        payload = self._call("GET", "models")
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list):
            raise ColibriError(f"{self.base_url} returned no model list")
        ids = []
        for item in data:
            if isinstance(item, dict) and isinstance(item.get("id"), str) and item["id"]:
                ids.append(item["id"])
        return ids

    def chat(self, body: dict) -> dict:
        payload = self._call("POST", "chat/completions", body)
        if not isinstance(payload, dict):
            raise ColibriError(f"{self.base_url} returned a chat completion that is not an object")
        return payload

    def _call(self, method: str, path: str, body: dict | None = None) -> dict:
        url = self.base_url + "/" + path.lstrip("/")
        data = None if body is None else json.dumps(body).encode()
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if data is not None:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            if self.transport is not None:
                response = self.transport(request, self.timeout)
            else:
                response = urllib.request.urlopen(request, timeout=self.timeout)
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise ColibriError(
                f"HTTP {error.code} from {url}: {detail}"
            ) from error
        except urllib.error.URLError as error:
            raise ColibriError(f"Cannot reach {url}: {error.reason}") from error
        except TimeoutError as error:
            raise ColibriError(f"Timed out calling {url}: {error}") from error
        with response:
            raw = response.read()
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as error:
            raise ColibriError(f"{url} returned non-JSON: {error}") from error
        if not isinstance(payload, dict):
            raise ColibriError(f"{url} returned JSON that is not an object")
        return payload
