"""Colibri preflight and chat mapping. No live Colibri or Ollama server."""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import compare_models
import shared.colibri as colibri
import shared.harness as harness


ANSWER = "Verdict: benign\nThreat type: none\nSummary: Normal activity.\nEvidence: e1"
EVENTS = [{
    "@timestamp": "2026-09-14T09:45:00.000Z",
    "event": {"id": "e1"},
    "message": "test",
}]


def ollama_client():
    result = Mock()
    result.list.return_value.models = [
        SimpleNamespace(model="foundation-sec-alpha:latest", digest="digest-alpha"),
    ]
    result.show.return_value.capabilities = ["completion"]
    result.show.return_value.template = (
        "<|system|>\n{{ .System }}\n<|user|>\n{{ .Content }}\n<|assistant|>\n"
    )
    result.show.return_value.parameters = 'stop "<|end_of_text|>"\n'
    result.show.return_value.modelfile = ""
    result.chat.return_value = harness_response()
    return result


def harness_response():
    from ollama import ChatResponse, Message
    return ChatResponse(
        message=Message(role="assistant", content=ANSWER),
        done_reason="stop",
        eval_count=40,
        prompt_eval_count=80,
    )


class FakeBody:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def read(self) -> bytes:
        return self.payload

    def __enter__(self):
        return self

    def __exit__(self, *args) -> bool:
        return False


def completion(
    content: str = ANSWER,
    finish: str = "stop",
    reasoning: str = "checked the beacon interval",
    *,
    prompt_tokens: int = 11,
    completion_tokens: int = 7,
    cached: int = 3,
) -> dict:
    return {
        "choices": [{
            "finish_reason": finish,
            "message": {
                "role": "assistant",
                "content": content,
                "reasoning_content": reasoning,
            },
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "prompt_tokens_details": {"cached_tokens": cached},
        },
    }


def serving(models_by_url: dict[str, list[str]], payload: dict, seen: list):
    def transport(request, timeout):
        seen.append((request, timeout))
        url = request.full_url
        if request.get_method() == "GET":
            ids: list[str] = []
            for base, model_ids in models_by_url.items():
                if url.startswith(base):
                    ids = model_ids
                    break
            body = {"object": "list", "data": [{"id": model_id} for model_id in ids]}
            return FakeBody(json.dumps(body).encode())
        return FakeBody(json.dumps(payload).encode())

    return transport


class ColibriClientTests(unittest.TestCase):
    def test_model_spec_and_base_url(self) -> None:
        spec = colibri.parse_model_spec("colibri/glm-5.2-colibri")
        self.assertEqual(spec, {"provider": "colibri", "name": "glm-5.2-colibri"})
        self.assertIsNone(colibri.parse_model_spec("qwen3:32b")["provider"])
        with self.assertRaises(ValueError):
            colibri.parse_model_spec("colibri/")
        with self.assertRaises(ValueError):
            colibri.parse_base_url("http://user:secret@127.0.0.1:8000/v1", "memory")
        self.assertEqual(
            colibri.parse_base_url("http://127.0.0.1:8001/v1/", "memory"),
            "http://127.0.0.1:8001/v1",
        )

    def test_api_key_is_a_bearer_header(self) -> None:
        seen = []

        def transport(request, timeout):
            seen.append(request)
            return FakeBody(json.dumps({"data": [{"id": "glm-5.2-colibri"}]}).encode())

        client = colibri.ColibriClient(
            "http://127.0.0.1:8000/v1", api_key="local-secret", transport=transport,
        )
        self.assertEqual(client.list_models(), ["glm-5.2-colibri"])
        self.assertEqual(seen[0].get_header("Authorization"), "Bearer local-secret")
        self.assertTrue(seen[0].full_url.endswith("/v1/models"))


class ColibriHuntTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = patch.dict(
            "os.environ", {"COLIBRI_BASE_URL": "", "COLIBRI_API_KEY": ""},
        )
        self.env.start()
        self.addCleanup(self.env.stop)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.configurations = self.root / "inference_config"
        self.configurations.mkdir()
        self.log = self.root / "case.jsonl"
        self.log.write_text(json.dumps(EVENTS[0]) + "\n")
        self.seen: list = []

    def write_conf(self, name: str, text: str) -> None:
        (self.configurations / name).write_text(text, encoding="utf-8")

    def compare(self, models: list[str], payload: dict | None = None, **kwargs):
        served = kwargs.pop("served", {"http://127.0.0.1:8000/v1": ["glm-5.2-colibri"]})
        api = kwargs.pop("api", ollama_client())
        transport = serving(served, payload or completion(), self.seen)
        with contextlib.redirect_stdout(io.StringIO()):
            directory = compare_models.run_comparison(
                api,
                models,
                [str(self.log)],
                self.root / "results",
                self.configurations,
                colibri_transport=transport,
                **kwargs,
            )
        rows = [json.loads(line) for line in (directory / "results.jsonl").read_text().splitlines()]
        return api, directory, rows

    def test_mixed_run_keeps_ollama_and_posts_colibri_chat(self) -> None:
        api, directory, rows = self.compare(
            ["foundation-sec-alpha", "colibri/glm-5.2-colibri"],
            colibri_api_key="local-secret",
        )
        self.assertEqual([row["model"] for row in rows], [
            "foundation-sec-alpha:latest",
            "glm-5.2-colibri",
        ])
        self.assertEqual([row["provider"] for row in rows], ["ollama", "colibri"])
        self.assertEqual(api.chat.call_args.kwargs["options"]["seed"], harness.SEED)
        self.assertFalse(api.chat.call_args.kwargs["shift"])
        shown = [call.args[0] for call in api.show.call_args_list]
        self.assertNotIn("glm-5.2-colibri", shown)
        posts = [request for request, _timeout in self.seen if request.get_method() == "POST"]
        self.assertEqual(len(posts), 1)
        body = json.loads(posts[0].data)
        self.assertEqual(body["model"], "glm-5.2-colibri")
        self.assertEqual(body["max_tokens"], harness.NUM_CTX)
        self.assertFalse(body["enable_thinking"])
        self.assertNotIn("seed", body)
        self.assertNotIn("top_k", body)
        self.assertEqual(posts[0].get_header("Authorization"), "Bearer local-secret")
        coli = rows[1]
        self.assertEqual(coli["raw_content"], ANSWER)
        self.assertEqual(coli["thinking"], "checked the beacon interval")
        self.assertEqual(coli["tokens"]["prompt_eval_count"], 11)
        self.assertEqual(coli["tokens"]["eval_count"], 7)
        self.assertEqual(coli["tokens"]["prompt_eval_cached_count"], 3)
        self.assertFalse(coli["request"]["seed_applied"])
        self.assertEqual(coli["request"]["context_control"], "server")
        self.assertNotIn("options", coli["request"])
        self.assertNotIn("kv_cache_type", coli["request"])
        self.assertNotIn("shift", coli["request"])
        self.assertIsNone(coli["timing"]["evaluation_seconds"])
        self.assertIsNotNone(coli["timing"]["wall_seconds"])
        blob = (directory / "results.jsonl").read_text() + (directory / "manifest.json").read_text()
        blob += (directory / "report.html").read_text()
        self.assertNotIn("local-secret", blob)
        html = (directory / "report.html").read_text()
        self.assertIn("colibri · glm-5.2-colibri", html)
        self.assertIn("ollama · foundation-sec-alpha:latest", html)
        self.assertIn("seed not applied", html)

    def test_missing_colibri_model_fails_before_a_hunt(self) -> None:
        api = ollama_client()
        with self.assertRaises(ValueError) as error:
            self.compare(
                ["colibri/glm-5.2-colibri"],
                api=api,
                served={"http://127.0.0.1:8000/v1": ["other-model"]},
            )
        self.assertIn("is not serving", str(error.exception))
        self.assertFalse(api.chat.called)
        self.assertFalse(api.list.called)
        self.assertFalse(any(request.get_method() == "POST" for request, _timeout in self.seen))
        self.assertFalse((self.root / "results").exists())

    def test_two_colibri_ids_on_one_base_url_fail_preflight(self) -> None:
        api = ollama_client()
        with self.assertRaises(ValueError) as error:
            self.compare(
                ["colibri/glm-5.2-colibri", "colibri/kimi-k3"],
                api=api,
            )
        self.assertIn("One coli serve process serves one model", str(error.exception))
        self.assertFalse(api.list.called)
        self.assertEqual(self.seen, [])
        self.assertFalse((self.root / "results").exists())

    def test_top_k_on_a_colibri_configuration_is_rejected(self) -> None:
        self.write_conf(
            "glm.conf",
            "model=glm-5.2-colibri\nprovider=colibri\ntop_k=20\n",
        )
        api = ollama_client()
        with self.assertRaises(ValueError) as error:
            self.compare(["glm-5.2-colibri"], api=api)
        self.assertIn("top_k", str(error.exception))
        self.assertFalse(api.list.called)
        self.assertEqual(self.seen, [])

    def test_prefix_and_configuration_provider_must_agree(self) -> None:
        self.write_conf(
            "glm.conf",
            "model=glm-5.2-colibri\nprovider=ollama\n",
        )
        with self.assertRaises(ValueError) as error:
            self.compare(["colibri/glm-5.2-colibri"])
        self.assertIn("does not match", str(error.exception))
        self.assertEqual(self.seen, [])

    def test_configuration_base_url_selects_the_server(self) -> None:
        self.write_conf(
            "glm.conf",
            "model=glm-5.2-colibri\n"
            "provider=colibri\n"
            "base_url=http://127.0.0.1:8001/v1\n"
            "thinking=high\n"
            "temperature=0.2\n"
            "top_p=0.9\n"
            "num_ctx=4096\n",
        )
        _api, _directory, rows = self.compare(
            ["glm-5.2-colibri"],
            served={"http://127.0.0.1:8001/v1": ["glm-5.2-colibri"]},
        )
        gets = [request for request, _timeout in self.seen if request.get_method() == "GET"]
        self.assertTrue(gets[0].full_url.startswith("http://127.0.0.1:8001/v1/"))
        body = json.loads(next(
            request.data for request, _timeout in self.seen if request.get_method() == "POST"
        ))
        self.assertEqual(body["max_tokens"], 4096)
        self.assertEqual(body["temperature"], 0.2)
        self.assertEqual(body["top_p"], 0.9)
        self.assertEqual(body["reasoning_effort"], "high")
        self.assertTrue(body["enable_thinking"])
        self.assertNotIn("top_k", body)
        self.assertEqual(rows[0]["request"]["base_url"], "http://127.0.0.1:8001/v1")
        self.assertNotIn("http://127.0.0.1:8000", rows[0]["request"]["base_url"])

    def test_length_finish_is_invalid_and_reasoning_is_kept(self) -> None:
        seen = []
        client = colibri.ColibriClient(
            "http://127.0.0.1:8000/v1",
            transport=serving(
                {"http://127.0.0.1:8000/v1": ["glm-5.2-colibri"]},
                completion(finish="length", reasoning="still thinking"),
                seen,
            ),
        )
        result = harness.run_hunt(
            "glm-5.2-colibri",
            EVENTS,
            provider="colibri",
            colibri_client=client,
            base_url="http://127.0.0.1:8000/v1",
            capabilities=["completion", "thinking"],
        )
        self.assertEqual(result["status"], "invalid")
        self.assertIn("token limit", result["validation_errors"][0])
        self.assertEqual(result["raw_content"], ANSWER)
        self.assertEqual(result["thinking"], "still thinking")
        self.assertIsNone(result["timing"]["prompt_eval_duration_seconds"])
        self.assertIsNone(result["timing"]["eval_duration_seconds"])
