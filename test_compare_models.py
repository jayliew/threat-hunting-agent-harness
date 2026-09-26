"""Tests for the comparison runner and the shared hunt contract.

Hunt-contract tests live here because both CLIs share run_hunt() from main.py.
The rest of the file covers the comparison matrix and HTML report.
"""
from __future__ import annotations

import contextlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from ollama import ChatResponse, Message

import compare_models
import main as harness


ANSWER = 'Verdict: benign\nThreat type: none\nSummary: Normal activity.\nEvidence: e1'
EVENTS = [{
    "@timestamp": "2026-09-14T09:45:00.000Z",
    "event": {"id": "e1"},
    "message": "test",
}]


def response(content=ANSWER, reason="stop"):
    return ChatResponse(message=Message(role="assistant", content=content),
                        done_reason=reason, eval_count=40, prompt_eval_count=80,
                        total_duration=2_000_000_000,
                        load_duration=500_000_000, prompt_eval_duration=200_000_000,
                        eval_duration=1_300_000_000)


def client():
    result = Mock()
    result.list.return_value.models = [
        SimpleNamespace(model="alpha:latest", digest="digest-alpha"),
        SimpleNamespace(model="beta:1", digest="digest-beta"),
    ]
    result.show.return_value.capabilities = ["completion"]
    result.show.return_value.template = (
        "<|system|>\n{{ .System }}\n<|user|>\n{{ .Content }}\n<|assistant|>\n"
    )
    result.chat.return_value = response()
    return result


class HuntTests(unittest.TestCase):
    def test_shared_hunt_preserves_request_response_and_timing(self):
        api = client()
        result = harness.run_hunt('alpha:latest', EVENTS, client=api, capabilities=['completion', 'thinking'])
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(result['sections']['Verdict'], 'benign')
        self.assertEqual(result['response']['message']['content'], ANSWER)
        self.assertEqual(result['timing']['load_duration_seconds'], 0.5)
        self.assertEqual(result['timing']['eval_duration_seconds'], 1.3)
        self.assertEqual(api.chat.call_args.kwargs['think'], harness.THINK)
        self.assertFalse(api.chat.call_args.kwargs['shift'])
        self.assertFalse(result['request']['shift'])
        self.assertEqual(len(api.chat.call_args.kwargs['messages']), 2)
        self.assertFalse(api.show.called)
        self.assertEqual(result['tokens']['prompt_eval_count'], 80)
        self.assertEqual(result['tokens']['eval_count'], 40)
        self.assertEqual(result['tokens']['input_tokens'], 80)
        self.assertEqual(result['tokens']['thinking_tokens'], 0)
        self.assertEqual(result['tokens']['output_tokens'], 40)
        self.assertEqual(result['tokens']['split'], 'exact')
        self.assertIsNone(result['tokens']['prompt_eval_cached_count'])
        self.assertEqual(result['context']['allocated'], harness.NUM_CTX)
        self.assertEqual(result['context']['used'], 120)
        self.assertEqual(result['context']['limit'], 'ok')
        self.assertEqual(result['warnings'], [])

    def test_non_thinking_model_omits_think(self):
        api = client()
        harness.run_hunt('alpha:latest', EVENTS, client=api, capabilities=['completion'])
        self.assertNotIn('think', api.chat.call_args.kwargs)
        self.assertFalse(api.chat.call_args.kwargs['shift'])

    def test_client_without_shift_kwarg_still_sends_shift_false(self):
        class FixedChat:
            def chat(self, model='', messages=None, *, stream=False, think=None,
                     options=None, keep_alive=None, tools=None, logprobs=None,
                     top_logprobs=None, format=None):
                raise AssertionError('chat() cannot accept shift')

            def _request(self, cls, method, path, *, json, stream=False):
                self.body = json
                self.method = method
                self.path = path
                self.stream = stream
                return response()

        api = FixedChat()
        result = harness.run_hunt('alpha:latest', EVENTS, client=api, capabilities=['completion'])
        self.assertEqual(result['status'], 'ok')
        self.assertFalse(result['request']['shift'])
        self.assertFalse(api.body['shift'])
        self.assertEqual(api.method, 'POST')
        self.assertEqual(api.path, '/api/chat')
        self.assertFalse(api.stream)
        self.assertEqual(api.body['options']['num_ctx'], harness.NUM_CTX)

    def test_invalid_evidence_and_partial_answer_are_preserved(self):
        api = client()
        api.chat.return_value = response(ANSWER.replace('e1', 'e999'), 'length')
        result = harness.run_hunt('alpha:latest', EVENTS, client=api)
        self.assertEqual(result['status'], 'invalid')
        self.assertEqual(result['unknown_evidence_ids'], ['e999'])
        self.assertEqual(len(result['validation_errors']), 2)
        self.assertIn('e999', result['raw_content'])

    def test_timeout_is_a_recorded_error(self):
        api = client()
        api.chat.side_effect = TimeoutError('request expired')
        result = harness.run_hunt('alpha:latest', EVENTS, client=api)
        self.assertEqual(result['status'], 'error')
        self.assertIn('TimeoutError', result['error'])
        self.assertIsNone(result['response'])
        self.assertGreaterEqual(result['timing']['wall_seconds'], 0)
        self.assertEqual(result['tokens'], harness.empty_tokens())
        self.assertEqual(result['context']['allocated'], harness.NUM_CTX)
        self.assertIsNone(result['context']['used'])

    def test_run_hunt_skips_chat_when_show_template_is_bare(self):
        api = client()
        api.show.return_value.template = "{{ .Prompt }}"
        result = harness.run_hunt('alpha:latest', EVENTS, client=api)
        self.assertEqual(result['status'], 'error')
        self.assertIn("{{ .Prompt }}", result['error'])
        self.assertFalse(api.chat.called)
        self.assertEqual(result['tokens'], harness.empty_tokens())
        self.assertIsNone(result['context']['used'])

    def test_run_hunt_records_native_context_from_show(self):
        api = client()
        api.show.return_value.modelinfo = {"llama.context_length": 131072}
        result = harness.run_hunt('alpha:latest', EVENTS, client=api)
        self.assertEqual(result['context']['model_max'], 131072)

    def test_context_window_full_is_an_error_and_keeps_validation_errors(self):
        api = client()
        chat = response(ANSWER.replace('e1', 'e999'))
        chat.prompt_eval_count = harness.NUM_CTX
        api.chat.return_value = chat
        result = harness.run_hunt(
            'alpha:latest', EVENTS, client=api, capabilities=['completion']
        )
        self.assertEqual(result['status'], 'error')
        self.assertEqual(result['context']['limit'], 'error')
        self.assertIn('Context window full', result['error'])
        self.assertIn(str(harness.NUM_CTX), result['error'])
        self.assertTrue(result['validation_errors'])
        self.assertIn('e999', result['validation_errors'][0])

    def test_context_window_near_full_warns_without_changing_status(self):
        api = client()
        chat = response()
        chat.prompt_eval_count = harness.NUM_CTX - 50
        chat.eval_count = 40
        api.chat.return_value = chat
        result = harness.run_hunt(
            'alpha:latest', EVENTS, client=api, capabilities=['completion']
        )
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(result['context']['used'], harness.NUM_CTX - 10)
        self.assertEqual(result['context']['limit'], 'warn')
        self.assertEqual(len(result['warnings']), 1)
        self.assertIn('Context window nearly full', result['warnings'][0])

    def test_thinking_and_output_tokens_sum_to_eval_count(self):
        api = client()
        chat = response()
        chat.message.thinking = 'abcd'
        api.chat.return_value = chat
        result = harness.run_hunt(
            'alpha:latest', EVENTS, client=api, capabilities=['completion']
        )
        tokens = result['tokens']
        self.assertEqual(tokens['split'], 'estimated')
        self.assertEqual(tokens['thinking_tokens'] + tokens['output_tokens'], tokens['eval_count'])
        self.assertEqual(result['status'], 'ok')

    def test_cached_prompt_tokens_are_recorded_when_present(self):
        api = client()
        chat = response()
        api.chat.return_value = SimpleNamespace(
            message=chat.message,
            done_reason=chat.done_reason,
            eval_count=40,
            prompt_eval_count=80,
            prompt_eval_cached_count=20,
            total_duration=chat.total_duration,
            load_duration=chat.load_duration,
            prompt_eval_duration=chat.prompt_eval_duration,
            eval_duration=chat.eval_duration,
            model_dump=lambda mode="json": chat.model_dump(mode=mode),
        )
        result = harness.run_hunt('alpha:latest', EVENTS, client=api, capabilities=['completion'])
        self.assertEqual(result['tokens']['prompt_eval_cached_count'], 20)
        self.assertEqual(result['tokens']['prompt_uncached_count'], 60)
        self.assertEqual(result['context']['used'], 120)

    def test_hunt_without_usage_fields_records_null_tokens(self):
        api = client()
        api.chat.return_value = ChatResponse(
            message=Message(role="assistant", content=ANSWER), done_reason="stop"
        )
        result = harness.run_hunt('alpha:latest', EVENTS, client=api, capabilities=['completion'])
        self.assertEqual(result['tokens'], harness.empty_tokens())
        self.assertIsNone(result['context']['used'])
        self.assertEqual(result['context']['allocated'], harness.NUM_CTX)

    def test_single_hunt_cli_uses_shared_runner(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'case.jsonl'
            path.write_text(json.dumps(EVENTS[0])+'\n')
            api = client()
            result = harness.run_hunt('alpha', EVENTS, client=api)
            with patch.object(sys, 'argv', ['main.py', str(path), '--model', 'alpha', '--output-dir', str(Path(tmp)/'results'), '--profiles-dir', str(Path(tmp)/'profiles')]), \
                    patch.object(harness, 'Client', return_value=api), \
                    patch.object(harness, 'run_hunt', return_value=result) as run, \
                    contextlib.redirect_stdout(io.StringIO()) as out:
                harness.main()
            self.assertEqual(run.call_args.args[:2], ('alpha:latest', EVENTS))
            self.assertIn(ANSWER, out.getvalue())
            self.assertIn(
                'Tokens: input 80 · thinking 0 · output 40 (exact) · context 120 / '
                f'{harness.NUM_CTX}',
                out.getvalue(),
            )
            result['status'] = 'invalid'
            result['validation_errors'] = ['Invalid output']
            with patch.object(sys, 'argv', ['main.py', str(path), '--output-dir', str(Path(tmp)/'results'), '--profiles-dir', str(Path(tmp)/'profiles')]), \
                    patch.object(harness, 'Client', return_value=api), \
                    patch.object(harness, 'run_hunt', return_value=result), \
                    contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit) as exit:
                harness.main()
            self.assertEqual(exit.exception.code, 1)

    def test_single_hunt_cli_rejects_bad_template_before_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'case.jsonl'
            path.write_text(json.dumps(EVENTS[0])+'\n')
            api = client()
            api.show.return_value.template = "{{ .Prompt }}"
            with patch.object(sys, 'argv', ['main.py', str(path), '--model', 'alpha', '--output-dir', str(Path(tmp)/'results'), '--profiles-dir', str(Path(tmp)/'profiles')]), \
                    patch.object(harness, 'Client', return_value=api), \
                    patch.object(harness, 'run_hunt') as run, \
                    contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()) as err, \
                    self.assertRaises(SystemExit) as exit:
                harness.main()
            self.assertEqual(exit.exception.code, 1)
            self.assertFalse(run.called)
            self.assertFalse(api.chat.called)
            self.assertIn("{{ .Prompt }}", err.getvalue())
            self.assertIn("Preflight failed", err.getvalue())

    def test_single_hunt_cli_rejects_empty_log_before_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'empty.jsonl'
            path.write_text('')
            api = client()
            with patch.object(sys, 'argv', ['main.py', str(path), '--model', 'alpha']), \
                    patch.object(harness, 'Client', return_value=api), \
                    patch.object(harness, 'run_hunt') as run, \
                    contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()) as err, \
                    self.assertRaises(SystemExit) as exit:
                harness.main()
            self.assertEqual(exit.exception.code, 1)
            self.assertFalse(run.called)
            self.assertFalse(api.chat.called)
            self.assertIn("Preflight failed", err.getvalue())
            self.assertIn("log contains no events", err.getvalue())


class ResultsDirectoryNameTests(unittest.TestCase):
    def test_eastern_daylight_sunday_morning(self):
        when = datetime(2026, 9, 20, 13, 28, tzinfo=timezone.utc)
        self.assertEqual(
            compare_models.results_directory_name(when),
            "20-Sep-2026-Sun_09-28am-ET",
        )

    def test_eastern_standard_saturday_evening(self):
        when = datetime(2026, 1, 11, 2, 5, tzinfo=timezone.utc)
        self.assertEqual(
            compare_models.results_directory_name(when),
            "10-Jan-2026-Sat_09-05pm-ET",
        )


class DefaultScenarioLogsTests(unittest.TestCase):
    def test_default_scenario_logs_include_all_six_fixtures(self):
        self.assertEqual(
            compare_models.DEFAULT_SCENARIO_LOGS,
            [
                "logs/password-spray.jsonl",
                "logs/http-beaconing.jsonl",
                "logs/internal-network-scan.jsonl",
                "logs/shared-vpn-logins.ecs.jsonl",
                "logs/managed-telemetry.ecs.jsonl",
                "logs/scheduled-discovery.ecs.jsonl",
            ],
        )


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.logs = []
        for name in ('one', 'two'):
            path = self.root / (name + '.jsonl')
            path.write_text(json.dumps(EVENTS[0])+'\n')
            self.logs.append(str(path))

    def run_quietly(self, api):
        with contextlib.redirect_stdout(io.StringIO()):
            return compare_models.run_comparison(api, ['alpha', 'beta:1'], self.logs, self.root / 'results')

    def test_matrix_runs_sequentially_and_continues_after_failure(self):
        api = client()
        api.chat.side_effect = [response(), TimeoutError('slow'), response(ANSWER.replace('e1', 'e999')), response()]
        directory = self.run_quietly(api)
        rows = [json.loads(s) for s in (directory / 'results.jsonl').read_text().splitlines()]
        self.assertEqual([r['status'] for r in rows], ['ok', 'error', 'invalid', 'ok'])
        self.assertEqual([r['model'] for r in rows], ['alpha:latest']*2 + ['beta:1']*2)
        self.assertEqual([r['case'] for r in rows], self.logs*2)
        self.assertEqual([r['model_digest'] for r in rows], ['digest-alpha']*2 + ['digest-beta']*2)
        requests = [c.kwargs for c in api.chat.call_args_list]
        self.assertEqual([r['keep_alive'] for r in requests], ['5m', 0, '5m', 0])
        self.assertTrue(all(len(r['messages']) == 2 for r in requests))
        self.assertEqual(requests[0]['messages'], requests[2]['messages'])
        self.assertEqual(rows[0]['prompt_sha256'], rows[2]['prompt_sha256'])
        html = (directory / 'report.html').read_text()
        self.assertIn('4 / 4 runs recorded', html)
        self.assertIn('TimeoutError', html)
        self.assertIn('e999', html)
        self.assertIn('color-scheme:dark', html)
        self.assertIn('background:#0f1419', html)
        self.assertTrue((directory / 'manifest.json').exists())

    def test_missing_models_reported_together_without_starting_run(self):
        api = client()
        with self.assertRaises(ValueError) as error:
            compare_models.run_comparison(api, ['missing-one', 'missing-two'], self.logs, self.root/'results')
        self.assertIn('missing-one', str(error.exception))
        self.assertIn('missing-two', str(error.exception))
        self.assertFalse(api.chat.called)
        self.assertFalse((self.root/'results').exists())

    def test_empty_and_malformed_logs_prevent_inference(self):
        empty = self.root/'empty.jsonl'
        empty.write_text('')
        malformed = self.root/'bad.jsonl'
        malformed.write_text('not-json\n')
        api = client()
        with self.assertRaises(ValueError) as error:
            compare_models.prepare_comparison(api, ['alpha'], [str(empty), str(malformed)])
        self.assertIn('empty.jsonl', str(error.exception))
        self.assertIn('bad.jsonl', str(error.exception))
        self.assertFalse(api.chat.called)

    def test_duplicate_aliases_and_paths_run_once(self):
        models, logs = compare_models.prepare_comparison(client(), ['alpha', 'alpha:latest'], self.logs*2)
        self.assertEqual(len(models), 1)
        self.assertEqual(len(logs), 2)
        self.assertIn("<|system|>", models[0]["chat_template"])

    def test_bare_prompt_template_fails_preflight_without_inference(self):
        api = client()
        api.show.return_value.template = "{{ .Prompt }}"
        with self.assertRaises(ValueError) as error:
            compare_models.prepare_comparison(api, ['alpha'], self.logs[:1])
        self.assertIn("{{ .Prompt }}", str(error.exception))
        self.assertFalse(api.chat.called)

    def test_gemma_renderer_passes_and_report_names_renderer(self):
        api = client()
        api.list.return_value.models = [
            SimpleNamespace(model="gemma4:26b", digest="digest-gemma"),
        ]
        api.show.return_value.template = "{{ .Prompt }}"
        api.show.return_value.modelfile = "TEMPLATE {{ .Prompt }}\nRENDERER gemma4\n"
        models, _logs = compare_models.prepare_comparison(api, ['gemma4:26b'], self.logs[:1])
        self.assertEqual(models[0]['renderer'], 'gemma4')
        self.assertEqual(models[0]['chat_template'], '{{ .Prompt }}')
        directory = self.root / 'gemma'
        directory.mkdir()
        manifest = {
            'created_at': 'now',
            'request_settings': {'num_ctx': 32768, 'think': False},
            'models': models,
            'cases': [{'name': 'one.jsonl'}],
        }
        compare_models.write_report(directory, manifest, [])
        html = (directory / 'report.html').read_text()
        self.assertIn('Framing: RENDERER gemma4', html)
        self.assertNotIn('{{ .Prompt }}', html)

    def test_deepseek_markers_pass_preflight(self):
        api = client()
        api.list.return_value.models = [
            SimpleNamespace(model="deepseek-r1:32b", digest="digest-ds"),
        ]
        api.show.return_value.template = "<\uff5cUser\uff5c>{{ .Content }}<\uff5cAssistant\uff5c>"
        api.show.return_value.modelfile = "TEMPLATE \"\"\"deepseek\"\"\"\n"
        models, _logs = compare_models.prepare_comparison(api, ['deepseek-r1:32b'], self.logs[:1])
        self.assertEqual(models[0]['name'], 'deepseek-r1:32b')
        self.assertEqual(models[0]['renderer'], '')
        self.assertFalse(api.chat.called)

    def test_raw_foundation_sec_still_fails_preflight(self):
        api = client()
        api.list.return_value.models = [
            SimpleNamespace(
                model="hf.co/fdtn-ai/Foundation-Sec-8B-Instruct-Q8_0-GGUF:latest",
                digest="digest-raw",
            ),
        ]
        api.show.return_value.template = "{{ .Prompt }}"
        api.show.return_value.modelfile = "TEMPLATE {{ .Prompt }}\n"
        with self.assertRaises(ValueError) as error:
            compare_models.prepare_comparison(
                api,
                ['hf.co/fdtn-ai/Foundation-Sec-8B-Instruct-Q8_0-GGUF:latest'],
                self.logs[:1],
            )
        self.assertIn('Modelfile.foundation-sec-8b-instruct', str(error.exception))
        self.assertFalse(api.chat.called)

    def test_run_hunt_rechecks_template_with_renderer(self):
        api = client()
        allowed = harness.run_hunt(
            'gemma4:26b', EVENTS, client=api,
            capabilities=['completion', 'thinking'],
            chat_template='{{ .Prompt }}',
            renderer='gemma4',
        )
        self.assertEqual(allowed['status'], 'ok')
        self.assertEqual(allowed['renderer'], 'gemma4')
        self.assertTrue(api.chat.called)
        api.chat.reset_mock()
        refused = harness.run_hunt(
            'hf.co/fdtn-ai/Foundation-Sec-8B-Instruct-Q8_0-GGUF:latest',
            EVENTS, client=api, capabilities=['completion'],
            chat_template='{{ .Prompt }}', renderer='gemma4',
        )
        self.assertEqual(refused['status'], 'error')
        self.assertIn('Modelfile.foundation-sec-8b-instruct', refused['error'])
        self.assertFalse(api.chat.called)

    def test_report_escapes_model_output(self):
        api = client()
        api.chat.return_value = response(ANSWER.replace('Normal activity.', '<script>alert("x")</script> & text'))
        directory = self.run_quietly(api)
        html = (directory/'report.html').read_text()
        self.assertNotIn('<script>', html)
        self.assertIn('&lt;script&gt;', html)
        rows = (directory/'results.jsonl').read_text()
        self.assertIn('<script>', rows)

    def test_interrupt_retains_completed_rows_and_report(self):
        api = client()
        def generate(**kwargs):
            if api.chat.call_count == 2:
                saved = list((self.root/'results').glob('*/results.jsonl'))
                self.assertEqual(len(saved), 1)
                self.assertEqual(len(saved[0].read_text().splitlines()), 1)
                self.assertIn('1 / 4 runs recorded', saved[0].with_name('report.html').read_text())
                raise KeyboardInterrupt()
            return response()
        api.chat.side_effect = generate
        with self.assertRaises(KeyboardInterrupt):
            self.run_quietly(api)

    def test_output_directory_uses_readable_eastern_name(self):
        when = datetime(2026, 9, 20, 13, 28, tzinfo=timezone.utc)
        with patch.object(compare_models, "eastern_now", return_value=when):
            first = self.run_quietly(client())
            second = self.run_quietly(client())
        self.assertEqual(first.name, "20-Sep-2026-Sun_09-28am-ET")
        self.assertEqual(second.name, "20-Sep-2026-Sun_09-28am-ET-2")
        self.assertTrue((first / "results.jsonl").exists())

    def test_missing_details_render_as_unknown(self):
        directory = self.run_quietly(client())
        html = (directory / 'report.html').read_text()
        self.assertIn('Quantization: —', html)
        self.assertIn('Thinking: not supported', html)
        self.assertIn(f'used 120 / allocated {harness.NUM_CTX}', html)
        rows = [json.loads(s) for s in (directory / 'results.jsonl').read_text().splitlines()]
        self.assertEqual(rows[0]['tokens']['prompt_eval_count'], 80)
        self.assertEqual(rows[0]['tokens']['eval_count'], 40)
        self.assertEqual(rows[0]['tokens']['input_tokens'], 80)
        self.assertEqual(rows[0]['tokens']['thinking_tokens'], 0)
        self.assertEqual(rows[0]['tokens']['output_tokens'], 40)
        self.assertEqual(rows[0]['tokens']['split'], 'exact')
        self.assertIsNone(rows[0]['tokens']['prompt_eval_cached_count'])
        self.assertEqual(rows[0]['context']['used'], 120)
        self.assertEqual(rows[0]['context']['limit'], 'ok')
        self.assertEqual(rows[0]['warnings'], [])
        self.assertIn('Thinking: 0', html)
        self.assertEqual(rows[0]['context']['allocated'], harness.NUM_CTX)
        self.assertFalse(rows[0]['request']['shift'])
        manifest = json.loads((directory / 'manifest.json').read_text())
        self.assertEqual(manifest['request_settings']['num_ctx'], harness.NUM_CTX)
        self.assertFalse(manifest['request_settings']['shift'])
        self.assertIn('Context shift is disabled for every model.', html)
        self.assertIsNone(manifest['models'][0]['quantization_level'])

    def test_report_includes_model_settings_tokens_and_context(self):
        api = client()
        template = api.show.return_value.template

        def show(name):
            thinking = name.startswith('alpha')
            return SimpleNamespace(
                capabilities=['completion', 'thinking'] if thinking else ['completion'],
                template=template,
                details=SimpleNamespace(
                    quantization_level='Q4_K_M' if thinking else 'Q8_0',
                    parameter_size='32.8B' if thinking else '8B',
                    format='gguf',
                    family='qwen' if thinking else 'llama',
                ),
                modelinfo=(
                    {'qwen3.context_length': 40960} if thinking
                    else {'llama.context_length': 131072}
                ),
            )

        api.show.side_effect = show
        directory = self.run_quietly(api)
        manifest = json.loads((directory / 'manifest.json').read_text())
        by_name = {model['name']: model for model in manifest['models']}
        self.assertEqual(by_name['alpha:latest']['quantization_level'], 'Q4_K_M')
        self.assertEqual(by_name['alpha:latest']['context_length'], 40960)
        self.assertEqual(by_name['beta:1']['quantization_level'], 'Q8_0')
        self.assertEqual(by_name['beta:1']['context_length'], 131072)
        html = (directory / 'report.html').read_text()
        self.assertIn('Q4_K_M · 32.8B · gguf', html)
        self.assertIn('Q8_0 · 8B · gguf', html)
        self.assertIn('Thinking: disabled', html)
        self.assertIn('Thinking: not supported', html)
        self.assertIn(f'used 120 / allocated {harness.NUM_CTX} (model max 40960)', html)
        self.assertIn(f'used 120 / allocated {harness.NUM_CTX} (model max 131072)', html)
        self.assertIn('Input: 80', html)
        self.assertIn('Output: 40', html)
        rows = [json.loads(s) for s in (directory / 'results.jsonl').read_text().splitlines()]
        self.assertEqual(rows[0]['context']['model_max'], 40960)
        self.assertEqual(rows[2]['context']['model_max'], 131072)

    def test_pending_cards_show_settings_without_tokens(self):
        directory = self.root / 'pending'
        directory.mkdir()
        manifest = {
            'created_at': 'now',
            'request_settings': {'num_ctx': 32768, 'num_predict': 1024, 'think': False},
            'models': [{
                'name': 'alpha:latest',
                'capabilities': ['completion', 'thinking'],
                'quantization_level': 'Q4_K_M',
                'parameter_size': '7B',
                'format': 'gguf',
                'context_length': 40960,
            }],
            'cases': [{'name': 'one.jsonl'}],
        }
        compare_models.write_report(directory, manifest, [])
        html = (directory / 'report.html').read_text()
        self.assertIn('Pending', html)
        self.assertIn('used — / allocated 32768', html)
        self.assertIn('model max 40960', html)
        self.assertIn('Quantization: Q4_K_M · 7B · gguf', html)
        self.assertIn('Thinking: disabled', html)
        self.assertIn('Input: —', html)

    def test_report_shows_thinking_tokens_and_context_warning(self):
        directory = self.root / 'think'
        directory.mkdir()
        manifest = {
            'created_at': 'now',
            'request_settings': {'num_ctx': 32768, 'think': True},
            'models': [{'name': 'alpha:latest', 'capabilities': ['completion', 'thinking']}],
            'cases': [{'name': 'one.jsonl'}],
        }
        result = {
            'case': 'one.jsonl',
            'model': 'alpha:latest',
            'status': 'ok',
            'sections': {'Verdict': 'benign'},
            'timing': {},
            'unknown_evidence_ids': [],
            'validation_errors': [],
            'error': None,
            'warnings': ['Context window nearly full: used 30000 / 32768 configured tokens.'],
            'thinking': 'trace',
            'raw_content': ANSWER,
            'request': {'think': True},
            'tokens': {
                'prompt_eval_count': 80,
                'prompt_eval_cached_count': 20,
                'prompt_uncached_count': 60,
                'eval_count': 40,
                'input_tokens': 80,
                'thinking_tokens': 12,
                'output_tokens': 28,
                'split': 'estimated',
            },
            'context': {'allocated': 32768, 'used': 30000, 'model_max': None, 'limit': 'warn'},
        }
        compare_models.write_report(directory, manifest, [result])
        html = (directory / 'report.html').read_text()
        self.assertIn('Thinking: enabled', html)
        self.assertIn('Thinking: 12 (estimated)', html)
        self.assertIn('Output: 28 (estimated)', html)
        self.assertIn('Cached: 20', html)
        self.assertIn('class="warn"', html)
        self.assertIn('Context window nearly full: used 30000 / 32768 configured tokens.', html)
        self.assertIn('<th>Thinking tokens</th>', html)

    def test_cli_exits_nonzero_after_recording_invalid_runs(self):
        api = client()
        api.chat.return_value = response('bad answer')
        with patch.object(compare_models, 'Client', return_value=api), patch.object(sys, 'argv', ['compare_models.py', '--models', 'alpha', '--logs', self.logs[0], '--output-dir', str(self.root/'results')]), contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as exit:
            compare_models.main()
        self.assertEqual(exit.exception.code, 1)
        self.assertEqual(len(list((self.root/'results').glob('*/report.html'))), 1)


class DeclaredProfileTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.profiles = self.root / "profiles"
        self.profiles.mkdir()
        self.log = self.root / "case.jsonl"
        self.log.write_text(json.dumps(EVENTS[0]) + "\n")

    def write_profile(self, name: str, text: str) -> None:
        (self.profiles / name).write_text(text, encoding="utf-8")

    def test_checked_in_qwen_profile_parses(self):
        profile = compare_models.parse_profile_file(
            Path(__file__).resolve().parent / "profiles" / "qwen3-32b.profile"
        )
        self.assertEqual(profile["source"], "profiles/qwen3-32b.profile")
        self.assertEqual(
            profile["fields"],
            {"model": "qwen3:32b", "thinking": "true", "num_ctx": "40,960"},
        )
        self.assertEqual(compare_models.profile_num_ctx(profile), 40960)
        self.assertIs(compare_models.profile_think(profile), True)
        self.assertIn("# weight_precision=Developer Q8_0", profile["text"])
        self.assertIn("# kv_cache=Q8_0", profile["text"])
        self.assertIn("# repeat_penalty=1.0", profile["text"])

    def test_checked_in_profiles_match_the_planned_setups(self):
        expected = {
            "qwen3-32b.profile": ("qwen3:32b", "40,960", 40960, "true", True),
            "llama3.3-70b.profile": ("llama3.3:70b", "16,384", 16384, None, None),
            "granite4.2-30b.profile": ("granite4.2:30b", "65,536", 65536, "high", "high"),
            "deepseek-r1-32b.profile": ("deepseek-r1:32b", "65,536", 65536, "true", True),
            "command-r.profile": ("command-r:latest", "131,072", 131072, None, None),
            "gemma4-31b.profile": ("gemma4:31b", "32,768", 32768, "true", True),
            "mistral-small3.2-24b.profile": (
                "mistral-small3.2:24b",
                "131,072",
                131072,
                None,
                None,
            ),
            "mistral-nemo-12b.profile": ("mistral-nemo:12b", "131,072", 131072, None, None),
            "foundation-sec-8b-instruct.profile": (
                "foundation-sec-8b-instruct",
                "131,072",
                131072,
                None,
                None,
            ),
        }
        root = Path(__file__).resolve().parent / "profiles"
        for name, (model, text, number, thinking, parsed) in expected.items():
            profile = compare_models.parse_profile_file(root / name)
            fields = {"model": model, "num_ctx": text}
            if thinking is not None:
                fields["thinking"] = thinking
            self.assertEqual(profile["fields"], fields)
            self.assertEqual(compare_models.profile_num_ctx(profile), number)
            self.assertEqual(compare_models.profile_think(profile), parsed)
            self.assertIn("# weight_precision=", profile["text"])
            self.assertIn("# repeat_penalty=", profile["text"])
            if thinking is None:
                self.assertIn("# thinking=", profile["text"])
            if name == "granite4.2-30b.profile":
                self.assertIn(
                    "# thinking levels: false, low, medium, high",
                    profile["text"],
                )
        self.assertIsNone(compare_models.quantization_note("Developer Q8_0", "Q8_0"))
        self.assertIsNone(compare_models.quantization_note("Developer QAT Q4_0", "Q4_0"))
        self.assertIsNone(
            compare_models.quantization_note("Existing developer Q8_0", "Q8_0")
        )
        self.assertIn("Q4_K_M", compare_models.quantization_note("Q4_K_M", "Q8_0"))

    def test_comments_and_blank_lines_are_ignored(self):
        profile = compare_models.parse_profile_text(
            "# intended setup\n\nmodel=qwen3:32b\n", "memory"
        )
        self.assertEqual(profile["fields"], {"model": "qwen3:32b"})

    def test_ambiguous_profiles_stop_before_inference(self):
        self.write_profile("think.profile", "model=alpha\ntemperature=0.6\n")
        self.write_profile("direct.profile", "model=alpha:latest\ntemperature=0\n")
        api = client()
        with self.assertRaises(ValueError) as error:
            compare_models.run_comparison(
                api, ["alpha"], [str(self.log)], self.root / "results", self.profiles
            )
        message = str(error.exception)
        self.assertIn("Multiple profiles", message)
        self.assertIn("think.profile", message)
        self.assertIn("direct.profile", message)
        self.assertFalse(api.chat.called)
        self.assertFalse((self.root / "results").exists())

    def test_explicit_profiles_keep_same_model_runs_separate(self):
        self.write_profile("think.profile", "model=alpha\ntemperature=0.6\n")
        self.write_profile("direct.profile", "model=alpha\ntemperature=0\n")
        api = client()
        with contextlib.redirect_stdout(io.StringIO()):
            directory = compare_models.run_comparison(
                api,
                ["alpha"],
                [str(self.log)],
                self.root / "results",
                self.profiles,
                [self.profiles / "think.profile", self.profiles / "direct.profile"],
            )
        rows = [json.loads(line) for line in (directory / "results.jsonl").read_text().splitlines()]
        self.assertEqual([row["model"] for row in rows], ["alpha:latest", "alpha:latest"])
        self.assertEqual(len({row["run_key"] for row in rows}), 2)
        self.assertNotEqual(rows[0]["declared_profile"], rows[1]["declared_profile"])
        html = (directory / "report.html").read_text()
        self.assertIn("temperature=0.6", html)
        self.assertIn("temperature=0", html)
        self.assertEqual(
            sorted(path.name for path in (directory / "declared-profiles").iterdir()),
            ["direct.profile", "think.profile"],
        )

    def test_profile_num_ctx_is_sent_and_allocated(self):
        self.write_profile("alpha.profile", "model=alpha\nnum_ctx=16,384\n")
        api = client()
        with contextlib.redirect_stdout(io.StringIO()):
            directory = compare_models.run_comparison(
                api, ["alpha"], [str(self.log)], self.root / "results", self.profiles
            )
        self.assertEqual(api.chat.call_args.kwargs["options"]["num_ctx"], 16384)
        rows = [json.loads(line) for line in (directory / "results.jsonl").read_text().splitlines()]
        self.assertEqual(rows[0]["request"]["options"]["num_ctx"], 16384)
        self.assertEqual(rows[0]["context"]["allocated"], 16384)
        manifest = json.loads((directory / "manifest.json").read_text())
        self.assertEqual(manifest["models"][0]["num_ctx"], 16384)
        self.assertEqual(manifest["request_settings"]["num_ctx"], harness.NUM_CTX)
        html = (directory / "report.html").read_text()
        self.assertIn("used 120 / allocated 16384", html)

    def test_profile_without_num_ctx_uses_fallback(self):
        self.write_profile("alpha.profile", "model=alpha\n")
        api = client()
        with contextlib.redirect_stdout(io.StringIO()):
            directory = compare_models.run_comparison(
                api, ["alpha"], [str(self.log)], self.root / "results", self.profiles
            )
        self.assertEqual(api.chat.call_args.kwargs["options"]["num_ctx"], harness.NUM_CTX)
        rows = [json.loads(line) for line in (directory / "results.jsonl").read_text().splitlines()]
        self.assertEqual(rows[0]["context"]["allocated"], harness.NUM_CTX)
        manifest = json.loads((directory / "manifest.json").read_text())
        self.assertEqual(manifest["models"][0]["num_ctx"], harness.NUM_CTX)

    def test_profile_thinking_is_sent_before_inference(self):
        self.write_profile("alpha.profile", "model=alpha\nthinking=true\n")
        api = client()
        api.show.return_value.capabilities = ["completion", "thinking"]
        seen = {}

        def generate(**kwargs):
            self.assertIs(kwargs["think"], True)
            seen["before"] = True
            return response()

        api.chat.side_effect = generate
        with contextlib.redirect_stdout(io.StringIO()):
            directory = compare_models.run_comparison(
                api, ["alpha"], [str(self.log)], self.root / "results", self.profiles
            )
        self.assertTrue(seen["before"])
        manifest = json.loads((directory / "manifest.json").read_text())
        self.assertIs(manifest["models"][0]["think"], True)
        html = (directory / "report.html").read_text()
        self.assertIn("Thinking: enabled", html)

    def test_profile_thinking_level_is_sent_before_inference(self):
        self.write_profile("alpha.profile", "model=alpha\nthinking=high\n")
        api = client()
        api.show.return_value.capabilities = ["completion", "thinking"]

        def generate(**kwargs):
            self.assertEqual(kwargs["think"], "high")
            return response()

        api.chat.side_effect = generate
        with contextlib.redirect_stdout(io.StringIO()):
            compare_models.run_comparison(
                api, ["alpha"], [str(self.log)], self.root / "results", self.profiles
            )

    def test_profile_thinking_without_capability_stops_before_chat(self):
        self.write_profile("alpha.profile", "model=alpha\nthinking=true\n")
        api = client()
        with self.assertRaises(ValueError) as error:
            compare_models.run_comparison(
                api, ["alpha"], [str(self.log)], self.root / "results", self.profiles
            )
        self.assertIn("thinking capability", str(error.exception))
        self.assertFalse(api.chat.called)
        self.assertFalse((self.root / "results").exists())

    def test_bad_thinking_stops_before_chat(self):
        self.write_profile("alpha.profile", "model=alpha\nthinking=sometimes\n")
        api = client()
        api.show.return_value.capabilities = ["completion", "thinking"]
        with self.assertRaises(ValueError) as error:
            compare_models.run_comparison(
                api, ["alpha"], [str(self.log)], self.root / "results", self.profiles
            )
        self.assertIn("thinking must be", str(error.exception))
        self.assertFalse(api.chat.called)
        self.assertFalse((self.root / "results").exists())

    def test_bad_num_ctx_stops_before_chat(self):
        self.write_profile("alpha.profile", "model=alpha\nnum_ctx=16,384\u201324,576\n")
        api = client()
        with self.assertRaises(ValueError) as error:
            compare_models.run_comparison(
                api, ["alpha"], [str(self.log)], self.root / "results", self.profiles
            )
        self.assertIn("num_ctx must be a positive integer", str(error.exception))
        self.assertFalse(api.chat.called)
        self.assertFalse((self.root / "results").exists())

    def test_malformed_profile_stops_before_results_and_inference(self):
        self.write_profile("alpha.profile", "weight_quant=Q6_K\n")
        api = client()
        with self.assertRaises(ValueError) as error:
            compare_models.run_comparison(
                api, ["alpha"], [str(self.log)], self.root / "results", self.profiles
            )
        self.assertIn("missing Model", str(error.exception))
        self.assertFalse(api.chat.called)
        self.assertFalse((self.root / "results").exists())

    def test_profile_is_saved_and_shown_before_inference(self):
        self.write_profile(
            "alpha.profile",
            "model=alpha\n"
            "weight_quant=Q6_K\n"
            "temperature=<script>alert(1)</script>\n"
            "kv_cache=f16\n",
        )
        api = client()
        api.show.return_value.details = SimpleNamespace(
            quantization_level="Q4_K_M",
            parameter_size="32B",
            format="gguf",
            family="qwen",
        )
        seen = {}

        def generate(**kwargs):
            saved = list((self.root / "results").glob("*/manifest.json"))
            self.assertEqual(len(saved), 1)
            manifest = json.loads(saved[0].read_text())
            self.assertEqual(manifest["profiles"][0]["fields"]["kv_cache"], "f16")
            self.assertIn("Q4_K_M", manifest["profiles"][0]["quantization_note"])
            report = saved[0].with_name("report.html").read_text()
            self.assertIn("Declared profiles", report)
            self.assertIn("0 / 1 runs recorded", report)
            self.assertNotIn("<script>", report)
            self.assertIn("&lt;script&gt;", report)
            copied = saved[0].parent / "declared-profiles" / "alpha.profile"
            self.assertIn("<script>", copied.read_text())
            seen["before"] = True
            return response()

        api.chat.side_effect = generate
        with contextlib.redirect_stdout(io.StringIO()) as out:
            directory = compare_models.run_comparison(
                api, ["alpha"], [str(self.log)], self.root / "results", self.profiles
            )
        self.assertTrue(seen["before"])
        printed = out.getvalue()
        self.assertLess(printed.index("Declared profile"), printed.index("Results:"))
        self.assertIn("installed quantization is Q4_K_M", printed)
        rows = [json.loads(line) for line in (directory / "results.jsonl").read_text().splitlines()]
        self.assertTrue(rows[0]["declared_profile"].endswith("alpha.profile"))
        self.assertEqual(api.chat.call_args.kwargs["options"]["temperature"], 0)

    def test_missing_profile_is_recorded_as_absent(self):
        api = client()
        with contextlib.redirect_stdout(io.StringIO()) as out:
            directory = compare_models.run_comparison(
                api, ["alpha"], [str(self.log)], self.root / "results", self.profiles
            )
        self.assertIn("No declared profile for alpha:latest", out.getvalue())
        html = (directory / "report.html").read_text()
        self.assertIn("No declared profile", html)
        manifest = json.loads((directory / "manifest.json").read_text())
        self.assertEqual(manifest["profiles"], [])
        rows = [json.loads(line) for line in (directory / "results.jsonl").read_text().splitlines()]
        self.assertIsNone(rows[0]["declared_profile"])
        self.assertFalse((directory / "declared-profiles").exists())

    def test_single_hunt_records_profile_before_analysis(self):
        self.write_profile("alpha.profile", "model=alpha\nkv_cache=f16\n")
        api = client()
        result = harness.run_hunt("alpha", EVENTS, client=api)
        output = self.root / "single-results"
        seen = {}

        def fake_run(*args, **kwargs):
            saved = list(output.glob("*/declared-profiles/alpha.profile"))
            self.assertEqual(len(saved), 1)
            self.assertIn("kv_cache=f16", saved[0].read_text())
            report = saved[0].parents[1] / "report.html"
            self.assertIn("0 / 1 runs recorded", report.read_text())
            seen["before"] = True
            return result

        with patch.object(sys, "argv", [
            "main.py", str(self.log), "--model", "alpha",
            "--profiles-dir", str(self.profiles),
            "--output-dir", str(output),
        ]), patch.object(harness, "Client", return_value=api), \
                patch.object(harness, "run_hunt", side_effect=fake_run), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            harness.main()
        self.assertTrue(seen["before"])
        printed = out.getvalue()
        self.assertLess(printed.index("kv_cache="), printed.index("--- Analysis ---"))
        rows = [
            json.loads(line)
            for path in output.glob("*/results.jsonl")
            for line in path.read_text().splitlines()
        ]
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["declared_profile"].endswith("alpha.profile"))
        self.assertEqual(rows[0]["status"], "ok")

    def test_single_hunt_requires_a_choice_when_settings_differ(self):
        self.write_profile("think.profile", "model=alpha\ntemperature=0.6\nkv_cache=f16\n")
        self.write_profile("direct.profile", "model=alpha\ntemperature=0\nkv_cache=q8_0\n")
        api = client()
        output = self.root / "single-results"
        with patch.object(sys, "argv", [
            "main.py", str(self.log), "--model", "alpha",
            "--profiles-dir", str(self.profiles),
            "--output-dir", str(output),
        ]), patch.object(harness, "Client", return_value=api), \
                patch.object(harness, "run_hunt") as run, \
                contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()) as err, \
                self.assertRaises(SystemExit) as exit:
            harness.main()
        self.assertEqual(exit.exception.code, 1)
        self.assertFalse(run.called)
        self.assertFalse(api.chat.called)
        self.assertIn("think.profile", err.getvalue())
        self.assertIn("direct.profile", err.getvalue())
        self.assertFalse(output.exists())

        result = harness.run_hunt("alpha", EVENTS, client=api)

        def run_with(profile_name):
            with patch.object(sys, "argv", [
                "main.py", str(self.log), "--model", "alpha",
                "--profile", str(self.profiles / profile_name),
                "--profiles-dir", str(self.profiles),
                "--output-dir", str(output),
            ]), patch.object(harness, "Client", return_value=api), \
                    patch.object(harness, "run_hunt", return_value=result), \
                    contextlib.redirect_stdout(io.StringIO()):
                harness.main()

        run_with("think.profile")
        run_with("direct.profile")
        recorded = []
        for path in sorted(output.glob("*/results.jsonl")):
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(len(rows), 1)
            recorded.append(rows[0]["declared_profile"])
            copied = list(path.parent.glob("declared-profiles/*.profile"))
            self.assertEqual(len(copied), 1)
            self.assertTrue(rows[0]["declared_profile"].endswith(copied[0].name))
        self.assertEqual(len(recorded), 2)
        self.assertNotEqual(recorded[0], recorded[1])
        self.assertTrue(any(path.endswith("think.profile") for path in recorded))
        self.assertTrue(any(path.endswith("direct.profile") for path in recorded))

    def test_single_hunt_rejects_bad_profile_before_inference(self):
        self.write_profile("alpha.profile", "not a profile\n")
        api = client()
        with patch.object(sys, "argv", [
            "main.py", str(self.log), "--model", "alpha",
            "--profiles-dir", str(self.profiles),
        ]), patch.object(harness, "Client", return_value=api), \
                patch.object(harness, "run_hunt") as run, \
                contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()) as err, \
                self.assertRaises(SystemExit) as exit:
            harness.main()
        self.assertEqual(exit.exception.code, 1)
        self.assertFalse(run.called)
        self.assertFalse(api.chat.called)
        self.assertIn("expected 'key=value'", err.getvalue())


if __name__ == '__main__':
    unittest.main()
