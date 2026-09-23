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
        self.assertEqual(len(api.chat.call_args.kwargs['messages']), 2)
        self.assertFalse(api.show.called)
        self.assertEqual(result['tokens']['prompt_eval_count'], 80)
        self.assertEqual(result['tokens']['eval_count'], 40)
        self.assertIsNone(result['tokens']['prompt_eval_cached_count'])
        self.assertEqual(result['context']['allocated'], harness.NUM_CTX)
        self.assertEqual(result['context']['used'], 120)

    def test_non_thinking_model_omits_think(self):
        api = client()
        harness.run_hunt('alpha:latest', EVENTS, client=api, capabilities=['completion'])
        self.assertNotIn('think', api.chat.call_args.kwargs)

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
            with patch.object(sys, 'argv', ['main.py', str(path), '--model', 'alpha']), \
                    patch.object(harness, 'Client', return_value=api), \
                    patch.object(harness, 'run_hunt', return_value=result) as run, \
                    contextlib.redirect_stdout(io.StringIO()) as out:
                harness.main()
            self.assertEqual(run.call_args.args[:2], ('alpha', EVENTS))
            self.assertIn(ANSWER, out.getvalue())
            result['status'] = 'invalid'
            result['validation_errors'] = ['Invalid output']
            with patch.object(sys, 'argv', ['main.py', str(path)]), \
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
            self.assertIn("{{ .Prompt }}", err.getvalue())


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
                "logs/shared-vpn-logins.jsonl",
                "logs/managed-telemetry.jsonl",
                "logs/scheduled-discovery.jsonl",
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
        self.assertIsNone(rows[0]['tokens']['prompt_eval_cached_count'])
        self.assertEqual(rows[0]['context']['used'], 120)
        self.assertEqual(rows[0]['context']['allocated'], harness.NUM_CTX)
        manifest = json.loads((directory / 'manifest.json').read_text())
        self.assertEqual(manifest['request_settings']['num_ctx'], harness.NUM_CTX)
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

    def test_report_labels_thinking_enabled_and_output_includes_thinking(self):
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
            'thinking': 'trace',
            'raw_content': ANSWER,
            'request': {'think': True},
            'tokens': {
                'prompt_eval_count': 80,
                'prompt_eval_cached_count': 20,
                'prompt_uncached_count': 60,
                'eval_count': 40,
            },
            'context': {'allocated': 32768, 'used': 120, 'model_max': None},
        }
        compare_models.write_report(directory, manifest, [result])
        html = (directory / 'report.html').read_text()
        self.assertIn('Thinking: enabled', html)
        self.assertIn('Output: 40 (includes thinking)', html)
        self.assertIn('Cached: 20', html)

    def test_cli_exits_nonzero_after_recording_invalid_runs(self):
        api = client()
        api.chat.return_value = response('bad answer')
        with patch.object(compare_models, 'Client', return_value=api), patch.object(sys, 'argv', ['compare_models.py', '--models', 'alpha', '--logs', self.logs[0], '--output-dir', str(self.root/'results')]), contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as exit:
            compare_models.main()
        self.assertEqual(exit.exception.code, 1)
        self.assertEqual(len(list((self.root/'results').glob('*/report.html'))), 1)


if __name__ == '__main__':
    unittest.main()
