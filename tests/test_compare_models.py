"""Tests for the comparison runner and the shared hunt contract.

Hunt-contract tests live here because both CLIs share run_hunt() from shared/harness.py.
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
import main as single_hunt
import shared.harness as harness
import shared.model_config as model_config
import shared.inference_configurations as inference_configurations
import shared.run_reports as run_reports


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
        SimpleNamespace(model="foundation-sec-alpha:latest", digest="digest-alpha"),
        SimpleNamespace(model="foundation-sec-8b-beta:1", digest="digest-beta"),
    ]
    result.show.return_value.capabilities = ["completion"]
    result.show.return_value.template = (
        "<|system|>\n{{ .System }}\n<|user|>\n{{ .Content }}\n<|assistant|>\n"
    )
    result.show.return_value.parameters = 'stop "<|end_of_text|>"\n'
    result.show.return_value.modelfile = ""
    result.chat.return_value = response()
    return result


class HuntTests(unittest.TestCase):
    def test_both_clis_use_a_model_added_through_configuration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "formats.toml"
            config.write_text('''
[[formats]]
label = "Future Instruct"
name_pattern = "future-instruct"
markers = ["<system>", "<user>", "<assistant>"]
parsers = ["future-parser"]
required_stops = ["<end>"]
''')
            formats = model_config.load_model_formats(config)
            log = root / "case.jsonl"
            log.write_text(json.dumps(EVENTS[0]) + '\n')
            for cli, arguments in (
                (single_hunt, ['main.py', str(log), '--model', 'future-instruct']),
                (compare_models, ['compare_models.py', '--models', 'future-instruct', '--logs', str(log)]),
            ):
                with self.subTest(cli=cli.__name__):
                    api = client()
                    api.list.return_value.models = [
                        SimpleNamespace(model='future-instruct:latest', digest='future-digest')
                    ]
                    api.show.return_value.template = '<system>{{ .System }}<user>{{ .Content }}<assistant>'
                    api.show.return_value.modelfile = 'PARSER future-parser\n'
                    api.show.return_value.parameters = 'stop "<end>"\nstop "<turn>"\n'
                    output = root / cli.__name__
                    argv = arguments + ['--output-dir', str(output), '--inference-config-dir', str(root / 'inference_config')]
                    with patch.object(model_config, 'MODEL_FORMATS', formats), \
                            patch.object(cli, 'Client', return_value=api), \
                            patch.object(sys, 'argv', argv), \
                            contextlib.redirect_stdout(io.StringIO()):
                        cli.main()
                    api.chat.assert_called_once()
                    rows = list(output.glob('*/results.jsonl'))
                    self.assertEqual(len(rows), 1)
                    result = json.loads(rows[0].read_text())
                    self.assertEqual(result['model'], 'future-instruct:latest')
                    self.assertEqual(result['status'], 'ok')
                    self.assertTrue(rows[0].with_name('report.html').is_file())

    def test_shared_hunt_preserves_request_response_and_timing(self):
        api = client()
        result = harness.run_hunt('foundation-sec-alpha:latest', EVENTS, client=api, capabilities=['completion', 'thinking'])
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(result['sections']['Verdict'], 'benign')
        self.assertEqual(result['response']['message']['content'], ANSWER)
        self.assertEqual(result['timing']['load_duration_seconds'], 0.5)
        self.assertEqual(result['timing']['prompt_eval_duration_seconds'], 0.2)
        self.assertEqual(result['timing']['eval_duration_seconds'], 1.3)
        self.assertEqual(result['timing']['evaluation_seconds'], 1.5)
        self.assertEqual(api.chat.call_args.kwargs['think'], harness.THINK)
        self.assertEqual(api.chat.call_args.kwargs['options']['seed'], harness.SEED)
        self.assertEqual(result['request']['options']['seed'], harness.SEED)
        self.assertEqual(result['request']['kv_cache_type'], harness.KV_CACHE_TYPE)
        self.assertEqual(harness.KV_CACHE_TYPE, 'f16')
        self.assertNotIn('kv_cache_type', api.chat.call_args.kwargs)
        self.assertFalse(api.chat.call_args.kwargs['shift'])
        self.assertFalse(result['request']['shift'])
        self.assertEqual(len(api.chat.call_args.kwargs['messages']), 2)
        messages = api.chat.call_args.kwargs['messages']
        self.assertIn('suspicious, benign, or inconclusive', messages[0]['content'])
        self.assertIn('suspicious, benign, or inconclusive', messages[1]['content'])
        self.assertIn('A plausible explanation or absence of threat indicators alone is insufficient.',
                      messages[0]['content'])
        self.assertIn('the Evidence field must cite both the observed activity and the records that corroborate',
                      messages[0]['content'])
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
        harness.run_hunt('foundation-sec-alpha:latest', EVENTS, client=api, capabilities=['completion'])
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
        result = harness.run_hunt('foundation-sec-alpha:latest', EVENTS, client=api, capabilities=['completion'])
        self.assertEqual(result['status'], 'ok')
        self.assertFalse(result['request']['shift'])
        self.assertFalse(api.body['shift'])
        self.assertEqual(result['request']['kv_cache_type'], harness.KV_CACHE_TYPE)
        self.assertNotIn('kv_cache_type', api.body)
        self.assertEqual(api.method, 'POST')
        self.assertEqual(api.path, '/api/chat')
        self.assertFalse(api.stream)
        self.assertEqual(api.body['options']['num_ctx'], harness.NUM_CTX)
        self.assertEqual(api.body['options']['num_predict'], -1)
        self.assertEqual(api.body['options']['seed'], 0)
        self.assertEqual(result['request']['options']['seed'], harness.SEED)
        self.assertEqual(result['request']['options']['num_predict'], harness.NUM_PREDICT)

    def test_invalid_evidence_and_partial_answer_are_preserved(self):
        api = client()
        api.chat.return_value = response(ANSWER.replace('e1', 'e999'), 'length')
        result = harness.run_hunt('foundation-sec-alpha:latest', EVENTS, client=api)
        self.assertEqual(result['status'], 'invalid')
        self.assertEqual(result['unknown_evidence_ids'], ['e999'])
        self.assertEqual(len(result['validation_errors']), 2)
        self.assertIn('e999', result['raw_content'])

    def test_timeout_is_a_recorded_error(self):
        api = client()
        api.chat.side_effect = TimeoutError('request expired')
        result = harness.run_hunt('foundation-sec-alpha:latest', EVENTS, client=api)
        self.assertEqual(result['status'], 'error')
        self.assertIn('TimeoutError', result['error'])
        self.assertIsNone(result['response'])
        self.assertGreaterEqual(result['timing']['wall_seconds'], 0)
        self.assertIsNone(result['timing']['evaluation_seconds'])
        self.assertEqual(result['tokens'], harness.empty_tokens())
        self.assertEqual(result['context']['allocated'], harness.NUM_CTX)
        self.assertIsNone(result['context']['used'])

    def test_run_hunt_skips_chat_when_show_template_is_bare(self):
        api = client()
        api.show.return_value.template = "{{ .Prompt }}"
        result = harness.run_hunt('foundation-sec-alpha:latest', EVENTS, client=api)
        self.assertEqual(result['status'], 'error')
        self.assertIn("{{ .Prompt }}", result['error'])
        self.assertFalse(api.chat.called)
        self.assertIsNone(result['timing']['evaluation_seconds'])
        self.assertEqual(result['tokens'], harness.empty_tokens())
        self.assertIsNone(result['context']['used'])

    def test_foundation_configuration_blocks_both_entry_points(self):
        for modelfile, parameters, expected in (
            ("PARSER llama3\n", 'stop "<|end_of_text|>"', "without a PARSER"),
            ("", 'stop "<|eot_id|>"', "must use only stop"),
            ("", "", "must use only stop"),
        ):
            with self.subTest(modelfile=modelfile, parameters=parameters):
                api = client()
                api.show.return_value.modelfile = modelfile
                api.show.return_value.parameters = parameters
                with self.assertRaisesRegex(ValueError, expected):
                    compare_models.prepare_comparison(
                        api, ['foundation-sec-alpha'], ['logs/http-beaconing.jsonl']
                    )
                result = harness.run_hunt('foundation-sec-alpha:latest', EVENTS, client=api)
                self.assertEqual(result['status'], 'error')
                self.assertIn(expected, result['error'])
                api.chat.assert_not_called()

    def test_run_hunt_records_native_context_from_show(self):
        api = client()
        api.show.return_value.modelinfo = {"llama.context_length": 131072}
        result = harness.run_hunt('foundation-sec-alpha:latest', EVENTS, client=api)
        self.assertEqual(result['context']['model_max'], 131072)

    def test_context_window_full_is_an_error_and_keeps_validation_errors(self):
        api = client()
        chat = response(ANSWER.replace('e1', 'e999'))
        chat.prompt_eval_count = harness.NUM_CTX
        api.chat.return_value = chat
        result = harness.run_hunt(
            'foundation-sec-alpha:latest', EVENTS, client=api, capabilities=['completion']
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
            'foundation-sec-alpha:latest', EVENTS, client=api, capabilities=['completion']
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
            'foundation-sec-alpha:latest', EVENTS, client=api, capabilities=['completion']
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
        result = harness.run_hunt('foundation-sec-alpha:latest', EVENTS, client=api, capabilities=['completion'])
        self.assertEqual(result['tokens']['prompt_eval_cached_count'], 20)
        self.assertEqual(result['tokens']['prompt_uncached_count'], 60)
        self.assertEqual(result['context']['used'], 120)

    def test_hunt_without_usage_fields_records_null_tokens(self):
        api = client()
        api.chat.return_value = ChatResponse(
            message=Message(role="assistant", content=ANSWER), done_reason="stop"
        )
        result = harness.run_hunt('foundation-sec-alpha:latest', EVENTS, client=api, capabilities=['completion'])
        self.assertEqual(result['tokens'], harness.empty_tokens())
        self.assertIsNone(result['timing']['evaluation_seconds'])
        self.assertIsNone(result['context']['used'])
        self.assertEqual(result['context']['allocated'], harness.NUM_CTX)

    def test_single_hunt_cli_uses_shared_runner(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'case.jsonl'
            path.write_text(json.dumps(EVENTS[0])+'\n')
            api = client()
            result = harness.run_hunt('foundation-sec-alpha', EVENTS, client=api)
            with patch.object(sys, 'argv', ['main.py', str(path), '--model', 'foundation-sec-alpha', '--output-dir', str(Path(tmp)/'results'), '--inference-config-dir', str(Path(tmp)/'inference_config')]), \
                    patch.object(single_hunt, 'Client', return_value=api), \
                    patch.object(single_hunt, 'run_hunt', return_value=result) as run, \
                    contextlib.redirect_stdout(io.StringIO()) as out:
                single_hunt.main()
            self.assertEqual(run.call_args.args[:2], ('foundation-sec-alpha:latest', EVENTS))
            self.assertIn(ANSWER, out.getvalue())
            self.assertIn(
                'Tokens: input 80 · output 40 (exact) · context 120 / '
                f'{harness.NUM_CTX}',
                out.getvalue(),
            )
            self.assertIn(
                f"{result['status']} · {run_reports.seconds(result['timing']['evaluation_seconds'])}",
                out.getvalue(),
            )
            self.assertEqual(result['timing']['evaluation_seconds'], 1.5)
            result['status'] = 'invalid'
            result['validation_errors'] = ['Invalid output']
            with patch.object(sys, 'argv', ['main.py', str(path), '--output-dir', str(Path(tmp)/'results'), '--inference-config-dir', str(Path(tmp)/'inference_config')]), \
                    patch.object(single_hunt, 'Client', return_value=api), \
                    patch.object(single_hunt, 'run_hunt', return_value=result), \
                    contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit) as exit:
                single_hunt.main()
            self.assertEqual(exit.exception.code, 1)

    def test_single_hunt_cli_defaults_to_no_timeout_and_forwards_seconds(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'case.jsonl'
            path.write_text(json.dumps(EVENTS[0])+'\n')
            api = client()
            result = harness.run_hunt('foundation-sec-alpha', EVENTS, client=api)
            base = ['main.py', str(path), '--model', 'foundation-sec-alpha', '--output-dir', str(Path(tmp)/'results'), '--inference-config-dir', str(Path(tmp)/'inference_config')]
            with patch.object(sys, 'argv', base), \
                    patch.object(single_hunt, 'Client', return_value=api) as constructed, \
                    patch.object(single_hunt, 'run_hunt', return_value=result), \
                    contextlib.redirect_stdout(io.StringIO()):
                single_hunt.main()
            constructed.assert_called_once_with(timeout=None)
            with patch.object(sys, 'argv', base + ['--timeout', '45']), \
                    patch.object(single_hunt, 'Client', return_value=api) as constructed, \
                    patch.object(single_hunt, 'run_hunt', return_value=result), \
                    contextlib.redirect_stdout(io.StringIO()):
                single_hunt.main()
            constructed.assert_called_once_with(timeout=45.0)

    def test_single_hunt_cli_rejects_bad_template_before_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'case.jsonl'
            path.write_text(json.dumps(EVENTS[0])+'\n')
            api = client()
            api.show.return_value.template = "{{ .Prompt }}"
            with patch.object(sys, 'argv', ['main.py', str(path), '--model', 'foundation-sec-alpha', '--output-dir', str(Path(tmp)/'results'), '--inference-config-dir', str(Path(tmp)/'inference_config')]), \
                    patch.object(single_hunt, 'Client', return_value=api), \
                    patch.object(single_hunt, 'run_hunt') as run, \
                    contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()) as err, \
                    self.assertRaises(SystemExit) as exit:
                single_hunt.main()
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
            with patch.object(sys, 'argv', ['main.py', str(path), '--model', 'foundation-sec-alpha']), \
                    patch.object(single_hunt, 'Client', return_value=api), \
                    patch.object(single_hunt, 'run_hunt') as run, \
                    contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()) as err, \
                    self.assertRaises(SystemExit) as exit:
                single_hunt.main()
            self.assertEqual(exit.exception.code, 1)
            self.assertFalse(run.called)
            self.assertFalse(api.chat.called)
            self.assertIn("Preflight failed", err.getvalue())
            self.assertIn("log contains no events", err.getvalue())


class DurationFormatTests(unittest.TestCase):
    def test_displays_minutes_and_seconds(self):
        self.assertEqual(run_reports.seconds(None), "—")
        self.assertEqual(run_reports.seconds(1.5), "0m 1.50s")
        self.assertEqual(run_reports.seconds(90.5), "1m 30.50s")
        self.assertEqual(run_reports.seconds(125.2), "2m 5.20s")

    def test_duration_seconds_one_decimal_rounds_up(self):
        self.assertEqual(run_reports.duration_seconds_one_decimal(None), "—")
        self.assertEqual(run_reports.duration_seconds_one_decimal(0.2), "0.2s")
        self.assertEqual(run_reports.duration_seconds_one_decimal(1.3), "1.3s")
        self.assertEqual(run_reports.duration_seconds_one_decimal(1.01), "1.1s")
        self.assertEqual(run_reports.duration_seconds_one_decimal(1.001), "1.1s")

    def test_collapsible_run_details_default_closed(self):
        html = run_reports.collapsible_run_details("line one<br>line two")
        self.assertIn('<details class="run-details">', html)
        self.assertIn("<summary>Show run details</summary>", html)
        self.assertNotIn('<details class="run-details" open', html)
        self.assertIn("line one<br>line two", html)


class ContextUsedCellTests(unittest.TestCase):
    def test_includes_percent_of_allocated(self):
        self.assertEqual(run_reports.context_used_cell(None, 32768), "—")
        self.assertEqual(run_reports.context_used_cell(120, None), "120")
        self.assertEqual(run_reports.context_used_cell(120, 0), "120")
        self.assertEqual(run_reports.context_used_cell(120, 32768), "120 (0%)")
        self.assertEqual(run_reports.context_used_cell(30000, 32768), "30000 (92%)")
        self.assertEqual(run_reports.context_used_cell(32768, 32768), "32768 (100%)")


class ResultsDirectoryNameTests(unittest.TestCase):
    def test_eastern_daylight_sunday_morning(self):
        when = datetime(2026, 9, 20, 13, 28, tzinfo=timezone.utc)
        self.assertEqual(
            run_reports.results_directory_name(when),
            "20-Sep-2026-Sun_09-28am-ET",
        )

    def test_eastern_standard_saturday_evening(self):
        when = datetime(2026, 1, 11, 2, 5, tzinfo=timezone.utc)
        self.assertEqual(
            run_reports.results_directory_name(when),
            "10-Jan-2026-Sat_09-05pm-ET",
        )


class ReportTimestampTests(unittest.TestCase):
    def test_eastern_daylight_sunday_morning(self):
        self.assertEqual(
            run_reports.format_report_timestamp("2026-09-20T13:28:00+00:00"),
            "Sunday, September 20, 2026, 9:28 AM ET",
        )

    def test_eastern_standard_saturday_evening(self):
        self.assertEqual(
            run_reports.format_report_timestamp("2026-01-11T02:05:00Z"),
            "Saturday, January 10, 2026, 9:05 PM ET",
        )

    def test_naive_timestamp_is_read_as_eastern(self):
        self.assertEqual(
            run_reports.format_report_timestamp("2026-09-20T09:28:00"),
            "Sunday, September 20, 2026, 9:28 AM ET",
        )

    def test_unparsed_value_is_left_unchanged(self):
        self.assertEqual(run_reports.format_report_timestamp("now"), "now")


class DefaultScenarioLogsTests(unittest.TestCase):
    def test_default_scenario_logs_include_all_seven_fixtures(self):
        self.assertEqual(
            compare_models.DEFAULT_SCENARIO_LOGS,
            [
                "logs/password-spray.jsonl",
                "logs/http-beaconing.jsonl",
                "logs/internal-network-scan.jsonl",
                "logs/shared-vpn-logins.jsonl",
                "logs/managed-telemetry.jsonl",
                "logs/scheduled-discovery.jsonl",
                "logs/opaque-sync-transfers.jsonl",
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
            return compare_models.run_comparison(api, ['foundation-sec-alpha', 'foundation-sec-8b-beta:1'], self.logs, self.root / 'results')

    def test_matrix_runs_sequentially_and_continues_after_failure(self):
        api = client()
        api.chat.side_effect = [response(), TimeoutError('slow'), response(ANSWER.replace('e1', 'e999')), response()]
        with contextlib.redirect_stdout(io.StringIO()) as out:
            directory = compare_models.run_comparison(
                api, ['foundation-sec-alpha', 'foundation-sec-8b-beta:1'], self.logs, self.root / 'results'
            )
        printed = out.getvalue()
        self.assertIn('  ok · 0m 1.50s', printed)
        self.assertIn('  error · —', printed)
        self.assertIn('  invalid · 0m 1.50s', printed)
        rows = [json.loads(s) for s in (directory / 'results.jsonl').read_text().splitlines()]
        self.assertEqual([r['status'] for r in rows], ['ok', 'error', 'invalid', 'ok'])
        self.assertEqual(
            [r['timing']['evaluation_seconds'] for r in rows],
            [1.5, None, 1.5, 1.5],
        )
        self.assertEqual(rows[0]['timing']['load_duration_seconds'], 0.5)
        self.assertEqual([r['model'] for r in rows], ['foundation-sec-alpha:latest']*2 + ['foundation-sec-8b-beta:1']*2)
        self.assertEqual([r['case'] for r in rows], self.logs*2)
        self.assertEqual([r['model_digest'] for r in rows], ['digest-alpha']*2 + ['digest-beta']*2)
        requests = [c.kwargs for c in api.chat.call_args_list]
        self.assertEqual([r['keep_alive'] for r in requests], ['5m', 0, '5m', 0])
        self.assertTrue(all(len(r['messages']) == 2 for r in requests))
        self.assertEqual(requests[0]['messages'], requests[2]['messages'])
        self.assertEqual(rows[0]['prompt_sha256'], rows[2]['prompt_sha256'])
        html = (directory / 'report.html').read_text()
        self.assertIn('4 / 4 runs recorded', html)
        self.assertIn('<th>Eval time</th>', html)
        self.assertIn('excludes model load and unload', html)
        self.assertIn('ok · 0m 1.50s', html)
        self.assertIn('TimeoutError', html)
        self.assertIn('e999', html)
        self.assertIn('color-scheme:dark', html)
        self.assertIn('background:#0f1419', html)
        self.assertNotIn('#3dd68c', html)
        self.assertIn('.invalid,.error{color:#f07178}', html)
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
            compare_models.prepare_comparison(api, ['foundation-sec-alpha'], [str(empty), str(malformed)])
        self.assertIn('empty.jsonl', str(error.exception))
        self.assertIn('bad.jsonl', str(error.exception))
        self.assertFalse(api.chat.called)

    def test_duplicate_aliases_and_paths_run_once(self):
        models, logs = compare_models.prepare_comparison(client(), ['foundation-sec-alpha', 'foundation-sec-alpha:latest'], self.logs*2)
        self.assertEqual(len(models), 1)
        self.assertEqual(len(logs), 2)
        self.assertIn("<|system|>", models[0]["chat_template"])

    def test_bare_prompt_template_fails_preflight_without_inference(self):
        api = client()
        api.show.return_value.template = "{{ .Prompt }}"
        with self.assertRaises(ValueError) as error:
            compare_models.prepare_comparison(api, ['foundation-sec-alpha'], self.logs[:1])
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
        run_reports.write_report(directory, manifest, [])
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
        self.assertIn('modelfiles/Modelfile.foundation-sec-8b-instruct', str(error.exception))
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
        self.assertIn('modelfiles/Modelfile.foundation-sec-8b-instruct', refused['error'])
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
        with patch.object(run_reports, "eastern_now", return_value=when):
            first = self.run_quietly(client())
            second = self.run_quietly(client())
        self.assertEqual(first.name, "20-Sep-2026-Sun_09-28am-ET")
        self.assertEqual(second.name, "20-Sep-2026-Sun_09-28am-ET-2")
        self.assertTrue((first / "results.jsonl").exists())
        self.assertIn(
            "Sunday, September 20, 2026, 9:28 AM ET",
            (first / "report.html").read_text(),
        )

    def test_missing_details_render_as_unknown(self):
        directory = self.run_quietly(client())
        html = (directory / 'report.html').read_text()
        self.assertIn('Quantization: —', html)
        self.assertIn('Thinking: not supported', html)
        self.assertIn(f'used 120 / allocated {harness.NUM_CTX}', html)
        self.assertIn(run_reports.context_used_cell(120, harness.NUM_CTX), html)
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
        self.assertNotIn('Thinking: 0', html)
        self.assertIn('Input: 80', html)
        self.assertIn('Output: 40', html)
        self.assertEqual(rows[0]['context']['allocated'], harness.NUM_CTX)
        self.assertFalse(rows[0]['request']['shift'])
        manifest = json.loads((directory / 'manifest.json').read_text())
        self.assertEqual(manifest['request_settings']['num_ctx'], harness.NUM_CTX)
        self.assertEqual(manifest['request_settings']['seed'], harness.SEED)
        self.assertEqual(manifest['request_settings']['kv_cache_type'], harness.KV_CACHE_TYPE)
        self.assertFalse(manifest['request_settings']['shift'])
        self.assertIn('seed=0', html)
        self.assertIn('Context shift is disabled for every model.', html)
        self.assertIsNone(manifest['models'][0]['quantization_level'])
        self.assertIn(f'kv_cache_type={harness.KV_CACHE_TYPE}', html)

    def test_report_includes_model_settings_tokens_and_context(self):
        api = client()
        template = api.show.return_value.template

        def show(name):
            thinking = name.startswith('foundation-sec-alpha')
            return SimpleNamespace(
                capabilities=['completion', 'thinking'] if thinking else ['completion'],
                template=template,
                parameters='stop "<|end_of_text|>"\n',
                modelfile="",
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
        self.assertEqual(by_name['foundation-sec-alpha:latest']['quantization_level'], 'Q4_K_M')
        self.assertEqual(by_name['foundation-sec-alpha:latest']['context_length'], 40960)
        self.assertEqual(by_name['foundation-sec-8b-beta:1']['quantization_level'], 'Q8_0')
        self.assertEqual(by_name['foundation-sec-8b-beta:1']['context_length'], 131072)
        html = (directory / 'report.html').read_text()
        self.assertIn('Q4_K_M · 32.8B', html)
        self.assertIn('Q8_0 · 8B', html)
        self.assertNotIn('gguf', html)
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
                'name': 'foundation-sec-alpha:latest',
                'capabilities': ['completion', 'thinking'],
                'quantization_level': 'Q4_K_M',
                'parameter_size': '7B',
                'format': 'gguf',
                'context_length': 40960,
            }],
            'cases': [{'name': 'one.jsonl'}],
        }
        run_reports.write_report(directory, manifest, [])
        html = (directory / 'report.html').read_text()
        self.assertIn('Pending', html)
        self.assertIn('used — / allocated 32768', html)
        self.assertIn('model max 40960', html)
        self.assertIn('Quantization: Q4_K_M · 7B', html)
        self.assertNotIn('gguf', html)
        self.assertIn('Thinking: disabled', html)
        self.assertIn('Input: —', html)

    def test_report_shows_unlimited_num_predict(self):
        directory = self.root / 'unlimited'
        directory.mkdir()
        manifest = {
            'created_at': 'now',
            'request_settings': {
                'num_ctx': 32768,
                'num_predict': -1,
                'think': False,
                'kv_cache_type': 'f16',
            },
            'models': [],
            'cases': [],
        }
        run_reports.write_report(directory, manifest, [])
        html = (directory / 'report.html').read_text()
        self.assertIn('num_predict=-1 (no limit)', html)
        self.assertIn(
            'These are the harness fallbacks unless an inference config overrides them',
            html,
        )
        self.assertIn('not necessarily what each model was sent', html)

    def test_report_shows_thinking_tokens_and_context_warning(self):
        directory = self.root / 'think'
        directory.mkdir()
        manifest = {
            'created_at': 'now',
            'request_settings': {'num_ctx': 32768, 'think': True},
            'models': [{'name': 'foundation-sec-alpha:latest', 'capabilities': ['completion', 'thinking']}],
            'cases': [{'name': 'one.jsonl'}],
        }
        result = {
            'case': 'one.jsonl',
            'model': 'foundation-sec-alpha:latest',
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
        run_reports.write_report(directory, manifest, [result])
        html = (directory / 'report.html').read_text()
        self.assertIn('<summary>Show run details</summary>', html)
        self.assertIn('Thinking: enabled', html)
        self.assertIn('Thinking: 12 (estimated)', html)
        self.assertIn('Output: 28 (estimated)', html)
        self.assertNotIn('Cached:', html)
        self.assertNotIn('Uncached:', html)
        self.assertNotIn('Cached tokens', html)
        self.assertNotIn('cached-token', html)
        self.assertIn('class="warn"', html)
        self.assertIn('Context window nearly full: used 30000 / 32768 configured tokens.', html)
        self.assertIn('<th>Thinking tokens</th>', html)

    def test_cli_exits_nonzero_after_recording_invalid_runs(self):
        api = client()
        api.chat.return_value = response('bad answer')
        with patch.object(compare_models, 'Client', return_value=api), patch.object(sys, 'argv', ['compare_models.py', '--models', 'foundation-sec-alpha', '--logs', self.logs[0], '--output-dir', str(self.root/'results')]), contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as exit:
            compare_models.main()
        self.assertEqual(exit.exception.code, 1)
        self.assertEqual(len(list((self.root/'results').glob('*/report.html'))), 1)

    def test_compare_cli_defaults_to_no_timeout_and_forwards_seconds(self):
        api = client()
        argv = ['compare_models.py', '--models', 'foundation-sec-alpha', '--logs', self.logs[0], '--output-dir', str(self.root/'results'), '--inference-config-dir', str(self.root/'inference_config')]
        with patch.object(compare_models, 'Client', return_value=api) as constructed, \
                patch.object(sys, 'argv', argv), \
                contextlib.redirect_stdout(io.StringIO()):
            compare_models.main()
        constructed.assert_called_once_with(timeout=None)
        with patch.object(compare_models, 'Client', return_value=api) as constructed, \
                patch.object(sys, 'argv', argv + ['--timeout', '45']), \
                contextlib.redirect_stdout(io.StringIO()):
            compare_models.main()
        constructed.assert_called_once_with(timeout=45.0)


class DeclaredInferenceConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.configurations = self.root / "inference_config"
        self.configurations.mkdir()
        self.log = self.root / "case.jsonl"
        self.log.write_text(json.dumps(EVENTS[0]) + "\n")

    def write_inference_configuration(self, name: str, text: str) -> None:
        (self.configurations / name).write_text(text, encoding="utf-8")

    def test_checked_in_qwen_inference_configuration_parses(self):
        configuration = inference_configurations.parse_inference_configuration_file(
            Path(__file__).resolve().parent.parent / "inference_config" / "qwen3-32b.conf"
        )
        self.assertEqual(configuration["source"], "inference_config/qwen3-32b.conf")
        self.assertEqual(
            configuration["fields"],
            {
                "model": "qwen3:32b",
                "thinking": "true",
                "num_ctx": "40960",
                "temperature": "0.6",
                "top_p": "0.95",
                "top_k": "20",
            },
        )
        self.assertEqual(inference_configurations.inference_configuration_num_ctx(configuration), 40960)
        self.assertIs(inference_configurations.inference_configuration_think(configuration), True)
        self.assertEqual(
            inference_configurations.inference_configuration_sampling(configuration),
            {"temperature": 0.6, "top_p": 0.95, "top_k": 20},
        )
        self.assertIn("# weight_precision=Developer Q8_0", configuration["text"])
        self.assertIn("# kv_cache=Q8_0", configuration["text"])
        self.assertIn("# repeat_penalty=1.0", configuration["text"])
        self.assertIn("# basis: Qwen's explicit thinking-mode recommendation", configuration["text"])

    def test_checked_in_inference_configurations_match_the_planned_setups(self):
        expected = {
            "qwen3-32b.conf": ("qwen3:32b", "40960", 40960, "true", True),
            "llama3.3-70b.conf": ("llama3.3:70b", "16384", 16384, None, None),
            "granite4.2-30b.conf": ("granite4.2:30b", "65536", 65536, "high", "high"),
            "deepseek-r1-32b.conf": ("deepseek-r1:32b", "65536", 65536, "true", True),
            "command-r.conf": ("command-r:latest", "131072", 131072, None, None),
            "gemma4-31b.conf": ("gemma4:31b", "32768", 32768, "true", True),
            "mistral-small3.2-24b.conf": (
                "mistral-small3.2:24b",
                "131072",
                131072,
                None,
                None,
            ),
            "mistral-nemo-12b.conf": ("mistral-nemo:12b", "131072", 131072, None, None),
            "foundation-sec-8b-instruct.conf": (
                "foundation-sec-8b-instruct",
                "131072",
                131072,
                None,
                None,
            ),
        }
        sampling = {
            "qwen3-32b.conf": ("0.6", 0.6, "0.95", 0.95, "20", 20),
            "llama3.3-70b.conf": ("0.2", 0.2, "0.90", 0.9, "40", 40),
            "granite4.2-30b.conf": ("1.0", 1.0, "0.95", 0.95, "40", 40),
            "deepseek-r1-32b.conf": ("0.6", 0.6, "0.95", 0.95, "40", 40),
            "command-r.conf": ("0.3", 0.3, "0.90", 0.9, "40", 40),
            "gemma4-31b.conf": ("1.0", 1.0, "0.95", 0.95, "64", 64),
            "mistral-small3.2-24b.conf": ("0.15", 0.15, "0.90", 0.9, "40", 40),
            "mistral-nemo-12b.conf": ("0.3", 0.3, "0.90", 0.9, "40", 40),
            "foundation-sec-8b-instruct.conf": ("0.2", 0.2, "0.90", 0.9, "40", 40),
        }
        root = Path(__file__).resolve().parent.parent / "inference_config"
        for name, (model, text, number, thinking, parsed) in expected.items():
            configuration = inference_configurations.parse_inference_configuration_file(root / name)
            temperature, temperature_value, top_p, top_p_value, top_k, top_k_value = sampling[name]
            fields = {
                "model": model,
                "num_ctx": text,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
            }
            if thinking is not None:
                fields["thinking"] = thinking
            self.assertEqual(configuration["fields"], fields)
            self.assertEqual(
                inference_configurations.inference_configuration_sampling(configuration),
                {
                    "temperature": temperature_value,
                    "top_p": top_p_value,
                    "top_k": top_k_value,
                },
            )
            self.assertEqual(inference_configurations.inference_configuration_num_ctx(configuration), number)
            self.assertEqual(inference_configurations.inference_configuration_think(configuration), parsed)
            self.assertIn("# weight_precision=", configuration["text"])
            self.assertIn("# repeat_penalty=", configuration["text"])
            self.assertIn("# basis:", configuration["text"])
            if thinking is None:
                self.assertIn("# thinking=", configuration["text"])
            if name == "granite4.2-30b.conf":
                self.assertIn(
                    "# thinking levels: false, low, medium, high",
                    configuration["text"],
                )
        self.assertIsNone(inference_configurations.quantization_note("Developer Q8_0", "Q8_0"))
        self.assertIsNone(inference_configurations.quantization_note("Developer QAT Q4_0", "Q4_0"))
        self.assertIsNone(
            inference_configurations.quantization_note("Existing developer Q8_0", "Q8_0")
        )
        self.assertIn("Q4_K_M", inference_configurations.quantization_note("Q4_K_M", "Q8_0"))

    def test_comments_and_blank_lines_are_ignored(self):
        configuration = inference_configurations.parse_inference_configuration_text(
            "# intended setup\n\nmodel=qwen3:32b\n", "memory"
        )
        self.assertEqual(configuration["fields"], {"model": "qwen3:32b"})

    def test_inference_configuration_field_name_must_match_exactly(self):
        rejected = (
            "Model=alpha\n",
            "model=alpha\nNum Ctx=40960\n",
            "model=alpha\nnum-ctx=40960\n",
        )
        for text in rejected:
            with self.assertRaises(ValueError) as error:
                inference_configurations.parse_inference_configuration_text(text, "memory")
            message = str(error.exception)
            self.assertIn("unknown field", message)
            self.assertIn("expected an exact key", message)

    def test_ambiguous_inference_configurations_stop_before_inference(self):
        self.write_inference_configuration("think.conf", "model=foundation-sec-alpha\ntemperature=0.6\n")
        self.write_inference_configuration("direct.conf", "model=foundation-sec-alpha:latest\ntemperature=0\n")
        api = client()
        with self.assertRaises(ValueError) as error:
            compare_models.run_comparison(
                api, ["foundation-sec-alpha"], [str(self.log)], self.root / "results", self.configurations
            )
        message = str(error.exception)
        self.assertIn("Multiple inference configurations", message)
        self.assertIn("think.conf", message)
        self.assertIn("direct.conf", message)
        self.assertFalse(api.chat.called)
        self.assertFalse((self.root / "results").exists())

    def test_explicit_inference_configurations_keep_same_model_runs_separate(self):
        self.write_inference_configuration("think.conf", "model=foundation-sec-alpha\ntemperature=0.6\n")
        self.write_inference_configuration("direct.conf", "model=foundation-sec-alpha\ntemperature=0\n")
        api = client()
        with contextlib.redirect_stdout(io.StringIO()):
            directory = compare_models.run_comparison(
                api,
                ["foundation-sec-alpha"],
                [str(self.log)],
                self.root / "results",
                self.configurations,
                [self.configurations / "think.conf", self.configurations / "direct.conf"],
            )
        rows = [json.loads(line) for line in (directory / "results.jsonl").read_text().splitlines()]
        self.assertEqual([row["model"] for row in rows], ["foundation-sec-alpha:latest", "foundation-sec-alpha:latest"])
        self.assertEqual(len({row["run_key"] for row in rows}), 2)
        self.assertNotEqual(rows[0]["declared_inference_configuration"], rows[1]["declared_inference_configuration"])
        html = (directory / "report.html").read_text()
        self.assertIn("temperature=0.6", html)
        self.assertIn("temperature=0", html)
        self.assertEqual(
            sorted(path.name for path in (directory / "declared-inference-configurations").iterdir()),
            ["direct.conf", "think.conf"],
        )

    def test_inference_configuration_num_ctx_is_sent_and_allocated(self):
        self.write_inference_configuration("alpha.conf", "model=foundation-sec-alpha\nnum_ctx=16384\n")
        api = client()
        with contextlib.redirect_stdout(io.StringIO()):
            directory = compare_models.run_comparison(
                api, ["foundation-sec-alpha"], [str(self.log)], self.root / "results", self.configurations
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

    def test_inference_configuration_sampling_is_sent(self):
        self.write_inference_configuration(
            "alpha.conf",
            "model=foundation-sec-alpha\n"
            "temperature=0.2\n"
            "top_p=0.90\n"
            "top_k=40\n",
        )
        api = client()
        with contextlib.redirect_stdout(io.StringIO()):
            compare_models.run_comparison(
                api, ["foundation-sec-alpha"], [str(self.log)], self.root / "results", self.configurations
            )
        options = api.chat.call_args.kwargs["options"]
        self.assertEqual(options["temperature"], 0.2)
        self.assertEqual(options["top_p"], 0.9)
        self.assertEqual(options["top_k"], 40)

    def test_inference_configuration_without_sampling_keeps_temperature_zero(self):
        self.write_inference_configuration("alpha.conf", "model=foundation-sec-alpha\n")
        api = client()
        with contextlib.redirect_stdout(io.StringIO()):
            compare_models.run_comparison(
                api, ["foundation-sec-alpha"], [str(self.log)], self.root / "results", self.configurations
            )
        options = api.chat.call_args.kwargs["options"]
        self.assertEqual(options["temperature"], harness.TEMPERATURE)
        self.assertNotIn("top_p", options)
        self.assertNotIn("top_k", options)

    def test_bad_sampling_stops_before_chat(self):
        self.write_inference_configuration("alpha.conf", "model=foundation-sec-alpha\ntop_p=1.5\n")
        api = client()
        with self.assertRaises(ValueError) as error:
            compare_models.run_comparison(
                api, ["foundation-sec-alpha"], [str(self.log)], self.root / "results", self.configurations
            )
        self.assertIn("top_p must be a decimal from 0 through 1", str(error.exception))
        self.assertFalse(api.chat.called)
        self.assertFalse((self.root / "results").exists())

    def test_inference_configuration_without_num_ctx_uses_fallback(self):
        self.write_inference_configuration("alpha.conf", "model=foundation-sec-alpha\n")
        api = client()
        with contextlib.redirect_stdout(io.StringIO()):
            directory = compare_models.run_comparison(
                api, ["foundation-sec-alpha"], [str(self.log)], self.root / "results", self.configurations
            )
        self.assertEqual(api.chat.call_args.kwargs["options"]["num_ctx"], harness.NUM_CTX)
        rows = [json.loads(line) for line in (directory / "results.jsonl").read_text().splitlines()]
        self.assertEqual(rows[0]["context"]["allocated"], harness.NUM_CTX)
        manifest = json.loads((directory / "manifest.json").read_text())
        self.assertEqual(manifest["models"][0]["num_ctx"], harness.NUM_CTX)

    def test_inference_configuration_thinking_is_sent_before_inference(self):
        self.write_inference_configuration("alpha.conf", "model=foundation-sec-alpha\nthinking=true\n")
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
                api, ["foundation-sec-alpha"], [str(self.log)], self.root / "results", self.configurations
            )
        self.assertTrue(seen["before"])
        manifest = json.loads((directory / "manifest.json").read_text())
        self.assertIs(manifest["models"][0]["think"], True)
        html = (directory / "report.html").read_text()
        self.assertIn("Thinking: enabled", html)

    def test_inference_configuration_thinking_level_is_sent_before_inference(self):
        self.write_inference_configuration("alpha.conf", "model=foundation-sec-alpha\nthinking=high\n")
        api = client()
        api.show.return_value.capabilities = ["completion", "thinking"]

        def generate(**kwargs):
            self.assertEqual(kwargs["think"], "high")
            return response()

        api.chat.side_effect = generate
        with contextlib.redirect_stdout(io.StringIO()):
            compare_models.run_comparison(
                api, ["foundation-sec-alpha"], [str(self.log)], self.root / "results", self.configurations
            )

    def test_inference_configuration_thinking_without_capability_stops_before_chat(self):
        self.write_inference_configuration("alpha.conf", "model=foundation-sec-alpha\nthinking=true\n")
        api = client()
        with self.assertRaises(ValueError) as error:
            compare_models.run_comparison(
                api, ["foundation-sec-alpha"], [str(self.log)], self.root / "results", self.configurations
            )
        self.assertIn("thinking capability", str(error.exception))
        self.assertFalse(api.chat.called)
        self.assertFalse((self.root / "results").exists())

    def test_bad_thinking_stops_before_chat(self):
        self.write_inference_configuration("alpha.conf", "model=foundation-sec-alpha\nthinking=sometimes\n")
        api = client()
        api.show.return_value.capabilities = ["completion", "thinking"]
        with self.assertRaises(ValueError) as error:
            compare_models.run_comparison(
                api, ["foundation-sec-alpha"], [str(self.log)], self.root / "results", self.configurations
            )
        self.assertIn("thinking must be", str(error.exception))
        self.assertFalse(api.chat.called)
        self.assertFalse((self.root / "results").exists())

    def test_bad_num_ctx_stops_before_chat(self):
        self.write_inference_configuration("alpha.conf", "model=foundation-sec-alpha\nnum_ctx=16384\u201324576\n")
        api = client()
        with self.assertRaises(ValueError) as error:
            compare_models.run_comparison(
                api, ["foundation-sec-alpha"], [str(self.log)], self.root / "results", self.configurations
            )
        self.assertIn("num_ctx must be a positive integer", str(error.exception))
        self.assertFalse(api.chat.called)
        self.assertFalse((self.root / "results").exists())

    def test_num_ctx_rejects_commas(self):
        configuration = inference_configurations.parse_inference_configuration_text(
            "model=foundation-sec-alpha\nnum_ctx=16,384\n", "memory"
        )
        with self.assertRaises(ValueError) as error:
            inference_configurations.inference_configuration_num_ctx(configuration)
        self.assertIn("no commas", str(error.exception))
        self.assertIn("16,384", str(error.exception))

    def test_inexact_inference_configuration_field_stops_before_chat(self):
        self.write_inference_configuration("alpha.conf", "model=foundation-sec-alpha\nnum-ctx=16384\n")
        api = client()
        with self.assertRaises(ValueError) as error:
            compare_models.run_comparison(
                api, ["foundation-sec-alpha"], [str(self.log)], self.root / "results", self.configurations
            )
        message = str(error.exception)
        self.assertIn("unknown field 'num-ctx'", message)
        self.assertIn("num_ctx", message)
        self.assertFalse(api.chat.called)
        self.assertFalse((self.root / "results").exists())

    def test_malformed_inference_configuration_stops_before_results_and_inference(self):
        self.write_inference_configuration("alpha.conf", "weight_quant=Q6_K\n")
        api = client()
        with self.assertRaises(ValueError) as error:
            compare_models.run_comparison(
                api, ["foundation-sec-alpha"], [str(self.log)], self.root / "results", self.configurations
            )
        self.assertIn("missing Model", str(error.exception))
        self.assertFalse(api.chat.called)
        self.assertFalse((self.root / "results").exists())

    def test_inference_configuration_is_saved_and_shown_before_inference(self):
        self.write_inference_configuration(
            "alpha.conf",
            "model=foundation-sec-alpha\n"
            "weight_quant=Q6_K\n"
            "# note=<script>alert(1)</script>\n"
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
            self.assertEqual(manifest["inference_configurations"][0]["fields"]["kv_cache"], "f16")
            self.assertIn("Q4_K_M", manifest["inference_configurations"][0]["quantization_note"])
            report = saved[0].with_name("report.html").read_text()
            self.assertIn("Declared inference configs", report)
            self.assertIn(
                "Uncommented inference config values drive the Ollama request",
                report,
            )
            self.assertIn("this block is the archived source file", report)
            self.assertIn("0 / 1 runs recorded", report)
            self.assertNotIn("<script>", report)
            self.assertIn("&lt;script&gt;", report)
            copied = saved[0].parent / "declared-inference-configurations" / "alpha.conf"
            self.assertIn("<script>", copied.read_text())
            seen["before"] = True
            return response()

        api.chat.side_effect = generate
        with contextlib.redirect_stdout(io.StringIO()) as out:
            directory = compare_models.run_comparison(
                api, ["foundation-sec-alpha"], [str(self.log)], self.root / "results", self.configurations
            )
        self.assertTrue(seen["before"])
        printed = out.getvalue()
        self.assertLess(printed.index("Declared inference configuration"), printed.index("Results:"))
        self.assertIn("installed quantization is Q4_K_M", printed)
        rows = [json.loads(line) for line in (directory / "results.jsonl").read_text().splitlines()]
        self.assertTrue(rows[0]["declared_inference_configuration"].endswith("alpha.conf"))
        self.assertEqual(api.chat.call_args.kwargs["options"]["temperature"], 0)
        self.assertEqual(api.chat.call_args.kwargs["options"]["seed"], harness.SEED)

    def test_missing_inference_configuration_is_recorded_as_absent(self):
        api = client()
        with contextlib.redirect_stdout(io.StringIO()) as out:
            directory = compare_models.run_comparison(
                api, ["foundation-sec-alpha"], [str(self.log)], self.root / "results", self.configurations
            )
        self.assertIn("No declared inference configuration for foundation-sec-alpha:latest", out.getvalue())
        html = (directory / "report.html").read_text()
        self.assertIn("No declared inference config", html)
        manifest = json.loads((directory / "manifest.json").read_text())
        self.assertEqual(manifest["inference_configurations"], [])
        rows = [json.loads(line) for line in (directory / "results.jsonl").read_text().splitlines()]
        self.assertIsNone(rows[0]["declared_inference_configuration"])
        self.assertFalse((directory / "declared-inference-configurations").exists())

    def test_single_hunt_records_inference_configuration_before_analysis(self):
        self.write_inference_configuration("alpha.conf", "model=foundation-sec-alpha\nkv_cache=f16\n")
        api = client()
        result = harness.run_hunt("foundation-sec-alpha", EVENTS, client=api)
        output = self.root / "single-results"
        seen = {}

        def fake_run(*args, **kwargs):
            saved = list(output.glob("*/declared-inference-configurations/alpha.conf"))
            self.assertEqual(len(saved), 1)
            self.assertIn("kv_cache=f16", saved[0].read_text())
            report = saved[0].parents[1] / "report.html"
            self.assertIn("0 / 1 runs recorded", report.read_text())
            seen["before"] = True
            return result

        with patch.object(sys, "argv", [
            "main.py", str(self.log), "--model", "foundation-sec-alpha",
            "--inference-config-dir", str(self.configurations),
            "--output-dir", str(output),
        ]), patch.object(single_hunt, "Client", return_value=api), \
                patch.object(single_hunt, "run_hunt", side_effect=fake_run), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            single_hunt.main()
        self.assertTrue(seen["before"])
        printed = out.getvalue()
        self.assertLess(printed.index("kv_cache="), printed.index("--- Analysis ---"))
        rows = [
            json.loads(line)
            for path in output.glob("*/results.jsonl")
            for line in path.read_text().splitlines()
        ]
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["declared_inference_configuration"].endswith("alpha.conf"))
        self.assertEqual(rows[0]["status"], "ok")

    def test_single_hunt_requires_a_choice_when_settings_differ(self):
        self.write_inference_configuration("think.conf", "model=foundation-sec-alpha\ntemperature=0.6\nkv_cache=f16\n")
        self.write_inference_configuration("direct.conf", "model=foundation-sec-alpha\ntemperature=0\nkv_cache=q8_0\n")
        api = client()
        output = self.root / "single-results"
        with patch.object(sys, "argv", [
            "main.py", str(self.log), "--model", "foundation-sec-alpha",
            "--inference-config-dir", str(self.configurations),
            "--output-dir", str(output),
        ]), patch.object(single_hunt, "Client", return_value=api), \
                patch.object(single_hunt, "run_hunt") as run, \
                contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()) as err, \
                self.assertRaises(SystemExit) as exit:
            single_hunt.main()
        self.assertEqual(exit.exception.code, 1)
        self.assertFalse(run.called)
        self.assertFalse(api.chat.called)
        self.assertIn("think.conf", err.getvalue())
        self.assertIn("direct.conf", err.getvalue())
        self.assertFalse(output.exists())

        result = harness.run_hunt("foundation-sec-alpha", EVENTS, client=api)

        def run_with(configuration_name):
            with patch.object(sys, "argv", [
                "main.py", str(self.log), "--model", "foundation-sec-alpha",
                "--inference-configuration", str(self.configurations / configuration_name),
                "--inference-config-dir", str(self.configurations),
                "--output-dir", str(output),
            ]), patch.object(single_hunt, "Client", return_value=api), \
                    patch.object(single_hunt, "run_hunt", return_value=result), \
                    contextlib.redirect_stdout(io.StringIO()):
                single_hunt.main()

        run_with("think.conf")
        run_with("direct.conf")
        recorded = []
        for path in sorted(output.glob("*/results.jsonl")):
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(len(rows), 1)
            recorded.append(rows[0]["declared_inference_configuration"])
            copied = list(path.parent.glob("declared-inference-configurations/*.conf"))
            self.assertEqual(len(copied), 1)
            self.assertTrue(rows[0]["declared_inference_configuration"].endswith(copied[0].name))
        self.assertEqual(len(recorded), 2)
        self.assertNotEqual(recorded[0], recorded[1])
        self.assertTrue(any(path.endswith("think.conf") for path in recorded))
        self.assertTrue(any(path.endswith("direct.conf") for path in recorded))

    def test_single_hunt_rejects_bad_inference_configuration_before_inference(self):
        self.write_inference_configuration("alpha.conf", "not an inference configuration\n")
        api = client()
        with patch.object(sys, "argv", [
            "main.py", str(self.log), "--model", "foundation-sec-alpha",
            "--inference-config-dir", str(self.configurations),
        ]), patch.object(single_hunt, "Client", return_value=api), \
                patch.object(single_hunt, "run_hunt") as run, \
                contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()) as err, \
                self.assertRaises(SystemExit) as exit:
            single_hunt.main()
        self.assertEqual(exit.exception.code, 1)
        self.assertFalse(run.called)
        self.assertFalse(api.chat.called)
        self.assertIn("expected 'key=value'", err.getvalue())


if __name__ == '__main__':
    unittest.main()
