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

import compare
import main as harness


ANSWER = 'Verdict: benign\nThreat type: none\nSummary: Normal activity.\nEvidence: e1'
EVENTS = [{"id": "e1", "timestamp": 1}]


def response(content=ANSWER, reason="stop"):
    return ChatResponse(message=Message(role="assistant", content=content),
                        done_reason=reason, eval_count=40, total_duration=2_000_000_000,
                        load_duration=500_000_000, prompt_eval_duration=200_000_000,
                        eval_duration=1_300_000_000)


def client():
    result = Mock()
    result.list.return_value.models = [
        SimpleNamespace(model="alpha:latest", digest="digest-alpha"),
        SimpleNamespace(model="beta:1", digest="digest-beta"),
    ]
    result.show.return_value.capabilities = ["completion"]
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

    def test_single_hunt_cli_uses_shared_runner(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'case.jsonl'
            path.write_text(json.dumps(EVENTS[0])+'\n')
            result = harness.run_hunt('alpha', EVENTS, client=client())
            with patch.object(sys, 'argv', ['main.py', str(path), '--model', 'alpha']), patch.object(harness, 'run_hunt', return_value=result) as run, contextlib.redirect_stdout(io.StringIO()) as out:
                harness.main()
            run.assert_called_once_with('alpha', EVENTS)
            self.assertIn(ANSWER, out.getvalue())
            result['status'] = 'invalid'
            result['validation_errors'] = ['Invalid output']
            with patch.object(sys, 'argv', ['main.py', str(path)]), patch.object(harness, 'run_hunt', return_value=result), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exit:
                harness.main()
            self.assertEqual(exit.exception.code, 1)


class ResultsDirectoryNameTests(unittest.TestCase):
    def test_eastern_daylight_sunday_morning(self):
        when = datetime(2026, 9, 20, 13, 28, tzinfo=timezone.utc)
        self.assertEqual(
            compare.results_directory_name(when),
            "Sunday-09-20-26_09-28am-ET",
        )

    def test_eastern_standard_saturday_evening(self):
        when = datetime(2026, 1, 11, 2, 5, tzinfo=timezone.utc)
        self.assertEqual(
            compare.results_directory_name(when),
            "Saturday-01-10-26_09-05pm-ET",
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
            return compare.run_comparison(api, ['alpha', 'beta:1'], self.logs, self.root / 'results')

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
        self.assertTrue((directory / 'manifest.json').exists())

    def test_missing_models_reported_together_without_starting_run(self):
        api = client()
        with self.assertRaises(ValueError) as error:
            compare.run_comparison(api, ['missing-one', 'missing-two'], self.logs, self.root/'results')
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
            compare.prepare_comparison(api, ['alpha'], [str(empty), str(malformed)])
        self.assertIn('empty.jsonl', str(error.exception))
        self.assertIn('bad.jsonl', str(error.exception))
        self.assertFalse(api.chat.called)

    def test_duplicate_aliases_and_paths_run_once(self):
        models, logs = compare.prepare_comparison(client(), ['alpha', 'alpha:latest'], self.logs*2)
        self.assertEqual(len(models), 1)
        self.assertEqual(len(logs), 2)

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
        with patch.object(compare, "eastern_now", return_value=when):
            first = self.run_quietly(client())
            second = self.run_quietly(client())
        self.assertEqual(first.name, "Sunday-09-20-26_09-28am-ET")
        self.assertEqual(second.name, "Sunday-09-20-26_09-28am-ET-2")
        self.assertTrue((first / "results.jsonl").exists())

    def test_cli_exits_nonzero_after_recording_invalid_runs(self):
        api = client()
        api.chat.return_value = response('bad answer')
        with patch.object(compare, 'Client', return_value=api), patch.object(sys, 'argv', ['compare.py', '--models', 'alpha', '--logs', self.logs[0], '--output-dir', str(self.root/'results')]), contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as exit:
            compare.main()
        self.assertEqual(exit.exception.code, 1)
        self.assertEqual(len(list((self.root/'results').glob('*/report.html'))), 1)


if __name__ == '__main__':
    unittest.main()
