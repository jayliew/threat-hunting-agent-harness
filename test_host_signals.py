"""Tests for host placement, memory pressure, and swap warnings."""

from __future__ import annotations

from time import perf_counter
import unittest
from unittest.mock import Mock, patch

from ollama import ChatResponse, Message

import host_signals
import main as harness


MIB = 1024 * 1024
ANSWER = "Verdict: benign\nThreat type: none\nSummary: Normal activity.\nEvidence: e1"
EVENTS = [{
    "@timestamp": "2026-09-14T09:45:00.000Z",
    "event": {"id": "e1"},
    "message": "test",
}]


def sample(t, placement="100% GPU", pressure=1, swap=0):
    return {
        "t": t,
        "placement": placement,
        "pressure": pressure,
        "swap_used_bytes": swap,
    }


def interpret(samples, load=0.5, prompt=0.2):
    return host_signals.interpret_host_samples(
        samples, load_seconds=load, prompt_eval_seconds=prompt
    )


class ProcessorLabelTests(unittest.TestCase):
    def test_full_gpu_and_full_cpu(self):
        self.assertEqual(host_signals.processor_label(1000, 1000), "100% GPU")
        self.assertEqual(host_signals.processor_label(1000, 1001), "100% GPU")
        self.assertEqual(host_signals.processor_label(1000, 0), "100% CPU")

    def test_partial_uses_ceil_cpu_and_floor_gpu(self):
        self.assertEqual(host_signals.processor_label(1000, 170), "83%/17% CPU/GPU")

    def test_processor_for_matches_name_and_latest_alias(self):
        models = [{"name": "qwen3:32b", "model": "qwen3:32b", "size": 1000, "size_vram": 170}]
        self.assertEqual(host_signals.processor_for(models, "qwen3:32b"), "83%/17% CPU/GPU")
        latest = [{"name": "alpha:latest", "model": "alpha:latest", "size": 10, "size_vram": 10}]
        self.assertEqual(host_signals.processor_for(latest, "alpha"), "100% GPU")
        self.assertIsNone(host_signals.processor_for(models, "other"))

    def test_non_macos_readers_return_none(self):
        with patch.object(host_signals.sys, "platform", "linux"):
            self.assertIsNone(host_signals.read_pressure_level())
            self.assertIsNone(host_signals.read_swap_used_bytes())

    def test_read_placement_swallows_client_errors(self):
        with patch.object(host_signals, "Client", side_effect=OSError("down")):
            self.assertIsNone(host_signals.read_placement("qwen3:32b"))


class InterpretHostSamplesTests(unittest.TestCase):
    def test_full_gpu_and_quiet_host_have_no_warnings(self):
        result = interpret([sample(0.0), sample(2.0, swap=50 * MIB)])
        self.assertEqual(result["warnings"], [])
        self.assertEqual(result["host"]["signals"], [])
        self.assertEqual(result["host"]["placement"], "100% GPU")
        self.assertEqual(result["host"]["pressure_baseline"], "normal")
        self.assertEqual(result["host"]["pressure_peak"], "normal")

    def test_missing_model_does_not_report_placement(self):
        result = interpret([sample(0.0, placement=None), sample(2.0, placement=None)])
        self.assertEqual(result["warnings"], [])
        self.assertIsNone(result["host"]["placement"])

    def test_partial_and_cpu_only_placement(self):
        partial = interpret([sample(0.0, placement=None), sample(1.0, placement="83%/17% CPU/GPU")])
        self.assertEqual(
            partial["warnings"],
            ["Host: Ollama reports 83%/17% CPU/GPU rather than 100% GPU."],
        )
        self.assertEqual(partial["host"]["signals"], ["cpu_gpu_split"])
        self.assertEqual(partial["host"]["placement"], "83%/17% CPU/GPU")

        cpu = interpret([sample(0.0, placement="100% CPU")])
        self.assertIn("100% CPU", cpu["warnings"][0])
        self.assertEqual(cpu["host"]["signals"], ["cpu_only"])

    def test_pressure_enter_stay_and_rise(self):
        entered = interpret([sample(0.0, pressure=1), sample(1.0, pressure=2)])
        self.assertIn(
            "Host: memory pressure entered warning during inference.",
            entered["warnings"],
        )
        self.assertEqual(entered["host"]["pressure_baseline"], "normal")
        self.assertEqual(entered["host"]["pressure_peak"], "warning")

        stayed = interpret([sample(0.0, pressure=4), sample(1.0, pressure=4)])
        self.assertIn(
            "Host: memory pressure stayed at critical during inference.",
            stayed["warnings"],
        )

        rose = interpret([sample(0.0, pressure=2), sample(1.0, pressure=4)])
        self.assertIn(
            "Host: memory pressure rose from warning to critical during inference.",
            rose["warnings"],
        )
        self.assertIn("memory_pressure", rose["host"]["signals"])

        unknown = interpret([sample(0.0, pressure=None), sample(1.0, pressure=4)])
        self.assertIn(
            "Host: memory pressure reached critical during inference.",
            unknown["warnings"],
        )

    def test_swap_growth_uses_the_generation_window(self):
        grown = interpret([
            sample(0.0, swap=0),
            sample(0.6, swap=40 * MIB),
            sample(0.7, swap=40 * MIB),
            sample(2.0, swap=140 * MIB),
        ])
        self.assertIn("Host: swap grew 100 MiB during generation.", grown["warnings"])
        self.assertEqual(grown["host"]["swap_used_bytes_start"], 40 * MIB)
        self.assertEqual(grown["host"]["swap_used_bytes_end"], 140 * MIB)
        self.assertIn("swap", grown["host"]["signals"])

    def test_swap_growth_during_load_is_not_generation(self):
        loaded = interpret([
            sample(0.0, swap=0),
            sample(0.4, swap=500 * MIB),
            sample(2.0, swap=500 * MIB),
        ])
        self.assertEqual(loaded["warnings"], [])
        self.assertEqual(loaded["host"]["swap_used_bytes_start"], 500 * MIB)
        self.assertEqual(loaded["host"]["signals"], [])

    def test_swap_just_under_threshold_is_quiet(self):
        quiet = interpret([
            sample(0.0, swap=0),
            sample(2.0, swap=host_signals.SWAP_GROWTH_BYTES - 1),
        ])
        self.assertEqual(quiet["warnings"], [])
        self.assertNotIn("swap", quiet["host"]["signals"])

    def test_missing_durations_do_not_report_swap(self):
        samples = [sample(0.0, swap=0), sample(2.0, swap=500 * MIB)]
        missing_load = interpret(samples, load=None, prompt=0.2)
        missing_prompt = interpret(samples, load=0.5, prompt=None)
        self.assertEqual(missing_load["warnings"], [])
        self.assertEqual(missing_prompt["warnings"], [])
        self.assertIsNone(missing_load["host"]["swap_used_bytes_start"])

    def test_swap_without_before_or_after_sample_is_not_reported(self):
        no_before = interpret([
            sample(1.0, swap=0),
            sample(2.0, swap=500 * MIB),
        ], load=0.4, prompt=0.1)
        no_after = interpret([
            sample(0.0, swap=0),
            sample(0.4, swap=500 * MIB),
        ], load=0.5, prompt=0.3)
        self.assertEqual(no_before["warnings"], [])
        self.assertEqual(no_after["warnings"], [])
        self.assertIsNone(no_before["host"]["swap_used_bytes_end"])
        self.assertIsNone(no_after["host"]["swap_used_bytes_end"])


class HostMonitorTests(unittest.TestCase):
    def test_stop_does_not_wait_out_the_interval(self):
        monitor = host_signals.HostMonitor("alpha", reader=lambda model: {}, interval=30)
        monitor.start()
        started = perf_counter()
        samples = monitor.stop()
        self.assertLess(perf_counter() - started, 2)
        self.assertGreaterEqual(len(samples), 2)
        self.assertIsNone(samples[0]["placement"])

    def test_reader_errors_become_empty_samples(self):
        def boom(model):
            raise RuntimeError(model)

        monitor = host_signals.HostMonitor("alpha", reader=boom, interval=30)
        monitor.start()
        samples = monitor.stop()
        self.assertTrue(samples)
        self.assertTrue(all(sample["pressure"] is None for sample in samples))


class HuntHostSignalTests(unittest.TestCase):
    def test_detected_signals_warn_without_changing_status(self):
        api = Mock()
        api.chat.return_value = ChatResponse(
            message=Message(role="assistant", content=ANSWER),
            done_reason="stop",
            eval_count=40,
            prompt_eval_count=80,
            total_duration=2_000_000_000,
            load_duration=500_000_000,
            prompt_eval_duration=200_000_000,
            eval_duration=1_300_000_000,
        )

        class ScriptedMonitor:
            def start(self):
                return None

            def stop(self):
                return [
                    sample(0.0, placement="83%/17% CPU/GPU", pressure=1, swap=0),
                    sample(2.0, placement="83%/17% CPU/GPU", pressure=2, swap=200 * MIB),
                ]

        with patch.object(harness, "HostMonitor", return_value=ScriptedMonitor()):
            result = harness.run_hunt(
                "alpha:latest", EVENTS, client=api, capabilities=["completion"]
            )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(
            result["host"]["signals"],
            ["cpu_gpu_split", "memory_pressure", "swap"],
        )
        self.assertTrue(any("83%/17% CPU/GPU" in warning for warning in result["warnings"]))
        self.assertFalse(api.show.called)


if __name__ == "__main__":
    unittest.main()
