import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import benchmark_ram_cache as benchmark


class BenchmarkTests(unittest.TestCase):
    def test_hot_or_throttled_initial_samples_prevent_generation(self):
        for temperature, active in [(78, False), (69, True)]:
            with self.subTest(temperature=temperature, active=active), patch.object(benchmark, "thermal_sample", return_value={"temperature": temperature, "thermal_active": active}), patch.object(benchmark, "generate") as generate:
                with self.assertRaisesRegex(RuntimeError, "before generation"):
                    benchmark.generate_monitored(1, None, [], "primary")
                generate.assert_not_called()

    def test_a_restarted_common_service_prevents_the_next_request(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            api = Mock()
            api.request.side_effect = [{"loading": False}, {"is_training_running": False}]
            backend = Mock()
            backend.bridge.side_effect = RuntimeError("Model not loaded")
            benchmark.ensure_apps_idle(root, api, backend)
            api.reset_mock()
            backend.reset_mock()
            (root / "service.json").write_text(json.dumps({"pid": 20}), encoding="utf-8")
            with patch.object(benchmark, "process_alive", return_value=True), self.assertRaisesRegex(RuntimeError, "Stop the common"):
                benchmark.ensure_apps_idle(root, api, backend)
            api.request.assert_not_called()
            backend.bridge.assert_not_called()

    def test_another_unsloth_generation_prevents_benchmark_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            api = Mock()
            api.request.side_effect = [{"active_model": "9b", "loading": False}, {"is_training_running": False}, {"inference_active": True}]
            backend = Mock()
            with self.assertRaisesRegex(RuntimeError, "current Unsloth generation"):
                benchmark.ensure_apps_idle(Path(directory), api, backend)
            backend.bridge.assert_not_called()

    def test_a_busy_native_slot_prevents_measurement(self):
        connection = Mock()
        connection.getresponse.return_value = SimpleNamespace(status=200, read=lambda: b'[{"is_processing":true}]')
        with patch.object(benchmark.http.client, "HTTPConnection", return_value=connection), self.assertRaisesRegex(RuntimeError, "one idle native slot"):
            benchmark.ensure_native_idle(1, None)
        connection.close.assert_called_once()

    def test_one_supported_thermal_counter_is_kept_when_the_other_is_unavailable(self):
        result = SimpleNamespace(stdout="68, 1328, 0, Not Active, Not Active, [N/A], 2300, 16000, 28.4\n")
        with patch.object(benchmark.subprocess, "run", return_value=result):
            sample = benchmark.thermal_sample("primary")
        self.assertEqual(sample["thermal_counter"], 2300)
        self.assertFalse(sample["thermal_active"])

    def test_a_live_common_service_prevents_benchmark_loads(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            service = root / ".llama.cpp/desktop/service.json"
            service.parent.mkdir(parents=True)
            service.write_text(json.dumps({"pid": 10}), encoding="utf-8")
            with patch.object(Path, "home", return_value=root), patch.object(benchmark, "process_alive", return_value=True), patch.object(benchmark, "StudioApi") as api, patch("sys.argv", ["benchmark", "--app", "unsloth", "--cache-mib", "8192", "--output", "result.json"]):
                with self.assertRaisesRegex(RuntimeError, "Stop the common"):
                    benchmark.main()
                api.assert_not_called()

    def test_a_transient_thermal_event_invalidates_the_generation(self):
        recorded = threading.Event()
        initial = {"temperature": 69, "sm_clock_mhz": 1328, "thermal_active": False, "thermal_counter": 10}
        samples = iter([initial, {**initial, "temperature": 80, "thermal_active": True, "thermal_counter": 11}, {**initial, "thermal_counter": 11}])

        def sample(_uuid):
            value = next(samples)
            if value["thermal_active"]:
                recorded.set()
            return value

        def generate(*_args):
            self.assertTrue(recorded.wait(5))
            return {"output": "42"}

        with patch.object(benchmark, "thermal_sample", side_effect=sample), patch.object(benchmark, "generate", side_effect=generate):
            result = benchmark.generate_monitored(1, None, [], "primary")
        self.assertFalse(result["thermal"]["unthrottled"])
        self.assertTrue(result["thermal"]["temperature_limit_reached"])
        self.assertEqual(len(result["thermal"]["samples"]), 3)

    def test_cooling_timeout_does_not_force_a_hot_gpu_to_run(self):
        sample = {"temperature": 77, "thermal_active": False}
        with patch.object(benchmark, "thermal_sample", return_value=sample), patch.object(benchmark.time, "monotonic", side_effect=[0, 11]), self.assertRaisesRegex(RuntimeError, "did not cool"):
            benchmark.wait_cool("primary", 69, 10)


if __name__ == "__main__":
    unittest.main()
