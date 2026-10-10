import copy
import tempfile
import unittest
import urllib.error
from email.message import Message
from pathlib import Path
from unittest.mock import patch

from thermal_unsloth import (
    Controller,
    GpuSample,
    ProfileManager,
    StudioApi,
    ThermalPolicy,
    fallback_override,
    validate_config,
)

MODEL = "org/model-GGUF:Q4_K_M"
NORMAL: dict = {
    "gpu_ids": [1, 0], "gpu_index_kind": "physical", "gpu_memory_mode": "manual",
    "gpu_layers": 1024, "tensor_split": [3.0, 2.0], "tensor_parallel": False,
    "n_batch": 2048, "n_ubatch": 2048, "max_seq_length": 65536,
    "custom_context_length": 65536, "kv_cache_dtype": "f16", "disable_vision": False,
    "llama_extra_args": ["--split-mode", "layer", "--tensor-split=3,2", "--flash-attn", "on"],
}


class FakeStore:
    def __init__(self):
        self.rows = {MODEL: copy.deepcopy(NORMAL), "org/other-GGUF": {"max_seq_length": 8192}}

    def get(self):
        return copy.deepcopy(self.rows)

    def set(self, model, config):
        self.rows[model] = copy.deepcopy(config)

    def resolve(self, status):
        return MODEL, self.get()[MODEL]


class FakeApi:
    def __init__(self):
        self.busy = False
        self.training = False
        self.loads = []
        self.status: dict = {
            "active_model": "org/model-GGUF", "model_identifier": "org/model-GGUF",
            "is_gguf": True, "gguf_variant": "Q4_K_M", "context_length": 65536,
            "gpu_ids": [1, 0], "loading": [], "serving": ["org/model-GGUF"],
        }

    def request(self, route, payload=None):
        if route == "/api/inference/status":
            return copy.deepcopy(self.status)
        if route == "/api/liveness":
            return {"inference_active": self.busy}
        if route == "/api/train/status":
            return {"is_training_running": self.training}
        if route == "/api/inference/load":
            assert payload is not None
            self.loads.append(copy.deepcopy(payload))
            self.status["gpu_ids"] = payload["gpu_ids"]
            return {"context_length": payload["max_seq_length"]}
        raise AssertionError(route)


def sample(temp=79, active=False, counter=100):
    return GpuSample(1, "primary", "primary GPU", temp, active, counter, 16384)


class ThermalPolicyTests(unittest.TestCase):
    def test_configured_high_temperature_triggers_without_a_new_throttle(self):
        policy = ThermalPolicy(fallback_temperature=78)
        self.assertTrue(policy.update(sample(78, counter=500000), 0))
        self.assertTrue(policy.update(sample(68, counter=500000), 1))
        self.assertFalse(policy.update(sample(68, counter=500000), 121))

    def test_counter_reset_is_not_a_new_throttle(self):
        policy = ThermalPolicy()
        policy.update(sample(counter=100), 0)
        self.assertFalse(policy.update(sample(counter=1), 2))

    def test_high_temperature_and_historical_counter_do_not_trigger_switch(self):
        policy = ThermalPolicy()
        self.assertFalse(policy.update(sample(83, counter=500000), 0))

    def test_counter_increase_detects_a_throttle_between_polls(self):
        policy = ThermalPolicy()
        policy.update(sample(counter=100), 0)
        self.assertTrue(policy.update(sample(counter=101), 2))

    def test_detected_throttle_stays_pending_after_flag_clears(self):
        policy = ThermalPolicy()
        self.assertTrue(policy.update(sample(active=True), 0))
        self.assertTrue(policy.update(sample(active=False), 10))

    def test_recovery_requires_continuous_cooling_and_no_throttle(self):
        policy = ThermalPolicy()
        policy.update(sample(active=True), 0)
        policy.update(sample(68), 10)
        policy.update(sample(69), 100)
        policy.update(sample(68), 110)
        self.assertTrue(policy.update(sample(68), 229))
        self.assertFalse(policy.update(sample(68), 230))

    def test_missing_telemetry_breaks_the_cooling_window(self):
        policy = ThermalPolicy(hot=True)
        policy.update(sample(60), 0)
        policy.invalidate_cooling()
        self.assertTrue(policy.update(sample(60), 200))


class SwitchingTests(unittest.TestCase):
    def test_large_model_fallback_preserves_cpu_layers_and_strips_conflicting_flags(self):
        original = copy.deepcopy(NORMAL)
        original["llama_extra_args"] += ["--gpu-layers=1024", "--fit", "on"]
        result = fallback_override(original, 0, 512, gpu_layers=20)
        self.assertEqual(result["gpu_layers"], 20)
        self.assertEqual(result["gpu_ids"], [0])
        self.assertEqual(result["kv_cache_dtype"], "f16")
        self.assertNotIn("--gpu-layers=1024", result["llama_extra_args"])
        self.assertEqual(result["llama_extra_args"][-2:], ["--fit", "off"])

    def test_same_device_with_wrong_layer_count_is_reloaded_at_idle(self):
        self.manager.profiles[MODEL]["fallback_gpu_layers"] = 20
        self.api.status.update(gpu_ids=[0], gpu_layers=1024)
        self.controller.step(sample(active=True), 0, 0)
        self.assertEqual(len(self.api.loads), 1)
        self.assertEqual(self.store.rows[MODEL]["gpu_layers"], 20)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = FakeStore()
        self.api = FakeApi()
        self.manager = ProfileManager(self.store, Path(self.directory.name) / "state.json", {MODEL: {"context_limit": 65536, "batch_limit": 512}})
        self.controller = Controller(self.api, self.manager, lambda config: {
            "gpu_ids": config["gpu_ids"], "max_seq_length": config["max_seq_length"],
            "cache_type_kv": config["kv_cache_dtype"], "disable_vision": config["disable_vision"],
            "n_batch": config["n_batch"], "n_ubatch": config["n_ubatch"],
        })

    def test_busy_generation_is_not_cancelled_and_next_load_uses_fallback(self):
        self.api.busy = True
        self.controller.step(sample(active=True), 0, 0)
        self.assertEqual(self.api.loads, [])
        self.assertEqual(self.store.rows[MODEL]["gpu_ids"], [0])
        self.api.busy = False
        self.controller.step(sample(), 2, 0)
        self.assertEqual(self.api.loads[-1]["gpu_ids"], [0])
        self.assertIs(self.api.loads[-1]["force_cancel_active"], False)
        self.assertEqual(self.api.loads[-1]["max_seq_length"], 65536)
        self.assertEqual(self.api.loads[-1]["cache_type_kv"], "f16")

    def test_cooled_gpu_returns_only_at_an_idle_boundary(self):
        self.controller.step(sample(active=True), 0, 0)
        self.api.busy = True
        self.controller.step(sample(68), 10, 0)
        self.controller.step(sample(68), 130, 0)
        self.assertEqual(len(self.api.loads), 1)
        self.assertEqual(self.store.rows[MODEL], NORMAL)
        self.api.busy = False
        self.controller.step(sample(68), 132, 0)
        self.assertEqual(self.api.loads[-1]["gpu_ids"], [1, 0])

    def test_unloaded_model_is_not_started_but_future_autoload_uses_fallback(self):
        self.api.status["active_model"] = None
        self.controller.step(sample(active=True), 0, 0)
        self.assertEqual(self.api.loads, [])
        self.assertEqual(self.store.rows[MODEL]["gpu_ids"], [0])

    def test_training_prevents_runtime_reload(self):
        self.api.training = True
        self.controller.step(sample(active=True), 0, 0)
        self.assertEqual(self.api.loads, [])

    def test_model_loading_and_multiple_serving_models_prevent_reload(self):
        self.api.status["loading"] = ["another model"]
        self.controller.step(sample(active=True), 0, 0)
        self.assertEqual(self.api.loads, [])
        self.api.status["loading"] = []
        self.api.status["serving"] = ["a", "b"]
        self.controller.step(sample(), 2, 0)
        self.assertEqual(self.api.loads, [])

    def test_unverified_context_is_not_reduced_or_moved(self):
        self.api.status["context_length"] = 262144
        self.controller.step(sample(active=True), 0, 0)
        self.assertEqual(self.api.loads, [])

    def test_other_model_settings_and_vision_are_preserved(self):
        self.controller.step(sample(active=True), 0, 0)
        self.assertEqual(self.store.rows["org/other-GGUF"], {"max_seq_length": 8192})
        self.assertIs(self.api.loads[-1]["disable_vision"], False)

    def test_manual_edits_are_not_overwritten_when_restoring(self):
        self.manager.sync(True, 1, 0)
        self.store.rows[MODEL]["n_batch"] = 256
        self.manager.sync(False, 1, 0)
        self.assertEqual(self.store.rows[MODEL]["n_batch"], 256)

    def test_default_physical_index_and_normalized_profile_restore(self):
        del self.store.rows[MODEL]["gpu_index_kind"]
        original = copy.deepcopy(self.store.rows[MODEL])

        def normalize(config):
            result = copy.deepcopy(config)
            result.pop("gpu_index_kind", None)
            result.pop("tensor_split", None)
            return result

        setattr(self.store, "normalize", normalize)
        self.manager.sync(True, 1, 0)
        self.manager.sync(True, 1, 0)
        self.assertEqual(self.store.rows[MODEL]["gpu_ids"], [0])
        self.assertEqual(self.manager.state["conflicts"], [])
        self.manager.sync(False, 1, 0)
        self.assertEqual(self.store.rows[MODEL], original)

    def test_restart_recovers_original_profiles_instead_of_recording_fallback_as_normal(self):
        self.manager.sync(True, 1, 0)
        restarted = ProfileManager(self.store, self.manager.state_path, self.manager.profiles)
        self.assertTrue(restarted.hot)
        restarted.sync(False, 1, 0)
        self.assertEqual(self.store.rows[MODEL], NORMAL)

    def test_fallback_removes_stale_gpu_arguments_without_changing_quality(self):
        config = fallback_override(NORMAL, 0, 512)
        self.assertNotIn("--tensor-split=3,2", config["llama_extra_args"])
        self.assertEqual(config["kv_cache_dtype"], "f16")
        self.assertEqual(config["custom_context_length"], 65536)

    def test_load_race_is_deferred_without_cancelling(self):
        request = self.api.request

        def race(route, payload=None):
            if route.endswith("/load"):
                raise urllib.error.HTTPError("http://127.0.0.1", 409, "busy", Message(), None)
            return request(route, payload)

        with patch.object(self.api, "request", side_effect=race):
            self.assertEqual(self.controller.step(sample(active=True), 0, 0), "waiting-idle")
        self.assertEqual(self.api.loads, [])

    def test_desktop_idle_liveness_omits_inference_active(self):
        request = self.api.request

        def idle(route, payload=None):
            return {"status": "alive"} if route == "/api/liveness" else request(route, payload)

        with patch.object(self.api, "request", side_effect=idle):
            self.assertEqual(self.controller.step(sample(active=True), 0, 0), "switched-fallback")

    def test_model_changed_at_boundary_is_not_reloaded(self):
        request = self.api.request
        reads = 0

        def changed(route, payload=None):
            nonlocal reads
            result = request(route, payload)
            if route.endswith("/status") and "active_model" in result:
                reads += 1
                if reads == 2:
                    result["active_model"] = "another model"
            return result

        with patch.object(self.api, "request", side_effect=changed):
            self.assertEqual(self.controller.step(sample(active=True), 0, 0), "waiting-model")
        self.assertEqual(self.api.loads, [])


class ApiAndConfigTests(unittest.TestCase):
    def test_credentials_cannot_be_sent_to_remote_urls(self):
        for url in ("https://127.0.0.1:8888", "http://example.com", "http://127.0.0.1@example.com", "http://127.0.0.1:8888/remote", "http://localhost:8888"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                StudioApi(url)

    def test_authentication_retry_is_bounded(self):
        api = StudioApi("http://127.0.0.1:8888")
        attempts = []

        def denied(route, payload):
            attempts.append(route)
            raise urllib.error.HTTPError(api.base, 401, "denied", Message(), None)

        with patch.object(api, "authenticated_request", side_effect=denied), self.assertRaises(urllib.error.HTTPError):
            api.request("/api/inference/status")
        self.assertEqual(len(attempts), 2)

    def test_invalid_config_cannot_start_monitor(self):
        config = {"primary_gpu_uuid": "a", "fallback_gpu_uuid": "b", "profiles": {MODEL: {"context_limit": 65536, "batch_limit": 2048}}}
        validate_config(config)
        for changes in ({"poll_seconds": 0}, {"stable_seconds": 0}, {"return_temperature": 100}, {"fallback_gpu_uuid": "a"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_config({**config, **changes})


if __name__ == "__main__":
    unittest.main()
