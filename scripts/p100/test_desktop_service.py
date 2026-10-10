import tempfile
import threading
import unittest
from pathlib import Path
from typing import cast
from unittest.mock import call, patch

from desktop_service import DesktopController, DesktopServer, Handler, LmBackend, ThreadingHTTPServer
from shared_memory import MemoryStore, prepare_context


class Backend:
    def __init__(self):
        self.calls = []
        self.engine = {"devices": "CUDA1,CUDA0", "modelKey": "9b", "identifier": "9b", "context": 2048}

    def bridge(self, action="inspect", hot=False):
        self.calls.append((action, hot))
        if action == "reload":
            self.engine["devices"] = "CUDA0" if hot else "CUDA1,CUDA0"
        return self.engine


class Api:
    active = None

    def __init__(self, _url):
        pass

    def request(self, route):
        return {"active_model": self.active} if route.endswith("status") and "inference" in route else {"is_training_running": False}


class DesktopTests(unittest.TestCase):
    def test_cancelled_background_summary_can_release_its_native_slot_before_chat(self):
        from unittest.mock import Mock

        self.backend.bridge = Mock(side_effect=[{"idle": False}, {"idle": True}])
        with patch("desktop_service.time.sleep"):
            self.controller.wait_summary_idle(threading.Event())
        self.assertEqual(self.backend.bridge.call_count, 2)
        self.assertTrue(all(call.args == () for call in self.backend.bridge.call_args_list))

    def test_summary_cleanup_wait_does_not_cancel_a_foreign_native_generation(self):
        from unittest.mock import Mock

        self.backend.bridge = Mock(return_value={"idle": False})
        with patch("desktop_service.time.monotonic", side_effect=[0, 0, 16]), patch("desktop_service.time.sleep"), self.assertRaisesRegex(RuntimeError, "busy"):
            self.controller.wait_summary_idle(threading.Event())
        self.assertEqual(self.backend.bridge.call_count, 1)

    def test_idle_unsloth_can_release_vram_for_an_explicit_lmstudio_request(self):
        from unittest.mock import Mock

        api = Mock()
        state = {"active_model": "org/27b", "model_identifier": "org/27b", "loading": []}
        api.request.side_effect = [state, {"status": "alive", "inference_active": False}, {"is_training_running": False}, state, {}]
        with patch("desktop_service.StudioApi", return_value=api):
            self.controller.release_unsloth()
        self.assertEqual(api.request.call_args.args, ("/api/inference/unload", {"model_path": "org/27b", "force_cancel_active": False}))

    def test_generating_unsloth_cannot_be_unloaded_for_another_app(self):
        from unittest.mock import Mock

        api = Mock()
        api.request.side_effect = [{"active_model": "org/27b", "loading": []}, {"inference_active": True}]
        with patch("desktop_service.StudioApi", return_value=api), self.assertRaisesRegex(RuntimeError, "generating"):
            self.controller.release_unsloth()
        self.assertFalse(any(call.args[0].endswith("unload") for call in api.request.call_args_list))

    def test_hot_unsloth_27b_load_keeps_its_variant_and_cpu_layer_allocation(self):
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock

        root = Path(self.temporary.name)
        (root / "state.json").write_text('{"managed": {}, "conflicts": []}', encoding="utf-8")
        thermal = root / "config.json"
        thermal.write_text(json.dumps({"fallback_gpu_uuid": "fallback", "profiles": {"27b": {"batch_limit": 512, "fallback_gpu_layers": 20}}}), encoding="utf-8")
        self.controller.config.update(unsloth_thermal_config=str(thermal), unsloth_loads={"27b": {"model_path": "org/27b", "model_file": "C:/models/27b.gguf", "gguf_variant": "UD-Q4_K_XL"}})
        self.controller.policy.hot = True
        profile = {"gpu_ids": [1, 0], "gpu_layers": 1024, "kv_cache_dtype": "f16", "max_seq_length": 65536}
        store = Mock()
        store.resolve.return_value = ("27b", profile)
        store.make_load_kwargs.side_effect = lambda config: config
        api = Mock()
        api.request.side_effect = [{"model_identifier": "org/9b"}, {}, {"model_identifier": "org/27b", "gguf_variant": "UD-Q4_K_XL", "gpu_ids": [0], "gpu_layers": 20}]
        with patch.object(self.controller, "sample"), patch("desktop_service.StudioStore", return_value=store), patch("desktop_service.gpu_samples", return_value={"fallback": SimpleNamespace(index=0)}):
            self.controller.prepare_unsloth(api, {"model": "27b"})
        payload = api.request.call_args_list[1].args[1]
        self.assertEqual(payload["model_path"], "org/27b")
        self.assertEqual(payload["gguf_variant"], "UD-Q4_K_XL")
        self.assertEqual(payload["gpu_layers"], 20)
        self.assertEqual(payload["gpu_ids"], [0])
        self.assertFalse(payload["force_cancel_active"])

    def test_native_file_mismatch_is_rejected_despite_the_sdk_identifier(self):
        import json
        from types import SimpleNamespace

        backend = LmBackend({"lmstudio": {"identifier": "27b", "models": {"27b": {"modelKey": "27b", "model_file": "C:/models/27b.gguf"}}},
            "fallback_device": "CUDA0", "primary_devices": "CUDA1,CUDA0"})
        backend.config["lmstudio"]["node"] = "node"
        result = {"identifier": "27b", "modelKey": "27b", "native_model": "C:/models/9b.gguf"}
        with patch("desktop_service.subprocess.run", return_value=SimpleNamespace(returncode=0, stdout=json.dumps(result))), self.assertRaisesRegex(RuntimeError, "identity"):
            backend.bridge()
        self.assertIsNone(backend.engine)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.store = MemoryStore(Path(self.temporary.name) / "memory.sqlite3")
        self.backend = Backend()
        self.controller = DesktopController({"lmstudio": {"identifier": "9b", "models": {"9b": {"modelKey": "9b"}}}, "primary_devices": "CUDA1,CUDA0", "fallback_device": "CUDA0"}, self.backend, self.store)
        Api.active = None
        self.api = patch("desktop_service.StudioApi", Api)
        self.api.start()
        self.addCleanup(self.api.stop)

    def test_active_chat_prevents_reload(self):
        import time
        self.controller.leases["one"] = {"heartbeat": time.monotonic()}
        self.controller.policy.hot = True
        self.assertEqual(self.controller.switch(), "waiting-chat")
        self.assertEqual(self.backend.calls, [])

    def test_other_app_model_prevents_vram_conflict(self):
        Api.active = "27b"
        self.controller.policy.hot = True
        self.assertEqual(self.controller.switch(), "waiting-unsloth")
        self.assertEqual(self.backend.calls, [])

    def test_unknown_model_does_not_use_9b_fallback(self):
        self.backend.engine["modelKey"] = "27b"
        self.controller.policy.hot = True
        self.assertEqual(self.controller.switch(), "unverified-model")
        self.assertNotIn(("reload", True), self.backend.calls)

    def test_hot_and_cooled_transitions_change_device_order(self):
        self.controller.policy.hot = True
        self.assertEqual(self.controller.switch(), "switched-fallback")
        self.controller.policy.hot = False
        self.assertEqual(self.controller.switch(), "switched-primary")
        self.assertEqual(self.backend.engine["devices"], "CUDA1,CUDA0")

    def test_automatic_gpu_selection_is_replaced_by_the_explicit_profile(self):
        del self.backend.engine["devices"]
        self.controller.policy.hot = True
        self.assertEqual(self.controller.switch(), "switched-fallback")
        self.assertEqual(self.backend.engine["devices"], "CUDA0")

    def test_busy_native_engine_prevents_reload(self):
        self.backend.engine["idle"] = False
        self.controller.policy.hot = True
        self.assertEqual(self.controller.switch(), "waiting-lmstudio")
        self.assertNotIn(("reload", True), self.backend.calls)

    def test_fallback_context_is_lower_than_normal_context(self):
        self.controller.config.update(context_limit=131072, fallback_context_limit=65536)
        self.backend.engine.update(context=131072, devices="CUDA0")
        self.controller.policy.hot = True
        self.assertEqual(self.controller.switch(), "switched-fallback")

    def test_second_model_uses_its_own_context_limits(self):
        self.controller.config.update(context_limit=131072, fallback_context_limit=2048)
        self.controller.config["lmstudio"]["models"]["27b"] = {"modelKey": "27b", "context_limit": 65536, "fallback_context_limit": 65536}
        self.backend.engine.update(identifier="27b", modelKey="27b", context=65536, devices="CUDA0")
        self.controller.policy.hot = True
        self.assertEqual(self.controller.switch(), "matched-fallback")
        self.assertNotIn(("reload", True), self.backend.calls)

    def test_requested_model_does_not_silently_use_a_different_resident_model(self):
        self.controller.config["lmstudio"]["models"]["27b"] = {"modelKey": "27b"}
        with patch.object(self.controller, "sample"), self.assertRaisesRegex(RuntimeError, "requested model"):
            self.controller.prepare({"model": "27b", "messages": [{"role": "user", "content": "new"}]}, threading.Event())
        self.assertNotIn(("load", False), self.backend.calls)

    def test_waiting_other_app_cannot_prepare_or_load_a_model(self):
        Api.active = "27b"
        with patch.object(self.controller, "sample"), self.assertRaisesRegex(RuntimeError, "Another Desktop"):
            self.controller.prepare({"messages": [{"role": "user", "content": "new"}]}, threading.Event())
        self.assertEqual(self.backend.calls, [])

    def test_unavailable_unsloth_does_not_unload_the_resident_native_model(self):
        from types import SimpleNamespace
        from unittest.mock import Mock

        handler = object.__new__(Handler)
        self.backend.engine["idle"] = True
        self.controller.config["unsloth_url"] = "http://127.0.0.1:8888"
        handler.server = cast(DesktopServer, SimpleNamespace(controller=self.controller))
        api = Mock()
        api.request.side_effect = RuntimeError("Unavailable")
        with patch("desktop_service.StudioApi", return_value=api), self.assertRaisesRegex(RuntimeError, "Unavailable"):
            handler.proxy_unsloth({"messages": [{"role": "user", "content": "new"}]}, threading.Event())
        self.assertEqual(self.backend.calls, [])

    def test_oversized_latest_input_is_explicitly_rejected(self):
        with self.assertRaisesRegex(ValueError, "Latest input"):
            prepare_context([{"role": "user", "content": "x " * 4000}], app="lmstudio", thread="one", context_length=2048,
                max_tokens=128, count_tokens=lambda m: sum(len(x["content"].split()) for x in m), summarize=lambda *_: "summary",
                config={"trigger_tokens": 1000, "recent_turns": 4, "summary_tokens": 128, "safety_tokens": 16},
                state_dir=Path(self.temporary.name), scope="one", store=self.store)


class HttpTests(unittest.TestCase):
    def setUp(self):
        from types import SimpleNamespace

        self.server = DesktopServer(("127.0.0.1", 0), Handler)
        self.server.token = "test-key"
        self.server.controller = cast(DesktopController, SimpleNamespace(config={"lmstudio": {"identifier": "9b", "models": {"9b": {"modelKey": "9b"}, "27b": {"modelKey": "27b"}}}, "unsloth_models": ["unsloth/27b"]}, status="idle", policy=SimpleNamespace(hot=False)))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

    def request(self, method, path, body=None, headers=None):
        import http.client
        import json

        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port)
        self.addCleanup(connection.close)
        connection.request(method, path, json.dumps(body) if body is not None else None, headers or {})
        response = connection.getresponse()
        return response.status, json.loads(response.read())

    def test_missing_key_cannot_access_native_backend(self):
        self.assertEqual(self.request("GET", "/v1/models")[0], 401)

    def test_browser_origin_is_rejected_even_with_key(self):
        self.assertEqual(self.request("GET", "/status", headers={"Authorization": "Bearer test-key", "Origin": "https://example.test"})[0], 401)

    def test_configured_model_is_listed(self):
        status, result = self.request("GET", "/v1/models", headers={"Authorization": "Bearer test-key"})
        self.assertEqual(status, 200)
        self.assertEqual(result["data"][0]["id"], "9b")
        self.assertEqual({row["id"] for row in result["data"]}, {"9b", "27b", "unsloth/27b"})

    def test_unknown_model_is_rejected_before_backend_access(self):
        status, result = self.request("POST", "/v1/chat/completions", {"model": "unknown", "messages": [{"role": "user", "content": "new"}]}, {"Authorization": "Bearer test-key"})
        self.assertEqual(status, 400)
        self.assertIn("registered", result["detail"])

    def test_invalid_message_array_is_rejected_before_loading(self):
        status, _result = self.request("POST", "/v1/chat/completions", {"messages": "invalid"}, {"Authorization": "Bearer test-key"})
        self.assertEqual(status, 400)

    def test_invalid_protected_indices_are_rejected_before_loading(self):
        status, _ = self.request("POST", "/lmstudio/prepare", {"messages": [{"role": "user", "content": "new"}], "protected_indices": ["0"]}, {"Authorization": "Bearer test-key"})
        self.assertEqual(status, 400)


class ProxyTests(unittest.TestCase):
    def test_unsloth_stream_keeps_its_own_shared_core_and_authentication(self):
        import http.client
        import json
        from http.server import BaseHTTPRequestHandler
        from types import SimpleNamespace

        wire = b'data: {"choices":[{"delta":{"content":"42"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
        received = []

        class Upstream(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass

            def do_POST(self):
                received.append((self.headers["Authorization"], json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(wire)))
                self.end_headers()
                self.wfile.write(wire)

        upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
        threading.Thread(target=upstream.serve_forever, daemon=True).start()
        self.addCleanup(upstream.server_close)
        self.addCleanup(upstream.shutdown)

        class LocalApi:
            def __init__(self, url):
                self.base, self.token = url, "upstream-test-key"

            def request(self, _route):
                return {"active_model": None}

        from unittest.mock import Mock
        backend = Mock()
        backend.bridge.side_effect = RuntimeError("Model not loaded")
        controller = SimpleNamespace(config={"lmstudio": {"identifier": "9b", "models": {}}, "unsloth_models": ["unsloth/27b"],
            "unsloth_url": "http://127.0.0.1:" + str(upstream.server_port)}, lock=threading.RLock(), leases={}, background_cancel=None,
            backend=backend, prepare=Mock(), prepare_unsloth=Mock(), store=Mock(), report=Mock())
        gateway = DesktopServer(("127.0.0.1", 0), Handler)
        gateway.controller, gateway.token = cast(DesktopController, controller), "gateway-test-key"
        threading.Thread(target=gateway.serve_forever, daemon=True).start()
        self.addCleanup(gateway.server_close)
        self.addCleanup(gateway.shutdown)
        connection = http.client.HTTPConnection("127.0.0.1", gateway.server_port)
        self.addCleanup(connection.close)
        body = {"model": "unsloth/27b", "messages": [{"role": "user", "content": "new"}], "stream": True, "thread_id": "one", "max_tokens": 16384}
        with patch("desktop_service.StudioApi", LocalApi):
            connection.request("POST", "/v1/chat/completions", json.dumps(body), {"Authorization": "Bearer gateway-test-key"})
            response = connection.getresponse()
            self.assertEqual(response.status, 200)
            self.assertEqual(response.read(), wire)
        self.assertEqual(received, [("Bearer upstream-test-key", body)])
        controller.prepare.assert_not_called()
        controller.store.remember.assert_not_called()
        self.assertNotIn(call("unload"), backend.bridge.call_args_list)

    def test_stream_and_tool_calls_are_passed_through_without_internal_fields(self):
        import http.client
        import json
        from http.server import BaseHTTPRequestHandler
        from types import SimpleNamespace

        wire = b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call-one","function":{"name":"check","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}\n\ndata: [DONE]\n\n'
        received = []

        class Native(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass

            def do_POST(self):
                received.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(wire)))
                self.end_headers()
                self.wfile.write(wire)

        native = ThreadingHTTPServer(("127.0.0.1", 0), Native)
        threading.Thread(target=native.serve_forever, daemon=True).start()
        self.addCleanup(native.server_close)
        self.addCleanup(native.shutdown)

        class ProxyBackend:
            def __init__(self):
                self.engine = {"context": 2048}

            def open(self, path, body):
                connection = http.client.HTTPConnection("127.0.0.1", native.server_port)
                connection.request("POST", path, json.dumps(body))
                return connection, connection.getresponse()

        controller = SimpleNamespace(config={"lmstudio": {"identifier": "9b", "models": {"9b": {"modelKey": "9b"}}}}, lock=threading.RLock(),
            leases={}, background_cancel=None, backend=ProxyBackend(), report=lambda: None,
            prepare=lambda body, _cancel: {"messages": body["messages"]}, store=SimpleNamespace(remember=lambda *_: None))
        gateway = DesktopServer(("127.0.0.1", 0), Handler)
        gateway.controller, gateway.token = cast(DesktopController, controller), "test-key"
        threading.Thread(target=gateway.serve_forever, daemon=True).start()
        self.addCleanup(gateway.server_close)
        self.addCleanup(gateway.shutdown)
        connection = http.client.HTTPConnection("127.0.0.1", gateway.server_port)
        self.addCleanup(connection.close)
        tools = [{"type": "function", "function": {"name": "check", "parameters": {"type": "object"}}}]
        connection.request("POST", "/v1/chat/completions", json.dumps({"model": "9b", "messages": [{"role": "user", "content": "check"}],
            "tools": tools, "stream": True, "thread_id": "one", "reserve_tokens": 32}), {"Authorization": "Bearer test-key"})
        response = connection.getresponse()
        self.assertEqual(response.status, 200)
        self.assertEqual(response.read(), wire)
        self.assertEqual(received[0]["tools"], tools)
        self.assertNotIn("thread_id", received[0])
        self.assertNotIn("reserve_tokens", received[0])
        self.assertEqual(received[0]["max_tokens"], 512)


if __name__ == "__main__":
    unittest.main()
