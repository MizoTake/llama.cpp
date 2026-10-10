import argparse
import hashlib
import http.client
import json
import logging
import ntpath
import os
import secrets
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from context_unsloth import DEFAULTS, MARKER, _summary_prompt, _transcript
from import_desktop_history import import_lmstudio, import_unsloth
from shared_memory import (
    ROOT,
    CompletionCapture,
    MemoryStore,
    conversation_key,
    prepare_context,
)
from thermal_unsloth import (
    StudioApi,
    StudioStore,
    ThermalPolicy,
    fallback_override,
    gpu_samples,
    save_json,
)

logger = logging.getLogger(__name__)


class LmBackend:
    def __init__(self, config):
        self.config: dict = config
        self.engine: dict | None = None
        self.selected = config["lmstudio"]["identifier"]

    def bridge(self, action="inspect", hot=False):
        identifier = self.selected
        if action == "reload":
            if self.engine is None:
                raise RuntimeError("Model not loaded")
            identifier = self.engine["identifier"]
        profile = self.config["lmstudio"]["models"][identifier]
        payload: dict = {**self.config["lmstudio"], "identifier": identifier, "action": action, "hot": hot,
                   "contextLimit": profile.get("fallback_context_limit" if hot else "context_limit", self.config.get("fallback_context_limit" if hot else "context_limit", 65536)),
                   "devices": self.config["fallback_device"] if hot else self.config["primary_devices"],
                   "profiles": {key: str(ROOT / ("lmstudio-" + hashlib.sha256(key.encode()).hexdigest()[:16] + ".json")) for key in self.config["lmstudio"]["models"]}}
        result = subprocess.run([payload["node"], str(Path(__file__).with_name("lmstudio_bridge.mjs"))],
                                input=json.dumps(payload), capture_output=True, text=True, encoding="utf-8", timeout=180, check=False,
                                creationflags=subprocess.CREATE_NO_WINDOW)
        data = json.loads(result.stdout)
        if result.returncode or data.get("error"):
            raise RuntimeError(data.get("error", "LM Studio bridge failed"))
        if not data.get("unloaded"):
            actual = self.config["lmstudio"]["models"].get(data.get("identifier"), {})
            expected = actual.get("model_file")
            if expected is None or data.get("modelKey") != actual.get("modelKey") or ntpath.normcase(ntpath.normpath(data.get("native_model", ""))) != ntpath.normcase(ntpath.normpath(expected)):
                raise RuntimeError("Native model identity does not match the registered GGUF file")
        self.engine = data
        return data

    def open(self, path, body=None):
        if self.engine is None:
            self.bridge()
        if self.engine is None:
            raise RuntimeError("Model not loaded")
        connection = http.client.HTTPConnection("127.0.0.1", self.engine["port"], timeout=180)
        headers = {"Authorization": "Bearer " + self.engine["key"], "Content-Type": "application/json"}
        connection.request("GET" if body is None else "POST", path, None if body is None else json.dumps(body), headers)
        setattr(connection, "desktop_socket", connection.sock)
        response = connection.getresponse()
        if response.status >= 400:
            response.read(4096)
            connection.close()
            raise RuntimeError("Native llama.cpp returned HTTP " + str(response.status))
        return connection, response

    def request(self, path, body=None):
        connection, response = self.open(path, body)
        try:
            return json.loads(response.read())
        finally:
            connection.close()

    def count(self, messages, body=None):
        body = body or {}
        template = self.request("/apply-template", {**body, "messages": messages, "stream": False})["prompt"]
        return len(self.request("/tokenize", {"content": template, "add_special": False, "parse_special": True})["tokens"])

    def summarize(self, messages, limit, cancel):
        connection, response = self.open("/v1/chat/completions", {"messages": messages, "stream": True, "max_tokens": limit,
            "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}})
        cancel.set_connection(connection)
        text = []
        finished = False
        try:
            while not cancel.is_set():
                line = response.readline()
                if not line:
                    break
                if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                    continue
                for choice in json.loads(line[6:]).get("choices", []):
                    text.append(choice.get("delta", {}).get("content") or "")
                    finished = finished or choice.get("finish_reason") == "stop"
        finally:
            cancel.set_connection(None)
            connection.close()
        if cancel.is_set() or not finished:
            raise RuntimeError("Summary cancelled or incomplete")
        return "".join(text)


class Cancellation(threading.Event):
    def __init__(self):
        super().__init__()
        self.connection = None

    def set_connection(self, connection):
        self.connection = connection
        if self.is_set():
            self.set()

    def set(self):
        super().set()
        if self.connection is not None:
            sock = self.connection.sock or getattr(self.connection, "desktop_socket", None)
            if sock is not None:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
            self.connection.close()


class DesktopController:
    def __init__(self, config, backend=None, store=None):
        self.config = config
        self.backend = backend or LmBackend(config)
        self.store = store or MemoryStore()
        self.policy = ThermalPolicy(config.get("return_temperature", 68), config.get("stable_seconds", 120), fallback_temperature=config.get("fallback_temperature"))
        self.lock = threading.RLock()
        self.leases = {}
        self.status = "starting"
        self.applied_hot = None
        self.background_cancel = None

    def sample(self):
        try:
            sample = gpu_samples()[self.config["primary_uuid"]]
            self.policy.update(sample, time.monotonic())
        except Exception:
            self.policy.invalidate_cooling()
            raise

    def switch(self, requested=None):
        self.leases = {key: value for key, value in self.leases.items() if time.monotonic() - value["heartbeat"] < 60}
        if self.leases:
            return "waiting-chat"
        api = StudioApi(self.config.get("unsloth_url", "http://127.0.0.1:8888"))
        try:
            state = api.request("/api/inference/status")
            if state.get("active_model") or state.get("loading") or api.request("/api/train/status").get("is_training_running", True):
                return "waiting-unsloth"
        except (OSError, RuntimeError):
            # A closed app has no running GPU request; an unresponsive live app must defer.
            processes = subprocess.run(["powershell.exe", "-NoProfile", "-Command", "@(Get-Process -Name unsloth-studio -ErrorAction SilentlyContinue).Count"], capture_output=True, text=True, check=True, creationflags=subprocess.CREATE_NO_WINDOW)
            if processes.stdout.strip() != "0":
                return "waiting-unsloth-status"
        try:
            engine = self.backend.bridge()
        except RuntimeError as error:
            if str(error) == "Model not loaded":
                return "model-unloaded"
            raise
        desired = self.config["fallback_device"] if self.policy.hot else self.config["primary_devices"]
        profile = self.config["lmstudio"]["models"].get(engine["identifier"])
        if profile is None or engine["modelKey"] != profile["modelKey"]:
            return "unverified-model"
        if not engine.get("idle", True):
            return "waiting-lmstudio"
        if requested and requested != engine["identifier"]:
            return "requested-model"
        limit_name = "fallback_context_limit" if self.policy.hot else "context_limit"
        limit = profile.get(limit_name, self.config.get(limit_name, 65536))
        if engine.get("devices") == desired and engine["context"] <= limit:
            self.applied_hot = self.policy.hot
            return "matched-fallback" if self.policy.hot else "matched-primary"
        self.backend.bridge("reload", self.policy.hot)
        self.applied_hot = self.policy.hot
        return "switched-fallback" if self.policy.hot else "switched-primary"

    def prepare(self, body, cancel):
        self.sample()
        self.release_unsloth()
        identifier = body.get("model", self.config["lmstudio"]["identifier"])
        self.backend.selected = identifier
        self.status = self.switch(identifier)
        if self.status.startswith("waiting") or self.status == "unverified-model":
            raise RuntimeError("Another Desktop model is loaded or busy")
        if self.status in ("model-unloaded", "requested-model"):
            self.backend.bridge("load" if self.status == "model-unloaded" else "select", self.policy.hot)
        else:
            self.backend.bridge()
        engine = self.backend.engine
        if engine is None or engine["identifier"] != identifier:
            raise RuntimeError("Load the requested model in LM Studio after the current generation finishes")
        self.status = "prepared-fallback" if self.policy.hot else "prepared-primary"
        messages = body["messages"]
        protected = {id(messages[index]) for index in body.get("protected_indices", []) if 0 <= index < len(messages)}
        result = prepare_context(
            messages, app="lmstudio", thread=body.get("thread_id"), context_length=engine["context"],
            max_tokens=body.get("max_tokens"), count_tokens=lambda items: self.backend.count(items, body),
            summarize=lambda items, limit: self.backend.summarize(items, limit, cancel),
            config={**DEFAULTS, **self.config.get("context", {})}, state_dir=ROOT / "lmstudio-context",
            scope=[engine["modelKey"], conversation_key(messages, body.get("thread_id"))],
            protected_message_ids=protected, reserve_tokens=body.get("reserve_tokens", 0), store=self.store,
            share_memory=self.config.get("share_memory", True),
        )
        if cancel.is_set():
            raise RuntimeError("Request cancelled")
        fitted, event = result
        shared = [message for message in fitted if isinstance(message.get("content"), str) and message["content"].startswith("[Shared conversation memory")]
        summary = next((message["content"] for message in fitted if isinstance(message.get("content"), str) and message["content"].startswith(MARKER)), None)
        return {"messages": fitted, "event": event, "summary": summary, "shared": shared}

    def release_unsloth(self):
        api = StudioApi(self.config.get("unsloth_url", "http://127.0.0.1:8888"))
        try:
            state = api.request("/api/inference/status")
        except OSError:
            return
        if not state.get("active_model"):
            return
        live = api.request("/api/liveness")
        if state.get("loading") or len(state.get("serving", [])) > 1 or live.get("inference_active", live.get("status") != "alive") or api.request("/api/train/status").get("is_training_running", True):
            raise RuntimeError("Another Desktop is generating; wait before switching models")
        fresh = api.request("/api/inference/status")
        if any(fresh.get(field) != state.get(field) for field in ("active_model", "model_identifier", "loading", "gpu_ids")):
            raise RuntimeError("Another Desktop model state changed; wait before switching")
        api.request("/api/inference/unload", {"model_path": state.get("model_identifier") or state["active_model"], "force_cancel_active": False})

    def wait_summary_idle(self, cancel):
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and not cancel.is_set():
            try:
                engine = self.backend.bridge()
            except RuntimeError as error:
                if str(error) == "Model not loaded":
                    return
                raise
            if engine.get("idle"):
                return
            # Closing the owned summary socket does not release a native slot at once.
            time.sleep(0.25)
        raise RuntimeError("Native backend remains busy after summary cancellation")

    def report(self):
        engine = self.backend.engine or {}
        if engine.get("unloaded"):
            engine = {}
        save_json(ROOT / "status.json", {"status": self.status, "hot": self.policy.hot, "active_chats": len(self.leases), "api_port": self.config["port"],
            "lmstudio_model": engine.get("identifier"), "native_model": engine.get("native_model"), "devices": engine.get("devices")})

    def prepare_unsloth(self, api, body):
        spec = self.config.get("unsloth_loads", {}).get(body["model"])
        if spec is None:
            return
        self.sample()
        thermal_path = Path(self.config["unsloth_thermal_config"])
        thermal = json.loads(thermal_path.read_text(encoding="utf-8"))
        store = StudioStore()
        key, profile = store.resolve({"model_identifier": spec["model_path"], "active_model": spec["model_file"], "gguf_variant": spec["gguf_variant"]})
        state = json.loads((thermal_path.parent / "state.json").read_text(encoding="utf-8"))
        if key in state.get("conflicts", []):
            raise RuntimeError("Unsloth thermal profile has a conflicting manual edit")
        profile = state.get("managed", {}).get(key, {}).get("original", profile)
        limits = thermal["profiles"][key]
        if self.policy.hot:
            fallback = gpu_samples()[thermal["fallback_gpu_uuid"]]
            profile = fallback_override(profile, fallback.index, limits["batch_limit"], limits["fallback_gpu_layers"])
        status = api.request("/api/inference/status")
        matches = status.get("model_identifier") in (spec["model_path"], spec["model_file"])
        matches = matches and (status.get("gguf_variant") == spec["gguf_variant"] or status.get("model_identifier") == spec["model_file"])
        matches = matches and all(status.get(field) == profile.get(field) for field in ("gpu_ids", "gpu_layers"))
        if not matches:
            payload = store.make_load_kwargs(profile)
            payload.update(model_path=spec["model_path"], gguf_variant=spec["gguf_variant"], force_reload=True, force_cancel_active=False)
            api.request("/api/inference/load", payload)
            status = api.request("/api/inference/status")
        if status.get("model_identifier") not in (spec["model_path"], spec["model_file"]) or status.get("gpu_ids") != profile["gpu_ids"] or status.get("gpu_layers") != profile["gpu_layers"]:
            raise RuntimeError("Unsloth model identity or GPU placement did not match the requested profile")
        self.status = "unsloth-fallback" if self.policy.hot else "unsloth-primary"


class DesktopServer(ThreadingHTTPServer):
    controller: DesktopController
    token: str


class Handler(BaseHTTPRequestHandler):
    server: DesktopServer
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):
        pass

    def reply(self, status, value):
        data = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def authorized(self):
        token = self.headers.get("Authorization", "")
        return not self.headers.get("Origin") and secrets.compare_digest(token, "Bearer " + self.server.token)

    def do_GET(self):
        if not self.authorized():
            self.reply(401, {"error": "Local desktop API key required"})
        elif self.path == "/v1/models":
            config = self.server.controller.config
            models = list(config["lmstudio"]["models"]) + config.get("unsloth_models", [])
            self.reply(200, {"object": "list", "data": [{"id": model, "object": "model", "owned_by": "local"} for model in models]})
        elif self.path == "/status":
            self.reply(200, {"status": self.server.controller.status, "hot": self.server.controller.policy.hot})
        else:
            self.reply(404, {"error": "Unknown endpoint"})

    def do_POST(self):
        if not self.authorized():
            self.reply(401, {"error": "Local desktop API key required"})
            return
        cancel = Cancellation()
        stop_watch = threading.Event()
        self.response_started = False

        def watch():
            import select
            while not stop_watch.wait(0.2):
                try:
                    ready, _, _ = select.select([self.connection], [], [], 0)
                    if ready and not self.connection.recv(1, socket.MSG_PEEK):
                        cancel.set()
                        return
                except OSError:
                    cancel.set()
                    return

        try:
            if self.headers.get("Transfer-Encoding"):
                raise ValueError("Use a Content-Length request")
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 32 * 1024 * 1024:
                raise ValueError("Invalid request size")
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                raise TypeError("Request body must be an object")
            controller = self.server.controller
            if self.path in ("/lmstudio/complete", "/lmstudio/heartbeat"):
                with controller.lock:
                    lease = controller.leases.get(body.get("lease"))
                    if lease and self.path.endswith("heartbeat"):
                        lease["heartbeat"] = time.monotonic()
                    elif lease:
                        if body.get("messages"):
                            controller.store.remember("lmstudio", lease["conversation"], body["messages"])
                        del controller.leases[body["lease"]]
                self.reply(200, {"ok": True})
                return
            if self.path not in ("/lmstudio/prepare", "/v1/chat/completions"):
                self.reply(404, {"error": "Unknown endpoint"})
                return
            if not isinstance(body.get("messages"), list) or not body["messages"] or any(not isinstance(item, dict) for item in body["messages"]):
                raise ValueError("messages must be a nonempty message array")
            identifier = body.get("model", controller.config["lmstudio"]["identifier"])
            unsloth = identifier in controller.config.get("unsloth_models", [])
            if identifier not in controller.config["lmstudio"]["models"] and not unsloth:
                raise ValueError("Select a registered Desktop model")
            if unsloth and self.path != "/v1/chat/completions":
                raise ValueError("Use chat/completions for Unsloth models")
            for key, low, high in (("max_tokens", 1, 65536), ("reserve_tokens", 0, 65536)):
                if key in body and (type(body[key]) is not int or not low <= body[key] <= high):
                    raise ValueError("Invalid " + key)
            indices = body.get("protected_indices", [])
            if not isinstance(indices, list) or any(type(index) is not int or not 0 <= index < len(body["messages"]) for index in indices):
                raise ValueError("Invalid protected_indices")
            threading.Thread(target=watch, daemon=True).start()
            background = controller.background_cancel
            if background is not None:
                background.set()
            with controller.lock:
                if controller.leases:
                    self.reply(409, {"error": "Wait for the current LM Studio chat"})
                    return
                if background is not None:
                    controller.wait_summary_idle(cancel)
                if unsloth:
                    self.proxy_unsloth(body, cancel)
                    controller.report()
                    return
                prepared = controller.prepare(body, cancel)
                if self.path == "/lmstudio/prepare":
                    lease = secrets.token_hex(16)
                    controller.leases[lease] = {"conversation": conversation_key(body["messages"], body.get("thread_id")), "heartbeat": time.monotonic()}
                    prepared["lease"] = lease
                    self.reply(200, prepared)
                else:
                    self.proxy({**body, "messages": prepared["messages"]}, body["messages"], cancel)
                controller.report()
        except (BrokenPipeError, ConnectionResetError):
            cancel.set()
        except Exception as error:  # noqa: BLE001 - Keep a malformed request from stopping the local service.
            try:
                if self.response_started:
                    self.close_connection = True
                else:
                    self.reply(503 if isinstance(error, RuntimeError) else 400, {"error": type(error).__name__, "detail": str(error) if isinstance(error, ValueError) else "Native backend unavailable; check Desktop status and loaded model"})
            except OSError:
                pass
        finally:
            stop_watch.set()
            cancel.set()

    def proxy(self, body, original, cancel):
        native_body = {key: value for key, value in body.items() if key not in ("thread_id", "protected_indices", "reserve_tokens")}
        engine = self.server.controller.backend.engine
        if engine is None:
            raise RuntimeError("Model not loaded")
        native_body.setdefault("max_tokens", engine["context"] // 4)
        connection, response = self.server.controller.backend.open("/v1/chat/completions", native_body)
        cancel.set_connection(connection)
        self.relay(connection, response, body, original, cancel, "lmstudio")

    def proxy_unsloth(self, body, cancel):
        controller = self.server.controller
        api = StudioApi(controller.config["unsloth_url"])
        api.request("/api/inference/status")
        try:
            engine = controller.backend.bridge()
        except RuntimeError as error:
            if str(error) != "Model not loaded":
                raise
        else:
            if not engine.get("idle"):
                raise RuntimeError("LM Studio is generating; wait before using Unsloth")
            controller.backend.bridge("unload")
        controller.prepare_unsloth(api, body)
        from urllib.parse import urlsplit

        target = urlsplit(api.base)
        connection = http.client.HTTPConnection(target.hostname, target.port or 80, timeout=180)
        upstream = {key: value for key, value in body.items() if key not in ("protected_indices", "reserve_tokens")}
        if api.token is None:
            raise RuntimeError("Unsloth authentication unavailable")
        connection.request("POST", "/v1/chat/completions", json.dumps(upstream), {"Content-Type": "application/json", "Authorization": "Bearer " + api.token})
        setattr(connection, "desktop_socket", connection.sock)
        response = connection.getresponse()
        cancel.set_connection(connection)
        if response.status >= 400:
            data = response.read(4096)
            connection.close()
            self.reply(response.status, {"error": "Unsloth backend rejected the request", "detail": data.decode("utf-8", errors="replace")})
            return
        # Unsloth already fits and archives through the shared core.
        self.relay(connection, response, body, body["messages"], cancel, None)

    def relay(self, connection, response, body, original, cancel, app):
        capture = CompletionCapture()
        try:
            self.send_response(200)
            self.send_header("Content-Type", response.getheader("Content-Type", "application/json"))
            self.send_header("Connection", "close")
            self.end_headers()
            self.response_started = True
            self.close_connection = True
            if body.get("stream"):
                while not cancel.is_set():
                    line = response.readline()
                    if not line:
                        break
                    self.wfile.write(line)
                    self.wfile.flush()
                    capture.feed(line)
            else:
                data = response.read()
                self.wfile.write(data)
                capture.add(json.loads(data))
            message = capture.result()
            if app and not cancel.is_set() and message is not None:
                self.server.controller.store.remember(app, conversation_key(original, body.get("thread_id")), original + [message])
        finally:
            cancel.set_connection(None)
            connection.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    token = (ROOT / "api-key.txt").read_text(encoding="utf-8").strip()
    controller = DesktopController(config)
    try:
        previous = json.loads((ROOT / "status.json").read_text(encoding="utf-8"))
        controller.policy.hot = previous.get("hot") is True
    except (OSError, ValueError):
        pass
    server = DesktopServer(("127.0.0.1", config["port"]), Handler)
    server.daemon_threads = True
    server.controller = controller
    server.token = token

    def unsloth_monitor():
        thermal_config = config.get("unsloth_thermal_config")
        if not thermal_config:
            return
        while not (ROOT / "stop").exists():
            if (Path(thermal_config).parent / "stop").exists():
                time.sleep(2)
                continue
            process = subprocess.Popen([sys.executable, "-X", "utf8", str(Path(__file__).with_name("thermal_unsloth.py")),
                "--config", thermal_config, "--desktop-pid", str(os.getpid())], creationflags=subprocess.CREATE_NO_WINDOW)
            while process.poll() is None and not (ROOT / "stop").exists():
                time.sleep(2)
            if (ROOT / "stop").exists():
                return
            time.sleep(30)

    def monitor():
        while True:
            if (ROOT / "stop").exists():
                if controller.background_cancel is not None:
                    controller.background_cancel.set()
                if controller.lock.acquire(blocking=False):
                    try:
                        if not controller.leases:
                            server.shutdown()
                            return
                    finally:
                        controller.lock.release()
            try:
                controller.sample()
                if controller.lock.acquire(blocking=False):
                    try:
                        controller.status = controller.switch()
                        controller.report()
                    finally:
                        controller.lock.release()
            except Exception as error:  # noqa: BLE001 - Retry after app or telemetry interruptions.
                controller.status = "deferred-" + type(error).__name__
                controller.report()
            time.sleep(config.get("poll_seconds", 2))

    def summaries():
        while True:
            time.sleep(15)
            record = None
            if not controller.lock.acquire(blocking=False):
                continue
            try:
                if controller.leases or not controller.config.get("share_memory", True) or not controller.config.get("background_summaries", True):
                    continue
                controller.status = controller.switch()
                if controller.status.startswith("waiting") or controller.status in ("unverified-model", "model-unloaded"):
                    continue
                pending = controller.store.pending()
                if pending is None:
                    continue
                record, digest, messages = pending
                save_json(ROOT / "memory-status.json", {"state": "summarizing", "record": record})
                cancel = Cancellation()
                controller.background_cancel = cancel
                source = _transcript(messages)
                position, previous = controller.store.summary_progress(record, digest)
                engine = controller.backend.engine
                if engine is None:
                    raise RuntimeError("Model not loaded")
                for _pass in range(8):
                    if cancel.is_set() or position >= len(source):
                        break
                    remaining = source[position:]
                    low, high = 0, len(remaining)
                    while low < high:
                        middle = (low + high + 1) // 2
                        if controller.backend.count(_summary_prompt(previous, remaining[:middle])) <= min(16000, engine["context"] - 1536):
                            low = middle
                        else:
                            high = middle - 1
                    if low == 0:
                        break
                    previous = controller.backend.summarize(_summary_prompt(previous, remaining[:low]), 1024, cancel).strip()
                    position += low
                    controller.store.summary_progress(record, digest, position, previous)
                    controller.sample()
                    controller.status = controller.switch()
                    controller.report()
                    if controller.status.startswith("waiting"):
                        break
                if position == len(source) and previous and not cancel.is_set():
                    saved = controller.store.finish_summary(record, digest, previous)
                    save_json(ROOT / "memory-status.json", {"state": "completed" if saved else "superseded", "record": record})
                else:
                    controller.store.defer_summary(record, 30)
                    save_json(ROOT / "memory-status.json", {"state": "checkpointed", "record": record, "position": position})
            except Exception as error:  # noqa: BLE001 - Preserve the original history if background work fails.
                if record is not None:
                    controller.store.defer_summary(record)
                logger.warning("Background memory summary deferred (%s)", type(error).__name__)
                save_json(ROOT / "memory-status.json", {"state": "deferred", "error": type(error).__name__})
            finally:
                controller.background_cancel = None
                controller.lock.release()

    def import_histories():
        while True:
            if controller.config.get("share_memory", True) and controller.lock.acquire(blocking=False):
                try:
                    if not controller.leases:
                        import_lmstudio(Path.home() / ".lmstudio", controller.store)
                        import_unsloth(StudioApi(config.get("unsloth_url", "http://127.0.0.1:8888")), controller.store)
                except (OSError, RuntimeError, ValueError, KeyError):
                    logger.warning("Desktop history import deferred")
                finally:
                    controller.lock.release()
            time.sleep(60)

    threading.Thread(target=monitor, daemon=True).start()
    threading.Thread(target=unsloth_monitor, daemon=True).start()
    threading.Thread(target=summaries, daemon=True).start()
    threading.Thread(target=import_histories, daemon=True).start()
    save_json(ROOT / "service.json", {"pid": __import__("os").getpid(), "port": config["port"]})
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
