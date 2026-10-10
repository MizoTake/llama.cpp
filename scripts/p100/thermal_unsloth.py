import argparse
import copy
import csv
import ctypes
import ipaddress
import json
import os
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

DEFAULT_DIRECTORY = Path.home() / ".unsloth" / "studio" / "p100-thermal"


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    temporary.replace(path)


@dataclass
class GpuSample:
    index: int
    uuid: str
    name: str
    temperature: int
    thermal_active: bool
    thermal_counter: int
    memory_total_mib: int


class ThermalPolicy:
    def __init__(self, return_temperature=68, stable_seconds=120, hot=False, fallback_temperature=None):
        self.return_temperature = return_temperature
        self.stable_seconds = stable_seconds
        self.hot = hot
        self.counter = None
        self.cool_since = None
        self.fallback_temperature = fallback_temperature

    def invalidate_cooling(self):
        self.cool_since = None

    def update(self, sample, now):
        increased = self.counter is not None and sample.thermal_counter > self.counter
        self.counter = sample.thermal_counter
        if sample.thermal_active or increased or (self.fallback_temperature is not None and sample.temperature >= self.fallback_temperature):
            self.hot = True
            self.cool_since = None
        elif self.hot and sample.temperature <= self.return_temperature:
            if self.cool_since is None:
                self.cool_since = now
            if now - self.cool_since >= self.stable_seconds:
                self.hot = False
                self.cool_since = None
        else:
            self.cool_since = None
        return self.hot


def fallback_override(original, index, batch_limit, gpu_layers=1024):
    config = copy.deepcopy(original)
    extras = config.get("llama_extra_args", []) or []
    cleaned = []
    controlled = {"--device", "-dev", "--main-gpu", "-mg", "--split-mode", "-sm", "--tensor-split", "-ts", "--batch-size", "-b", "--ubatch-size", "-ub", "--gpu-layers", "--n-gpu-layers", "-ngl", "--fit", "-fit"}
    position = 0
    while position < len(extras):
        argument = extras[position]
        if argument.split("=", 1)[0] in controlled:
            position += 1 if "=" in argument else 2
        else:
            cleaned.append(argument)
            position += 1
    config.update(gpu_ids=[index], gpu_index_kind="physical", gpu_memory_mode="manual", gpu_layers=gpu_layers, tensor_split=[1.0], tensor_parallel=False)
    config["n_batch"] = min(config.get("n_batch", 512), batch_limit)
    config["n_ubatch"] = min(config.get("n_ubatch", 512), config["n_batch"])
    config["llama_extra_args"] = cleaned + ["--split-mode", "none", "--fit", "off"]
    return config


class ProfileManager:
    def __init__(self, store, state_path, profiles):
        self.store = store
        self.state_path = state_path
        self.profiles = profiles
        self.state: dict = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {"hot": False, "managed": {}, "conflicts": []}
        self.hot = self.state["hot"]

    def sync(self, hot, primary_index, fallback_index):
        self.hot = hot
        self.state["hot"] = hot
        rows = self.store.get()
        managed: dict = self.state["managed"]
        conflicts: list = self.state["conflicts"]
        if hot:
            for model, limits in self.profiles.items():
                if model in conflicts:
                    continue
                current = rows.get(model, {})
                if model not in managed:
                    context = current.get("custom_context_length") or current.get("max_seq_length", 0)
                    if not 0 < context <= limits["context_limit"] or current.get("gpu_index_kind", "physical") != "physical" or primary_index not in current.get("gpu_ids", []):
                        continue
                    applied = fallback_override(current, fallback_index, limits["batch_limit"], limits.get("fallback_gpu_layers", 1024))
                    if hasattr(self.store, "normalize"):
                        applied = self.store.normalize(applied)
                    managed[model] = {"original": current, "applied": applied}
                    # Save the original first so a process exit cannot lose the normal profile.
                    save_json(self.state_path, self.state)
                record = managed[model]
                if current == record["original"]:
                    self.store.set(model, record["applied"])
                elif current != record["applied"] and model not in conflicts:
                    conflicts.append(model)
        else:
            for model, record in list(managed.items()):
                if rows.get(model) == record["applied"]:
                    self.store.set(model, record["original"])
                elif rows.get(model) != record["original"] and model not in conflicts:
                    conflicts.append(model)
                del managed[model]
        save_json(self.state_path, self.state)


class Controller:
    def __init__(self, api, manager, make_load_kwargs, policy=None):
        self.api = api
        self.manager = manager
        self.make_load_kwargs = make_load_kwargs
        self.policy = policy or ThermalPolicy(hot=manager.hot)

    def step(self, sample, now, fallback_index):
        hot = self.policy.update(sample, now)
        self.manager.sync(hot, sample.index, fallback_index)
        status = self.api.request("/api/inference/status")
        if not status.get("active_model"):
            return "unloaded"
        if not status.get("is_gguf") or status.get("loading") or len(status.get("serving", [])) > 1:
            return "waiting-model"
        key, config = self.manager.store.resolve(status)
        if key not in self.manager.profiles or key in self.manager.state["conflicts"]:
            return "unverified-profile"
        context = status.get("context_length", 0)
        if not context or context > self.manager.profiles[key]["context_limit"]:
            return "unverified-context"
        desired = config.get("gpu_ids", [])
        layers_match = status.get("gpu_layers") is None or status["gpu_layers"] == config.get("gpu_layers")
        if not desired or (status.get("gpu_ids") == desired and layers_match):
            return "matched"
        liveness = self.api.request("/api/liveness")
        inference_active = liveness.get("inference_active", liveness.get("status") != "alive")
        if inference_active or self.api.request("/api/train/status").get("is_training_running", True):
            return "waiting-idle"
        fresh = self.api.request("/api/inference/status")
        identity = ("active_model", "model_identifier", "gguf_variant", "context_length", "gpu_ids", "gpu_layers")
        if any(fresh.get(field) != status.get(field) for field in identity) or fresh.get("loading") or len(fresh.get("serving", [])) > 1:
            return "waiting-model"
        payload = self.make_load_kwargs(config)
        payload.update(model_path=status.get("model_identifier") or status["active_model"], gguf_variant=status.get("gguf_variant"), max_seq_length=context, force_reload=True, force_cancel_active=False)
        try:
            self.api.request("/api/inference/load", payload)
        except urllib.error.HTTPError as error:
            if error.code == 409:
                return "waiting-idle"
            raise
        return "switched-fallback" if hot else "switched-primary"


class StudioStore:
    def __init__(self):
        backend = Path(sys.prefix) / "Lib" / "site-packages" / "studio" / "backend"
        if not backend.is_dir():
            raise RuntimeError("Use the Unsloth Desktop Python environment.")
        sys.path.insert(0, str(backend))
        from utils.account_context import OWNER, run_as
        from utils.openai_auto_switch_settings import (
            get_model_overrides,
            model_override_load_kwargs,
            normalize_model_override,
            resolve_override_for_load,
            set_model_override,
        )
        self.owner = OWNER
        self.run_as = run_as
        self.read = get_model_overrides
        self.write = set_model_override
        self.lookup = resolve_override_for_load
        self.load_kwargs = model_override_load_kwargs
        self.normalize = normalize_model_override

    def get(self):
        return self.run_as(self.owner, self.read)

    def set(self, model, config):
        values = copy.deepcopy(config)
        extras = values.pop("llama_extra_args", [])
        context = values.pop("max_seq_length", None)
        self.run_as(self.owner, self.write, model, extras, context, keep_empty_extra_args=True, **values)

    def resolve(self, status):
        return self.run_as(self.owner, self.lookup, status.get("model_identifier") or status["active_model"], status["active_model"], status.get("gguf_variant"))

    def make_load_kwargs(self, config):
        return self.load_kwargs(config, is_gguf=True)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class StudioApi:
    def __init__(self, base):
        parsed = urlsplit(base)
        if parsed.scheme != "http" or not parsed.hostname or not ipaddress.ip_address(parsed.hostname).is_loopback or parsed.username or parsed.password or parsed.path not in ("", "/") or parsed.query or parsed.fragment:
            raise ValueError("The Desktop API must be an HTTP loopback address.")
        self.base = base.rstrip("/")
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        self.token = None

    def request(self, route, payload=None):
        try:
            return self.authenticated_request(route, payload)
        except urllib.error.HTTPError as error:
            if error.code != 401:
                raise
            self.token = None
            return self.authenticated_request(route, payload)

    def authenticated_request(self, route, payload):
        if not self.token:
            secret = (Path.home() / ".unsloth" / "studio" / "auth" / ".desktop_secret").read_text(encoding="utf-8").strip()
            self.token = self._request("/api/auth/desktop-login", {"secret": secret})["access_token"]
        return self._request(route, payload)

    def _request(self, route, payload):
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(self.base + route, data=data, headers={"Content-Type": "application/json"})
        if self.token:
            request.add_header("Authorization", "Bearer " + self.token)
        with self.opener.open(request, timeout=300 if route.endswith("/load") else 15) as response:
            return json.load(response)


def gpu_samples():
    fields = "index,uuid,name,temperature.gpu,clocks_event_reasons.sw_thermal_slowdown,clocks_event_reasons.hw_thermal_slowdown,clocks_event_reasons_counters.sw_thermal_slowdown,clocks_event_reasons_counters.hw_thermal_slowdown,memory.total"
    result = subprocess.run(["nvidia-smi", "--query-gpu=" + fields, "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10, check=True, creationflags=subprocess.CREATE_NO_WINDOW)
    samples = {}
    for row in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
        sw, hw = row[4:6]
        if sw not in ("Active", "Not Active") or hw not in ("Active", "Not Active"):
            raise RuntimeError("GPU thermal flags are unavailable.")
        counters = [int(value) for value in row[6:8] if value.strip().isdigit()]
        samples[row[1]] = GpuSample(int(row[0]), row[1], row[2], int(row[3]), "Active" in (sw, hw), sum(counters), int(row[8]))
    return samples


def process_alive(pid):
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        return False
    try:
        code = ctypes.c_ulong()
        return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
    finally:
        kernel.CloseHandle(handle)


def validate_config(config):
    if config["primary_gpu_uuid"] == config["fallback_gpu_uuid"]:
        raise ValueError("Primary and fallback GPUs must differ.")
    if not 1 <= config.get("poll_seconds", 2) <= 60 or not 1 <= config.get("stable_seconds", 120) <= 3600 or not 1 <= config.get("return_temperature", 68) <= 90:
        raise ValueError("Invalid polling or cooling settings.")
    if config.get("fallback_temperature") is not None and not config.get("return_temperature", 68) < config["fallback_temperature"] <= 90:
        raise ValueError("Fallback temperature must exceed the return temperature.")
    for limits in config["profiles"].values():
        if not 1 <= limits["context_limit"] <= 1048576 or not 1 <= limits["batch_limit"] <= 65536 or not 0 <= limits.get("fallback_gpu_layers", 1024) <= 1024:
            raise ValueError("Invalid verified profile limits.")


def run(config_path, desktop_pid):
    import msvcrt
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    directory = config_path.parent
    lock = (directory / "monitor.lock").open("a+b")
    lock.seek(0)
    if not lock.read(1):
        lock.write(b"0")
        lock.flush()
    lock.seek(0)
    try:
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        lock.close()
        return
    stop = directory / "stop"
    stop.unlink(missing_ok=True)
    store = StudioStore()
    manager = ProfileManager(store, directory / "state.json", config["profiles"])
    policy = ThermalPolicy(config.get("return_temperature", 68), config.get("stable_seconds", 120), manager.hot, config.get("fallback_temperature"))
    if config.get("shared_status"):
        try:
            policy.hot = policy.hot or json.loads(Path(config["shared_status"]).read_text(encoding="utf-8")).get("hot") is True
        except (OSError, ValueError):
            pass
    controller = Controller(StudioApi(config.get("api", "http://127.0.0.1:8888")), manager, store.make_load_kwargs, policy)
    previous = None

    def report(event, **fields):
        nonlocal previous
        status = {"timestamp": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(), "desktop_pid": desktop_pid, "event": event, "hot": policy.hot, **fields}
        save_json(directory / "status.json", status)
        if event != previous:
            with (directory / "events.jsonl").open("a", encoding="utf-8", newline="\n") as output:
                output.write(json.dumps(status) + "\n")
            previous = event

    try:
        report("starting")
        while not stop.exists() and (not desktop_pid or process_alive(desktop_pid)):
            try:
                samples = gpu_samples()
                primary = samples[config["primary_gpu_uuid"]]
                fallback = samples[config["fallback_gpu_uuid"]]
            except (OSError, ValueError, RuntimeError, KeyError, IndexError, subprocess.SubprocessError) as error:
                policy.invalidate_cooling()
                report("telemetry-unavailable", error_type=type(error).__name__)
            else:
                try:
                    result = controller.step(primary, time.monotonic(), fallback.index)
                    report(result, temperature=primary.temperature, thermal_active=primary.thermal_active, primary_index=primary.index, fallback_index=fallback.index, conflicts=manager.state["conflicts"])
                except (OSError, ValueError, RuntimeError, KeyError, sqlite3.Error, urllib.error.URLError) as error:
                    report("switch-error", error_type=type(error).__name__, http_status=getattr(error, "code", None))
                    time.sleep(10)
            time.sleep(config.get("poll_seconds", 2))
    finally:
        try:
            manager.sync(False, -1, -1)
            report("stopped-profiles-restored", conflicts=manager.state["conflicts"])
        finally:
            lock.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_DIRECTORY / "config.json")
    parser.add_argument("--desktop-pid", type=int)
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    if args.stop:
        args.config.parent.mkdir(parents=True, exist_ok=True)
        (args.config.parent / "stop").touch()
    elif args.status:
        status = json.loads((args.config.parent / "status.json").read_text(encoding="utf-8"))
        status["process_alive"] = process_alive(status["pid"])
        print(json.dumps(status, indent=2))
    else:
        run(args.config.resolve(), args.desktop_pid)


if __name__ == "__main__":
    main()
