import argparse
import hashlib
import http.client
import json
import ntpath
import re
import subprocess
import threading
import time
from pathlib import Path

from desktop_service import LmBackend
from thermal_unsloth import StudioApi, StudioStore, fallback_override, gpu_samples, process_alive, save_json


def argument(command, key):
    match = re.search(r"(?:^|\s)" + re.escape(key) + r'\s+(?:"([^\"]*)"|(\S+))', command)
    return next((value for value in match.groups() if value is not None), None) if match else None


def native_process(model_file):
    result = subprocess.run(["powershell.exe", "-NoProfile", "-Command", "Get-CimInstance Win32_Process -Filter \"Name='llama-server.exe'\" | Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"], capture_output=True, text=True, encoding="utf-8", check=True, creationflags=subprocess.CREATE_NO_WINDOW)
    rows = json.loads(result.stdout)
    matches = [row for row in (rows if isinstance(rows, list) else [rows]) if ntpath.normcase(argument(row["CommandLine"], "--model") or argument(row["CommandLine"], "-m") or "") == ntpath.normcase(model_file)]
    if len(matches) != 1:
        raise RuntimeError("Benchmark native model identity is ambiguous")
    row = matches[0]
    command = row["CommandLine"]
    key_file = argument(command, "--api-key-file")
    key = Path(key_file).read_text(encoding="utf-8").strip() if key_file else argument(command, "--api-key")
    metadata = {"pid": row["ProcessId"], "model_file": model_file,
        "device": argument(command, "--device"), "gpu_layers": argument(command, "--n-gpu-layers") or argument(command, "--gpu-layers"),
        "tensor_split": argument(command, "--tensor-split"), "split_mode": argument(command, "--split-mode"),
        "cache_ram_mib": argument(command, "--cache-ram"), "ctx_checkpoints": argument(command, "--ctx-checkpoints"),
        "context": argument(command, "--ctx-size") or argument(command, "-c"),
        "cache_type_k": argument(command, "--cache-type-k") or argument(command, "-ctk") or "f16",
        "cache_type_v": argument(command, "--cache-type-v") or argument(command, "-ctv") or "f16"}
    return int(argument(command, "--port")), key, metadata


def generate(port, key, messages):
    body = {"messages": messages, "max_tokens": 16, "temperature": 0, "seed": 1234, "stream": True,
        "cache_prompt": True, "chat_template_kwargs": {"enable_thinking": False}}
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    started = time.perf_counter()
    connection.request("POST", "/v1/chat/completions", json.dumps(body), headers)
    response = connection.getresponse()
    if response.status != 200:
        connection.close()
        raise RuntimeError("Benchmark returned HTTP " + str(response.status))
    text = []
    first = None
    timings = {}
    usage = {}
    finish = None
    try:
        while line := response.readline():
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            value = json.loads(line[6:])
            timings = value.get("timings", timings)
            usage = value.get("usage") or usage
            for choice in value.get("choices", []):
                chunk = choice.get("delta", {}).get("content")
                if chunk:
                    first = first if first is not None else time.perf_counter() - started
                    text.append(chunk)
                finish = choice.get("finish_reason") or finish
    finally:
        connection.close()
    output = "".join(text).strip()
    if not output or finish != "stop" or first is None:
        raise RuntimeError("Benchmark response was empty or incomplete")
    return {"first_token_ms": round(first * 1000, 2), "total_ms": round((time.perf_counter() - started) * 1000, 2),
        "timings": timings, "usage": usage, "output": output, "output_sha256": hashlib.sha256(output.encode()).hexdigest(), "finish_reason": finish}


def fixture(label, answer, reference_rows=256):
    rows = "\n".join(f"{label}-{i:04}: cedar amber river stone violet {i * 17}" for i in range(reference_rows))
    return [{"role": "system", "content": f"Reference document {label}. Follow the user's final instruction exactly."},
        {"role": "user", "content": rows + f"\nIgnore the reference rows for the answer. Reply with exactly {answer}, nothing else."}]


def ensure_apps_idle(root, api, backend):
    service = root / "service.json"
    if service.exists() and process_alive(json.loads(service.read_text(encoding="utf-8"))["pid"]):
        raise RuntimeError("Stop the common Desktop service before the controlled benchmark")
    uf = api.request("/api/inference/status")
    if uf.get("loading") or api.request("/api/train/status").get("is_training_running", True):
        raise RuntimeError("Wait for the current Unsloth load or training")
    if uf.get("active_model") and api.request("/api/liveness").get("inference_active"):
        raise RuntimeError("Wait for the current Unsloth generation")
    try:
        engine = backend.bridge()
    except RuntimeError as error:
        if str(error) != "Model not loaded":
            raise
        engine = None
    if engine is not None and not engine["idle"]:
        raise RuntimeError("Wait for the current LM Studio generation")
    return uf, engine


def ensure_native_idle(port, key):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=15)
    try:
        connection.request("GET", "/slots", headers={"Authorization": "Bearer " + key} if key else {})
        response = connection.getresponse()
        slots = json.loads(response.read())
        if response.status != 200 or not isinstance(slots, list) or len(slots) != 1 or slots[0].get("is_processing"):
            raise RuntimeError("Benchmark needs one idle native slot")
        return len(slots)
    finally:
        connection.close()


def thermal_sample(uuid):
    fields = "temperature.gpu,clocks.sm,utilization.gpu,clocks_event_reasons.sw_thermal_slowdown,clocks_event_reasons.hw_thermal_slowdown,clocks_event_reasons_counters.sw_thermal_slowdown,clocks_event_reasons_counters.hw_thermal_slowdown,memory.used,power.draw"
    result = subprocess.run(["nvidia-smi", "--id=" + uuid, "--query-gpu=" + fields, "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10, check=True, creationflags=subprocess.CREATE_NO_WINDOW)
    row = [value.strip() for value in result.stdout.strip().split(",")]
    if len(row) != 9 or any(flag not in ("Active", "Not Active") for flag in row[3:5]):
        raise RuntimeError("Primary GPU telemetry is unavailable")
    counters = [int(value) for value in row[5:7] if value.isdigit()]
    if not counters:
        raise RuntimeError("Primary GPU thermal counters are unavailable")
    return {"temperature": int(row[0]), "sm_clock_mhz": int(row[1]), "utilization": int(row[2]),
        "thermal_active": "Active" in row[3:5], "thermal_counter": sum(counters),
        "memory_used_mib": int(row[7]), "power_w": float(row[8]) if row[8] != "[N/A]" else None}


def wait_cool(uuid, temperature, timeout):
    started = time.monotonic()
    announced = 0
    while True:
        sample = thermal_sample(uuid)
        if sample["temperature"] <= temperature and not sample["thermal_active"]:
            return sample
        elapsed = time.monotonic() - started
        if elapsed >= timeout:
            raise RuntimeError("Primary GPU did not cool within the benchmark timeout")
        if elapsed >= announced:
            print(json.dumps({"cooling_seconds": round(elapsed), "temperature": sample["temperature"], "target": temperature}), flush=True)
            announced += 30
        time.sleep(2)


def generate_monitored(port, key, messages, uuid):
    started = time.monotonic()
    samples = [{"seconds": 0, **thermal_sample(uuid)}]
    if samples[0]["temperature"] >= 78 or samples[0]["thermal_active"]:
        raise RuntimeError("Primary GPU reached the thermal boundary before generation")
    stop = threading.Event()
    errors = []

    def collect():
        while not stop.wait(1):
            try:
                samples.append({"seconds": round(time.monotonic() - started, 3), **thermal_sample(uuid)})
            except Exception as error:
                errors.append(type(error).__name__)
                return

    thread = threading.Thread(target=collect, daemon=True)
    thread.start()
    try:
        result = generate(port, key, messages)
    finally:
        stop.set()
        thread.join(timeout=15)
    if thread.is_alive() or errors:
        raise RuntimeError("Primary GPU monitoring failed during generation")
    samples.append({"seconds": round(time.monotonic() - started, 3), **thermal_sample(uuid)})
    limited = any(sample["thermal_active"] or sample["thermal_counter"] > samples[0]["thermal_counter"] for sample in samples)
    result["thermal"] = {"unthrottled": not limited, "temperature_limit_reached": any(sample["temperature"] >= 78 for sample in samples), "samples": samples}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--app", choices=["lmstudio", "unsloth"], required=True)
    parser.add_argument("--model", help="Registered model identifier; defaults to the app's 9B model.")
    parser.add_argument("--gpu-profile", choices=["primary", "fallback"], default="fallback")
    parser.add_argument("--cool-temperature", type=int, default=69)
    parser.add_argument("--initial-cool-temperature", type=int, default=69)
    parser.add_argument("--cooldown-timeout", type=int, default=600)
    parser.add_argument("--cache-mib", type=int, required=True)
    parser.add_argument("--checkpoints", type=int, help="Override the Unsloth checkpoint count for a controlled comparison.")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--reference-rows", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.cache_mib <= 65536 or not 1 <= args.rounds <= 10:
        parser.error("Use 0-65536 MiB and 1-10 rounds")
    if not 16 <= args.reference_rows <= 1024:
        parser.error("Use 16-1024 reference rows")
    if args.checkpoints is not None and (args.app != "unsloth" or not 0 <= args.checkpoints <= 256):
        parser.error("Use 0-256 checkpoints with Unsloth")
    if not 20 <= args.cool_temperature <= 77 or not 10 <= args.cooldown_timeout <= 3600:
        parser.error("Use a cooling target of 20-77C and timeout of 10-3600 seconds")
    if not 20 <= args.initial_cool_temperature <= args.cool_temperature:
        parser.error("The initial cooling target must be 20C or above and no higher than the per-request target")
    root = Path.home() / ".llama.cpp/desktop"
    service = root / "service.json"
    if service.exists() and process_alive(json.loads(service.read_text(encoding="utf-8"))["pid"]):
        raise RuntimeError("Stop the common Desktop service before the controlled benchmark")
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    backend = LmBackend(config)
    hot = args.gpu_profile == "fallback"
    primary_uuid = config["primary_uuid"]
    api = StudioApi(config["unsloth_url"])
    uf, engine = ensure_apps_idle(root, api, backend)
    if args.app == "lmstudio":
        if uf.get("active_model"):
            api.request("/api/inference/unload", {"model_path": uf["model_identifier"], "force_cancel_active": False})
        backend.config["ram_cache_mib"] = args.cache_mib
        identifier = args.model or "qwen/qwen3.5-9b"
        if identifier not in config["lmstudio"]["models"]:
            raise RuntimeError("Unregistered benchmark model")
        if engine is not None:
            backend.bridge("unload")
        if not hot:
            wait_cool(primary_uuid, args.initial_cool_temperature, args.cooldown_timeout)
        current_uf, current_engine = ensure_apps_idle(root, api, backend)
        if current_uf.get("active_model") or current_engine is not None:
            raise RuntimeError("Another model was loaded during the cooling wait")
        backend.selected = identifier
        engine = backend.bridge("load", hot)
        model_file = engine["native_model"]
    else:
        if engine is not None:
            backend.bridge("unload")
        if uf.get("active_model"):
            api.request("/api/inference/unload", {"model_path": uf["model_identifier"], "force_cancel_active": False})
        if not hot:
            wait_cool(primary_uuid, args.initial_cool_temperature, args.cooldown_timeout)
        current_uf, current_engine = ensure_apps_idle(root, api, backend)
        if current_uf.get("active_model") or current_engine is not None:
            raise RuntimeError("Another model was loaded during the cooling wait")
        identifier = args.model or "unsloth/Qwen3.5-9B-GGUF"
        if identifier not in config["unsloth_loads"]:
            raise RuntimeError("Unregistered benchmark model")
        store = StudioStore()
        thermal = json.loads(Path(config["unsloth_thermal_config"]).read_text(encoding="utf-8"))
        profile = thermal["profiles"][identifier]
        row = store.get()[identifier]
        if hot:
            row = fallback_override(row, gpu_samples()[thermal["fallback_gpu_uuid"]].index, profile["batch_limit"], profile.get("fallback_gpu_layers", 1024))
        elif row.get("gpu_ids", [])[:1] != [gpu_samples()[primary_uuid].index] or len(row["gpu_ids"]) != 2:
            raise RuntimeError("Saved primary profile must place the P100 first and use both GPUs")
        row.update(cache_ram=args.cache_mib, ctx_checkpoints=args.checkpoints if args.checkpoints is not None else (32 if args.cache_mib else 0))
        row["llama_extra_args"] = [arg for arg in row.get("llama_extra_args", []) if arg not in ("--cache-idle-slots", "--no-cache-idle-slots")]
        if args.cache_mib:
            row["llama_extra_args"].append("--cache-idle-slots")
        spec = config["unsloth_loads"][identifier]
        payload = {**store.make_load_kwargs(row), "model_path": identifier, "gguf_variant": spec["gguf_variant"], "force_cancel_active": False}
        api.request("/api/inference/load", payload)
        model_file = spec["model_file"]
    port, key, metadata = native_process(model_file)
    if int(metadata["cache_ram_mib"] or 8192) != args.cache_mib:
        raise RuntimeError("Native RAM cache does not match the requested setting")
    native_signature = dict(metadata)
    metadata["slots"] = ensure_native_idle(port, key)
    if not hot:
        if args.app == "lmstudio" and metadata["device"] != config["primary_devices"]:
            raise RuntimeError("Native primary GPU placement did not match")
        if args.app == "unsloth":
            state = api.request("/api/inference/status")
            metadata["gpu_ids"] = state.get("gpu_ids")
            if state.get("gpu_ids") != row["gpu_ids"] or state.get("gpu_layers") != row["gpu_layers"]:
                raise RuntimeError("Unsloth primary GPU placement did not match")
    report = {"app": args.app, "gpu_profile": args.gpu_profile, "requested_cache_mib": args.cache_mib, "rounds": args.rounds, "reference_rows": args.reference_rows, "initial_cool_temperature": args.initial_cool_temperature, "cool_temperature": args.cool_temperature, "native": metadata, "cases": []}
    for number in range(args.rounds):
        a = fixture(f"A{number}", "42", args.reference_rows)
        b = fixture(f"B{number}", "84", args.reference_rows)
        for stage, messages in [("cold_a", a), ("divert_b", b), ("return_a", a), ("same_a", a)]:
            if not hot:
                wait_cool(primary_uuid, args.cool_temperature, args.cooldown_timeout)
            current_uf, current_engine = ensure_apps_idle(root, api, backend)
            if (args.app == "unsloth" and current_engine is not None) or (args.app == "lmstudio" and current_uf.get("active_model")):
                raise RuntimeError("Another app loaded a model during the comparison")
            current_port, current_key, current_metadata = native_process(model_file)
            if (current_port, current_key, current_metadata) != (port, key, native_signature):
                raise RuntimeError("Native profile changed during the comparison")
            ensure_native_idle(port, key)
            row: dict = {"round": number, "stage": stage, **(generate(port, key, messages) if hot else generate_monitored(port, key, messages, primary_uuid))}
            if row["output"] != ("84" if stage == "divert_b" else "42"):
                raise RuntimeError("Benchmark output did not match the reference answer")
            report["cases"].append(row)
            save_json(args.output, report)
            print(json.dumps({key: row[key] for key in ["round", "stage", "first_token_ms", "total_ms"]} | {"cached_tokens": row["timings"].get("cache_n"), "processed_tokens": row["timings"].get("prompt_n")}), flush=True)
            if not hot and (not row["thermal"]["unthrottled"] or row["thermal"]["temperature_limit_reached"]):
                raise RuntimeError("Primary GPU reached the thermal boundary; cool and repeat this comparison")
    print(json.dumps({"app": args.app, "cache_mib": args.cache_mib, "report": str(args.output), "verified": True}), flush=True)


if __name__ == "__main__":
    main()
