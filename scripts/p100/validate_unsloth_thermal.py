import json
import time
import urllib.request
from pathlib import Path

from thermal_unsloth import StudioApi, gpu_samples, save_json


def main():
    root = Path.home() / ".llama.cpp/desktop"
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    thermal_path = Path(config["unsloth_thermal_config"])
    thermal = json.loads(thermal_path.read_text(encoding="utf-8"))
    api = StudioApi(config["unsloth_url"])
    initial = api.request("/api/inference/status")
    identifier = initial["model_identifier"]
    spec = config["unsloth_loads"][identifier]
    assert initial["gguf_variant"] == spec["gguf_variant"]
    started = time.monotonic()
    observations = []
    while time.monotonic() - started < 900:
        sample = gpu_samples()[thermal["primary_gpu_uuid"]]
        status = api.request("/api/inference/status")
        worker = json.loads((thermal_path.parent / "status.json").read_text(encoding="utf-8"))
        row = {"elapsed_seconds": round(time.monotonic() - started), "temperature": sample.temperature,
            "thermal_active": sample.thermal_active, "event": worker["event"], "hot": worker["hot"],
            "gpu_ids": status.get("gpu_ids"), "gpu_layers": status.get("gpu_layers"), "loading": bool(status.get("loading"))}
        observations.append(row)
        print(json.dumps(row), flush=True)
        if not status.get("loading") and status.get("model_identifier") == identifier and status.get("gpu_ids") == [sample.index, gpu_samples()[thermal["fallback_gpu_uuid"]].index] and status.get("gpu_layers") == 1024 and not worker["hot"]:
            break
        time.sleep(30)
    else:
        save_json(root / "unsloth-27b-recovery-result.json", {"verified": False, "reason": "Cooling condition was not reached", "observations": observations})
        raise RuntimeError("Cooling condition was not reached; fallback remains active")
    fields = ("model_identifier", "gguf_variant", "context_length", "gpu_ids", "gpu_layers", "offloaded_layers", "offload_total_layers", "cache_type_kv", "disable_vision")
    before = {key: status.get(key) for key in fields}
    assert before["context_length"] == 65536 and before["cache_type_kv"] == "f16" and not before["disable_vision"]
    body = {"model": identifier, "messages": [{"role": "user", "content": "What is 6 times 7? Reply only 42."}],
        "max_tokens": 128, "temperature": 0, "stream": False, "chat_template_kwargs": {"enable_thinking": False}, "thread_id": "thermal-27b-verification"}
    request = urllib.request.Request("http://127.0.0.1:" + str(config["port"]) + "/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + (root / "api-key.txt").read_text().strip(), "Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=300) as response:
        result = json.loads(response.read())
    assert result["model"] == identifier and result["choices"][0]["message"]["content"].strip() == "42"
    report = {"verified": True, "signal_source": "real GPU telemetry", "observations": observations, "status": before, "response": result}
    save_json(root / "unsloth-27b-recovery-result.json", report)
    print(json.dumps({"verified": True, "status": before, "text": "42"}), flush=True)


if __name__ == "__main__":
    main()
