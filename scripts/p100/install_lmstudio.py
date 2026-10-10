import argparse
import base64
import copy
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

NAME = "llama.cpp-win-x86_64-nvidia-cuda12-avx2-p100"
VERSION = "2.47.100"
BASE_NAME = "llama.cpp-win-x86_64-nvidia-cuda12-avx2-2.47.0"
FILES = (
    "llama-server.exe", "cublas64_12.dll", "cublasLt64_12.dll",
    "cudart64_12.dll", "libcrypto-3-x64.dll", "libssl-3-x64.dll",
)


def digest(path):
    with path.open("rb") as source:
        digest = hashlib.sha256()
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
        return digest.hexdigest()


def read_json(path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else copy.deepcopy(default)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as output:
        temporary = Path(output.name)
        output.write((json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def cli(home, *arguments):
    result = subprocess.run([str(home / "bin/lms.exe"), *arguments], capture_output=True, text=True, encoding="utf-8", check=True)
    return result.stdout


def check_idle(home):
    if json.loads(cli(home, "ps", "--json")):
        raise RuntimeError("Unload LM Studio models before changing the runtime defaults.")
    settings = read_json(home / "settings.json", {})
    if settings.get("useLlamaCppEngineProtocolRuntime3") is not True:
        raise RuntimeError("Enable LM Studio Engine Protocol in Developer settings first.")


def select_runtime(home, selection):
    cli(home, "runtime", "select", f"{selection['name']}@{selection['version']}")


def discover_devices(server):
    result = subprocess.run([str(server), "--list-devices"], capture_output=True, text=True, encoding="utf-8", check=True)
    devices = re.findall(r"(CUDA\d+): ([^\r\n]+)", result.stdout + result.stderr)
    primary = [device for device, name in devices if name.startswith("Tesla P100")]
    secondary = [device for device, name in devices if "RTX 4070" in name]
    if len(primary) != 1 or len(secondary) != 1:
        raise RuntimeError("Expected one P100 and one RTX 4070 in the package device list.")
    return primary + secondary


def tuned_config(original, devices):
    result = copy.deepcopy(original)
    overrides = {
        "enabled": True, "excludeAllConfig": False,
        "disabledParameters": ["--device", "--tensor-split", "--split-mode", "--main-gpu", "--fit"],
        "overrideParameters": [
            {"key": "--device", "value": ",".join(devices)},
            {"key": "--tensor-split", "value": "3,2"},
            {"key": "--split-mode", "value": "layer"},
            {"key": "--main-gpu", "value": "0"},
            {"key": "--fit", "value": "off"},
        ],
    }
    values = {
        "llm.load.contextLength": 65536,
        "llm.load.numParallelSessions": 1,
        "llm.load.llama.autoFit": False,
        "llm.load.llama.acceleration.offloadRatio": 1,
        "llm.load.llama.cpuThreadPoolSize": 6,
        "llm.load.llama.evalBatchSize": 2048,
        "llm.load.llama.physicalBatchSize": 2048,
        "llm.load.llama.flashAttention": True,
        "llm.load.llama.kCacheQuantizationType": {"checked": True, "value": "f16"},
        "llm.load.llama.vCacheQuantizationType": {"checked": True, "value": "f16"},
        "llm.load.llama.argumentsOverride": overrides,
    }
    fields = result.setdefault("load", {}).setdefault("fields", [])
    for field in fields:
        if field["key"] in values:
            field["value"] = values.pop(field["key"])
    fields.extend({"key": key, "value": value} for key, value in values.items())
    return result


def hardware_entry():
    return {"fields": [
        {"key": "load.gpuSplitConfig", "value": {"strategy": "evenly", "disabledGpus": [], "priority": [], "customRatio": []}},
        {"key": "load.gpuStrictVramCap", "value": False},
    ]}


def install(home, base, package, devices, dry_run=False):
    state = home / "p100-custom/installation.json"
    if state.exists():
        record = read_json(state, {})
        for name, expected in record["hashes"].items():
            if digest(Path(record["target"]) / name) != expected:
                raise RuntimeError("Installed runtime changed; restore or inspect it first.")
        if digest(Path(record["model_file"])) != record["model_after"]:
            raise RuntimeError("Managed model settings changed; inspect them first.")
        hardware = read_json(home / ".internal/hardware-config.json", {})
        entry = next((value for name, value in hardware["json"] if name == NAME), None)
        if entry != record["hardware_after"]:
            raise RuntimeError("Managed GPU settings changed; inspect them first.")
        if any(digest(package / name) != record["hashes"]["p100/" + name] for name in FILES):
            raise RuntimeError("Different package; restore and remove the previous custom runtime first.")
        return record
    checksums = dict(line.split("  ", 1)[::-1] for line in (package / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines())
    for name in FILES:
        if checksums.get(name) != digest(package / name):
            raise RuntimeError(f"Package checksum mismatch: {name}")
    manifest = read_json(base / "backend-manifest.json", {})
    if manifest.get("version") != "2.47.0" or manifest.get("engine_protocol_server", {}).get("runtime_kind") != "llama-server":
        raise RuntimeError("Unsupported stock runtime metadata.")
    target = home / "extensions/backends" / f"{NAME}-{VERSION}"
    if target.exists():
        raise RuntimeError("Unmanaged custom runtime directory already exists.")
    model = home / ".internal/user-concrete-model-default-config/qwen/qwen3.5-9b.json"
    hardware = home / ".internal/hardware-config.json"
    preferences = home / ".internal/backend-preferences-v1.json"
    hardware_data = read_json(hardware, {"json": [], "meta": {"values": ["map"]}})
    if any(name == NAME for name, _ in hardware_data["json"]):
        raise RuntimeError("Unmanaged custom GPU configuration already exists.")
    record = {
        "target": str(target), "package": str(package), "base": str(base), "devices": devices,
        "model_file": str(model), "model_before": base64.b64encode(model.read_bytes()).decode() if model.exists() else None,
        "selection_before": next((item for item in read_json(preferences, []) if item["model_format"] == "gguf"), None),
        "hashes": {}, "model_after": None, "hardware_after": hardware_entry(),
    }
    if dry_run:
        return record
    check_idle(home)
    staging_parent = home / "p100-custom"
    staging_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=staging_parent) as temporary:
        staged = Path(temporary) / "runtime"
        shutil.copytree(base, staged)
        (staged / "p100").mkdir()
        for name in FILES:
            shutil.copy2(package / name, staged / "p100" / name)
        manifest.update(name=NAME, version=VERSION)
        manifest["gpu"]["targets"] = ["6.0", "8.9"]
        manifest["engine_protocol_server"]["executable_relative_path"] = "p100/llama-server.exe"
        write_json(staged / "backend-manifest.json", manifest)
        write_json(staged / "engine-protocol-server-artifacts.json", {
            "schema_version": 1, "runtime_kind": "llama-server", "executable_relative_path": "p100/llama-server.exe",
            "files": [{"relative_path": "p100/" + name, "executable": name.endswith(".exe")} for name in FILES],
        })
        display = read_json(staged / "display-data.json", [])
        for _, entry in display:
            entry.update(displayName="P100 + RTX 4070 custom llama.cpp (CUDA 12.9)", description="Local SM60/SM89 server through LM Studio Engine Protocol.")
            entry["releaseNotes"] = [{"version": VERSION, "releaseNotes": "Uses the local P100-first package through llama-server."}]
        write_json(staged / "display-data.json", display)
        record["hashes"] = {str(path.relative_to(staged)).replace("\\", "/"): digest(path) for path in staged.rglob("*") if path.is_file()}
        target.parent.mkdir(parents=True, exist_ok=True)
        staged.rename(target)
    hardware_before = hardware.read_bytes() if hardware.exists() else None
    try:
        write_json(model, tuned_config(read_json(model, {"preset": "", "operation": {"fields": []}, "load": {"fields": []}}), devices))
        record["model_after"] = digest(model)
        hardware_data["json"].append([NAME, record["hardware_after"]])
        write_json(hardware, hardware_data)
        write_json(state, record)
    except Exception:
        if record["model_before"] is None:
            model.unlink(missing_ok=True)
        else:
            model.write_bytes(base64.b64decode(record["model_before"]))
        if hardware_before is None:
            hardware.unlink(missing_ok=True)
        else:
            hardware.write_bytes(hardware_before)
        failed = Path(tempfile.mkdtemp(prefix="failed-", dir=staging_parent))
        if target.resolve().parent != (home / "extensions/backends").resolve():
            raise RuntimeError("Refusing to move a runtime outside the backend directory.") from None
        target.rename(failed / target.name)
        raise
    return record


def restore(home):
    state = home / "p100-custom/installation.json"
    record = read_json(state, {})
    target = (home / "extensions/backends" / f"{NAME}-{VERSION}").resolve()
    model = home / ".internal/user-concrete-model-default-config/qwen/qwen3.5-9b.json"
    if Path(record["target"]).resolve() != target or Path(record["model_file"]).resolve() != model.resolve():
        raise RuntimeError("Installation paths do not match this LM Studio home.")
    hardware = home / ".internal/hardware-config.json"
    hardware_data = read_json(hardware, {})
    entry = next((value for name, value in hardware_data["json"] if name == NAME), None)
    if digest(model) != record["model_after"] or entry != record["hardware_after"]:
        raise RuntimeError("Managed settings changed; restore would overwrite edits.")
    check_idle(home)
    selection = next((item for item in read_json(home / ".internal/backend-preferences-v1.json", []) if item["model_format"] == "gguf"), None)
    own_selection = {"model_format": "gguf", "name": NAME, "version": VERSION}
    if selection and selection not in (own_selection, record["selection_before"]):
        raise RuntimeError("Runtime selection changed; restore would overwrite it.")
    if record["selection_before"]:
        select_runtime(home, record["selection_before"])
    if record["model_before"] is None:
        model.unlink()
    else:
        model.write_bytes(base64.b64decode(record["model_before"]))
    hardware_data["json"] = [item for item in hardware_data["json"] if item[0] != NAME]
    write_json(hardware, hardware_data)
    state.rename(state.with_name("restored-installation.json"))
    return {"restored": True, "runtime_retained": str(target)}


def main():
    parser = argparse.ArgumentParser(description="Add a P100-first server runtime to LM Studio without replacing stock libraries.")
    parser.add_argument("--home", type=Path, default=Path.home() / ".lmstudio")
    parser.add_argument("--package", type=Path)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--restore", action="store_true")
    args = parser.parse_args()
    args.home = args.home.resolve()
    if args.restore:
        print(json.dumps(restore(args.home)))
        return
    if not args.package:
        parser.error("--package is required when installing")
    args.package = args.package.resolve()
    devices = discover_devices(args.package / "llama-server.exe")
    record = install(args.home, args.base.resolve() if args.base else args.home / "extensions/backends" / BASE_NAME, args.package, devices, args.dry_run)
    print(json.dumps({"target": record["target"], "devices": devices, "dry_run": args.dry_run, "selection": f"{NAME}@{VERSION}"}))


if __name__ == "__main__":
    main()
