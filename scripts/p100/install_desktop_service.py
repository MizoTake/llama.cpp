import argparse
import hashlib
import json
import secrets
import shutil
import subprocess
import time
from pathlib import Path

from install_lmstudio import NAME, VERSION, discover_devices, write_json
from shared_memory import ROOT
from thermal_unsloth import gpu_samples

FILES = (
    "desktop_service.py", "shared_memory.py", "context_unsloth.py", "thermal_unsloth.py",
    "lmstudio_bridge.mjs", "start-desktop-service.ps1", "import_desktop_history.py",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def install(root, lm_home, source, dry_run=False, update=False):
    node = Path.home() / "AppData/Local/Programs/LM Studio/resources/app/.webpack/bin/node.exe"
    sdk = lm_home / "extensions/plugins/lmstudio/rag-v1/node_modules/@lmstudio/sdk/dist/index.mjs"
    runtime = lm_home / "extensions/backends" / f"{NAME}-{VERSION}"
    for path in (node, sdk, runtime / "p100/llama-server.exe"):
        if not path.is_file():
            raise RuntimeError("Required installed runtime was not found: " + str(path))
    manifest = root / "installation.json"
    entries = {name: sha(source / name) for name in FILES}
    if manifest.exists():
        record = json.loads(manifest.read_text(encoding="utf-8"))
        if any(sha(root / "runtime" / name) != value for name, value in record["files"].items()):
            raise RuntimeError("Installed helper changed; stop the service and review before updating")
        if entries != record["files"]:
            if not update:
                raise RuntimeError("Use --update after stopping the service")
            service = root / "service.json"
            if service.exists():
                from thermal_unsloth import process_alive

                if process_alive(json.loads(service.read_text(encoding="utf-8"))["pid"]):
                    raise RuntimeError("Stop the managed service before updating")
            if dry_run:
                print("DESKTOP_SERVICE_UPDATE_DRY_RUN")
                return
            before = {name: (root / "runtime" / name).read_bytes() if (root / "runtime" / name).exists() else None for name in FILES}
            if any(data is not None and name not in record["files"] for name, data in before.items()):
                raise RuntimeError("An unmanaged helper exists")
            try:
                for name in FILES:
                    shutil.copyfile(source / name, root / "runtime" / name)
                write_json(manifest, {**record, "files": entries})
            except Exception:
                for name, data in before.items():
                    if data is None:
                        (root / "runtime" / name).unlink(missing_ok=True)
                    else:
                        (root / "runtime" / name).write_bytes(data)
                raise
            print("DESKTOP_SERVICE_UPDATED")
            return
        print("DESKTOP_SERVICE_ALREADY_INSTALLED")
        return
    if (root / "runtime").exists() or (root / "config.json").exists():
        raise RuntimeError("Unmanaged Desktop configuration exists")
    devices = discover_devices(runtime / "p100/llama-server.exe")
    samples = gpu_samples()
    primary = [sample for sample in samples.values() if sample.name.startswith("Tesla P100")]
    if len(primary) != 1:
        raise RuntimeError("Expected one P100")
    config = {
        "port": 8877, "share_memory": True, "primary_uuid": primary[0].uuid,
        "primary_devices": ",".join(devices), "fallback_device": devices[1],
        "return_temperature": 68, "stable_seconds": 120, "poll_seconds": 5, "ram_cache_mib": 8192,
        "context_limit": 131072, "fallback_context_limit": 65536, "unsloth_url": "http://127.0.0.1:8888",
        "lmstudio": {"node": str(node), "sdk": str(sdk), "lms": str(lm_home / "bin/lms.exe"),
                     "runtime": str(runtime), "identifier": "qwen/qwen3.5-9b",
                     "models": {"qwen/qwen3.5-9b": {"modelKey": "qwen/qwen3.5-9b"}}},
    }
    if dry_run:
        print("DESKTOP_SERVICE_DRY_RUN", json.dumps({"files": list(entries), "port": config["port"], "primary_devices": config["primary_devices"]}))
        return
    if (root / "api-key.txt").exists():
        raise RuntimeError("An unmanaged Desktop API key exists")
    root.mkdir(parents=True, exist_ok=True)
    directory = root / "runtime"
    directory.mkdir()
    try:
        for name in FILES:
            shutil.copyfile(source / name, directory / name)
        write_json(root / "config.json", config)
        (root / "api-key.txt").write_text(secrets.token_urlsafe(32) + "\n", encoding="utf-8", newline="\n")
        write_json(manifest, {"files": entries, "config_hash": sha(root / "config.json")})
    except Exception:
        for name in FILES:
            (directory / name).unlink(missing_ok=True)
        directory.rmdir()
        for name in ("config.json", "api-key.txt", "installation.json"):
            (root / name).unlink(missing_ok=True)
        raise
    print("DESKTOP_SERVICE_INSTALLED", root)


def install_plugin(root, lm_home, source, update=False):
    plugin = root / "plugin-source"
    state = root / "plugin-installation.json"
    if plugin.exists():
        if not update:
            raise RuntimeError("Plugin source already exists; use --update after review")
        if state.exists():
            record = json.loads(state.read_text(encoding="utf-8"))
            if sha(plugin / "src/index.ts") != record["source_hash"]:
                raise RuntimeError("Plugin source was manually changed")
        elif sha(plugin / "src/index.ts") != sha(source / "lmstudio_plugin.ts"):
            raise RuntimeError("Unmanaged plugin source exists")
    else:
        (plugin / "src").mkdir(parents=True)
        write_json(plugin / "manifest.json", {"type": "plugin", "runner": "node", "owner": "local", "name": "p100-desktop", "revision": 1})
        write_json(plugin / "package.json", {"name": "p100-desktop", "version": "1.0.0", "type": "module"})
    shutil.copyfile(source / "lmstudio_plugin.ts", plugin / "src/index.ts")
    modules = lm_home / "extensions/plugins/lmstudio/rag-v1/node_modules"
    link_path = str(plugin / "node_modules").replace("'", "''")
    target_path = str(modules).replace("'", "''")
    if not (plugin / "node_modules").exists():
        subprocess.run(["powershell.exe", "-NoProfile", "-Command", f"New-Item -ItemType Junction -Path '{link_path}' -Target '{target_path}' | Out-Null"], check=True, capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
    result = subprocess.run([str(lm_home / "bin/lms.exe"), "dev", "--install", "--yes"], cwd=plugin, capture_output=True, text=True, encoding="utf-8", timeout=120, check=False, creationflags=subprocess.CREATE_NO_WINDOW)
    if result.returncode:
        raise RuntimeError("LM Studio plugin installation failed: " + result.stdout[-1500:] + result.stderr[-1500:])
    write_json(state, {"source_hash": sha(plugin / "src/index.ts"), "identifier": "local/p100-desktop"})
    print(result.stdout[-1500:])


def install_chat(root, lm_home):
    state = root / "starter-chat.json"
    if state.exists():
        print("DESKTOP_STARTER_CHAT_EXISTS")
        return
    timestamp = int(time.time() * 1000)
    name = str(timestamp) + ".conversation.json"
    chat = {
        "name": "P100 shared memory", "pinned": True, "createdAt": timestamp, "preset": "", "tokenCount": 0,
        "userLastMessagedAt": timestamp, "assistantLastMessagedAt": timestamp, "systemPrompt": "", "messages": [],
        "usePerChatPredictionConfig": False, "perChatPredictionConfig": {"fields": []},
        "clientInput": "", "clientInputFiles": [], "userFilesSizeBytes": 0,
        "lastUsedModel": {"identifier": "qwen/qwen3.5-9b", "indexedModelIdentifier": "qwen/qwen3.5-9b",
                          "instanceLoadTimeConfig": {"fields": []}, "instanceOperationTimeConfig": {"fields": []}},
        "notes": [], "plugins": ["local/p100-desktop"], "pluginConfigs": {}, "disabledPluginTools": [], "looseFiles": [],
    }
    path = lm_home / "conversations" / name
    with path.open("x", encoding="utf-8", newline="\n") as output:
        output.write(json.dumps(chat, ensure_ascii=False, indent=2) + "\n")
    write_json(state, {"conversation": name, "initial_hash": sha(path)})
    print("DESKTOP_STARTER_CHAT_CREATED", name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--plugin", action="store_true")
    parser.add_argument("--update", action="store_true")
    args = parser.parse_args()
    home = Path.home() / ".lmstudio"
    source = Path(__file__).resolve().parent
    install(ROOT, home, source, args.dry_run, args.update)
    if args.plugin and not args.dry_run:
        install_plugin(ROOT, home, source, args.update)
        install_chat(ROOT, home)


if __name__ == "__main__":
    main()
