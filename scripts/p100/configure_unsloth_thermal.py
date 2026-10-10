import argparse
import copy
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

from thermal_unsloth import (
    DEFAULT_DIRECTORY,
    StudioStore,
    gpu_samples,
    save_json,
    validate_config,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    store = StudioStore()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gguf-py"))
    from core.inference.model_ids import hf_cache_repo_id
    from gguf import GGUFReader
    from hub.utils.gguf import extract_quant_label

    root = Path.home() / ".llama.cpp/desktop"
    common = json.loads((root / "config.json").read_text(encoding="utf-8"))
    config_path = DEFAULT_DIRECTORY / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    rows = store.get()
    state = json.loads((DEFAULT_DIRECTORY / "state.json").read_text(encoding="utf-8"))
    if state.get("managed") or state.get("conflicts"):
        raise RuntimeError("Stop the monitor and resolve profile conflicts before configuration")
    samples = gpu_samples()
    primary = samples[config["primary_gpu_uuid"]]
    fallback = samples[config["fallback_gpu_uuid"]]
    profiles, files, writes = {}, {}, {}
    candidates = {Path(key) for key in rows if Path(key).is_file() and key.lower().endswith(".gguf")}
    candidates.update((Path.home() / ".cache/huggingface/hub").rglob("*.gguf"))
    for file in sorted(candidates):
        reader = GGUFReader(str(file))
        if "tokenizer.ggml.tokens" not in reader.fields:
            continue
        architecture = reader.fields["general.architecture"].contents()
        blocks = int(reader.fields[architecture + ".block_count"].contents())
        repo = hf_cache_repo_id(str(file))
        variant = extract_quant_label(file.name)
        aliases = [str(file)] + ([repo + ":" + variant] if repo else [])
        original = next((rows[key] for key in aliases if key in rows), None)
        if original is None or primary.index not in original.get("gpu_ids", []):
            continue
        projector_bytes = max((projector.stat().st_size for projector in file.parent.glob("mmproj*.gguf")), default=0)
        ratio = min(1, max(0.1, math.floor(fallback.memory_total_mib * 1024 * 1024 * 0.55 / (file.stat().st_size + projector_bytes) * 10) / 10))
        layers = 1024 if ratio == 1 else math.ceil(blocks * ratio)
        limits = {"context_limit": original.get("custom_context_length") or original["max_seq_length"],
            "batch_limit": min(original.get("n_batch", 512), 512 if ratio < 1 else 2048), "fallback_gpu_layers": layers}
        for alias in aliases:
            profiles[alias] = copy.deepcopy(limits)
            if alias not in rows:
                writes[alias] = copy.deepcopy(original)
        if repo in common.get("unsloth_models", []):
            if repo in files:
                raise RuntimeError("More than one configured variant exists for " + repo)
            files[repo] = {"model_path": repo, "gguf_variant": variant, "model_file": str(file)}
            profiles[repo] = copy.deepcopy(limits)
            writes[repo] = copy.deepcopy(original)
    if set(files) != set(common.get("unsloth_models", [])):
        raise RuntimeError("Every configured Unsloth model needs one verified downloaded variant")
    config.update(profiles=profiles, fallback_temperature=78, shared_status=str(root / "status.json"))
    validate_config(config)
    common.update(unsloth_thermal_config=str(config_path), fallback_temperature=config["fallback_temperature"], unsloth_loads=files)
    report = {"profiles": profiles, "loads": files, "fallback_temperature": config["fallback_temperature"],
        "return_temperature": config["return_temperature"], "stable_seconds": config["stable_seconds"]}
    if args.dry_run:
        print(json.dumps(report))
        return
    backup = DEFAULT_DIRECTORY / "backups" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    save_json(backup / "config.json", json.loads(config_path.read_text(encoding="utf-8")))
    save_json(backup / "common.json", json.loads((root / "config.json").read_text(encoding="utf-8")))
    save_json(backup / "overrides.json", rows)
    for key, original in writes.items():
        store.set(key, original)
    save_json(config_path, config)
    save_json(root / "config.json", common)
    print(json.dumps({**report, "backup": str(backup)}))


if __name__ == "__main__":
    main()
