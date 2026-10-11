import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--runtime", type=Path)
parser.add_argument("--dry-run", action="store_true")
parser.add_argument("--model", action="append", help="Configure only this saved model ID; may be repeated.")
parser.add_argument("--batch-size", type=int)
parser.add_argument("--ubatch-size", type=int)
parser.add_argument("--context-length", type=int)
parser.add_argument("--cache-ram", type=int, help="Host prompt cache limit in MiB; 0 disables the cache.")
parser.add_argument("--ctx-checkpoints", type=int)
args = parser.parse_args()
script_dir = Path(__file__).resolve().parent
default_runtime = script_dir if (script_dir / "llama-server.exe").is_file() else script_dir.parents[1] / "release" / "p100-first"
root = (args.runtime or default_runtime).resolve()
if not (root / "llama-server.exe").is_file() or not (root / "gguf-py").is_dir():
    raise RuntimeError("A packaged runtime with llama-server.exe and gguf-py is required.")
backend_dir = Path(sys.prefix) / "Lib" / "site-packages" / "studio" / "backend"
if not backend_dir.is_dir():
    raise RuntimeError("Run this script with the Unsloth Desktop Python environment.")
sys.path.insert(0, str(backend_dir))
sys.path.insert(0, str(root / "gguf-py"))

from storage.studio_db import get_app_setting
from utils.account_context import OWNER, run_as
from utils.llama_cpp_path_settings import set_custom_llama_cpp_path
from utils.openai_auto_switch_settings import BATCH_SIZE_MIN, BATCH_SIZE_MAX, MAX_SEQ_LENGTH_CEILING, MODEL_OVERRIDES_SETTING_KEY, get_model_overrides, resolve_model_override_key, set_model_override

for value in (args.batch_size, args.ubatch_size):
    if value is not None and not BATCH_SIZE_MIN <= value <= BATCH_SIZE_MAX:
        raise ValueError(f"Batch sizes must be between {BATCH_SIZE_MIN} and {BATCH_SIZE_MAX}.")
if args.context_length is not None and not 1 <= args.context_length <= MAX_SEQ_LENGTH_CEILING:
    raise ValueError(f"Context length must be between 1 and {MAX_SEQ_LENGTH_CEILING}.")
if args.cache_ram is not None and not 0 <= args.cache_ram <= 65536:
    raise ValueError("RAM cache must be between 0 and 65536 MiB.")
if args.ctx_checkpoints is not None and not 0 <= args.ctx_checkpoints <= 256:
    raise ValueError("Context checkpoints must be between 0 and 256.")

backup_keys = ("custom_llama_cpp_path", MODEL_OVERRIDES_SETTING_KEY)
gpu_rows = subprocess.check_output([
    "nvidia-smi", "--query-gpu=index,name,compute_cap", "--format=csv,noheader,nounits",
], text=True)
gpus = list(csv.DictReader(gpu_rows.splitlines(), fieldnames=("index", "name", "cc"), skipinitialspace=True))
p100 = next((gpu for gpu in gpus if gpu["name"].startswith("Tesla P100")), None)
if p100 is None:
    raise RuntimeError("No Tesla P100 was found.")
gpu_ids = [int(p100["index"])] + sorted(int(gpu["index"]) for gpu in gpus if gpu["cc"] in ("6.0", "8.9") and gpu is not p100)
ratio = ",".join("3" if index == int(p100["index"]) else "2" for index in gpu_ids)
previous = {key: run_as(OWNER, get_app_setting, key, None) for key in backup_keys}
backup_dir = Path.home() / ".unsloth" / "studio" / "backups" / "llama-p100"
backup_path = backup_dir / (datetime.now().strftime("%Y%m%d-%H%M%S-%f") + ".json")
model_configs = get_model_overrides()
targets = set(args.model or [])
if not targets or not targets.issubset(model_configs):
    from gguf import GGUFReader
    from core.inference.model_ids import hf_cache_repo_id
    from hub.utils.gguf import extract_quant_label

    for model_root in (Path.home() / ".lmstudio" / "models", Path.home() / ".cache" / "huggingface" / "hub"):
        for model_path in model_root.rglob("*.gguf"):
            try:
                reader = GGUFReader(str(model_path))
                has_tokenizer = "tokenizer.ggml.tokens" in reader.fields
                del reader
            except (OSError, ValueError):
                continue
            if not has_tokenizer:
                continue
            repo_id = hf_cache_repo_id(str(model_path))
            model_id = repo_id + ":" + extract_quant_label(model_path.name) if repo_id else str(model_path)
            if resolve_model_override_key(model_id) is None:
                model_configs[model_id] = {}
unknown = targets - {model_id for model_id in model_configs if "gguf" in model_id.casefold()}
if unknown:
    raise ValueError("Unknown GGUF model: " + ", ".join(sorted(unknown)))
planned = []
for model_id, previous_config in model_configs.items():
    if "gguf" not in model_id.casefold() or (targets and model_id not in targets):
        continue
    config = dict(previous_config)
    extras = list(config.pop("llama_extra_args", []) or [])
    controlled = {"--split-mode", "-sm", "--tensor-split", "-ts", "--flash-attn", "-fa", "--cache-type-k", "-ctk", "--cache-type-v", "-ctv", "--batch-size", "-b", "--ubatch-size", "-ub"}
    if args.context_length is not None:
        controlled.update({"--ctx-size", "-c"})
    cache_flags = {"--cache-idle-slots", "--no-cache-idle-slots"}
    if args.cache_ram is not None:
        controlled.update({"--cache-ram", "-cram", *cache_flags})
        config["cache_ram"] = args.cache_ram
    if args.ctx_checkpoints is not None:
        controlled.update({"--ctx-checkpoints", "--swa-checkpoints", "-ctxcp"})
        config["ctx_checkpoints"] = args.ctx_checkpoints
    cleaned = []
    index = 0
    while index < len(extras):
        flag = extras[index].split("=", 1)[0]
        if flag in controlled:
            index += 1 if "=" in extras[index] or flag in cache_flags else 2
        else:
            cleaned.append(extras[index])
            index += 1
    config.update(gpu_ids=gpu_ids, gpu_index_kind="physical", gpu_memory_mode="manual", gpu_layers=1024, tensor_parallel=False, kv_cache_dtype="f16")
    config["n_batch"] = args.batch_size if args.batch_size is not None else config.get("n_batch", 512)
    config["n_ubatch"] = args.ubatch_size if args.ubatch_size is not None else config.get("n_ubatch", 512)
    if config["n_ubatch"] > config["n_batch"]:
        raise ValueError("ubatch must not exceed batch for " + model_id)
    config.setdefault("n_parallel", 1)
    max_seq_length = config.pop("max_seq_length", 8192)
    if args.context_length is not None:
        max_seq_length = args.context_length
        config["custom_context_length"] = args.context_length
    extras = cleaned + ["--split-mode", "layer", "--tensor-split", ratio, "--flash-attn", "on"]
    if args.cache_ram is not None and args.cache_ram > 0:
        extras.append("--cache-idle-slots")
    planned.append((model_id, extras, max_seq_length, config))
if not args.dry_run:
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path.write_text(json.dumps(previous, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    run_as(OWNER, set_custom_llama_cpp_path, str(root))
for model_id, extras, max_seq_length, config in planned:
    if not args.dry_run:
        run_as(OWNER, set_model_override, model_id, extras, max_seq_length, **config)
    print("Planned model:" if args.dry_run else "Configured model:", model_id, f"batch/ubatch: {config['n_batch']}/{config['n_ubatch']}, context: {max_seq_length}")
print("llama.cpp:", root)
print("GPU IDs:", gpu_ids, "tensor split:", ratio, "Flash Attention: on, KV: f16")
print("Dry run; no settings saved." if args.dry_run else "Previous settings: " + str(backup_path))
