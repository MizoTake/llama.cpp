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

from gguf import GGUFReader
from storage.studio_db import get_app_setting
from utils.account_context import OWNER, run_as
from utils.llama_cpp_path_settings import set_custom_llama_cpp_path
from utils.openai_auto_switch_settings import MODEL_OVERRIDES_SETTING_KEY, get_model_overrides, resolve_model_override_key, set_model_override

backup_keys = ("custom_llama_cpp_path", MODEL_OVERRIDES_SETTING_KEY)
gpu_rows = subprocess.check_output([
    "nvidia-smi", "--query-gpu=index,name,compute_cap", "--format=csv,noheader,nounits",
], text=True)
gpus = list(csv.DictReader(gpu_rows.splitlines(), fieldnames=("index", "name", "cc"), skipinitialspace=True))
p100 = next((gpu for gpu in gpus if gpu["name"].startswith("Tesla P100")), None)
if p100 is None:
    raise RuntimeError("No Tesla P100 was found.")
gpu_ids = sorted(int(gpu["index"]) for gpu in gpus if gpu["cc"] in ("6.0", "8.9"))
ratio = ",".join("3" if index == int(p100["index"]) else "2" for index in gpu_ids)
previous = {key: run_as(OWNER, get_app_setting, key, None) for key in backup_keys}
backup_dir = Path.home() / ".unsloth" / "studio" / "backups" / "llama-p100"
backup_path = backup_dir / (datetime.now().strftime("%Y%m%d-%H%M%S-%f") + ".json")
if not args.dry_run:
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path.write_text(json.dumps(previous, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    run_as(OWNER, set_custom_llama_cpp_path, str(root))
model_configs = get_model_overrides()
for model_root in (Path.home() / ".lmstudio" / "models", Path.home() / ".cache" / "huggingface" / "hub"):
    for model_path in model_root.rglob("*.gguf"):
        try:
            reader = GGUFReader(str(model_path))
            has_tokenizer = "tokenizer.ggml.tokens" in reader.fields
            del reader
        except (OSError, ValueError):
            continue
        if has_tokenizer and resolve_model_override_key(str(model_path)) is None:
            model_configs[str(model_path)] = {}
for model_id, previous_config in model_configs.items():
    if "gguf" not in model_id.casefold():
        continue
    config = dict(previous_config)
    extras = list(config.pop("llama_extra_args", []) or [])
    controlled = {"--split-mode", "-sm", "--tensor-split", "-ts", "--flash-attn", "-fa"}
    cleaned = []
    index = 0
    while index < len(extras):
        flag = extras[index].split("=", 1)[0]
        if flag in controlled:
            index += 1 if "=" in extras[index] else 2
        else:
            cleaned.append(extras[index])
            index += 1
    config.update(gpu_ids=gpu_ids, gpu_index_kind="physical", gpu_memory_mode="auto", tensor_parallel=False)
    max_seq_length = config.pop("max_seq_length", None)
    extras = cleaned + ["--split-mode", "layer", "--tensor-split", ratio, "--flash-attn", "auto"]
    if not args.dry_run:
        run_as(OWNER, set_model_override, model_id, extras, max_seq_length, **config)
    print("Planned model:" if args.dry_run else "Configured model:", model_id)
print("llama.cpp:", root)
print("GPU IDs:", gpu_ids, "tensor split:", ratio)
print("Dry run; no settings saved." if args.dry_run else "Previous settings: " + str(backup_path))
