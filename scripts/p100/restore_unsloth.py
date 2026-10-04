import argparse
import json
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("backup", type=Path)
args = parser.parse_args()
backend_dir = Path(sys.prefix) / "Lib" / "site-packages" / "studio" / "backend"
if not backend_dir.is_dir():
    raise RuntimeError("Run this script with the Unsloth Desktop Python environment.")
sys.path.insert(0, str(backend_dir))
from storage.studio_db import upsert_app_settings
from utils.account_context import OWNER, run_as

settings = json.loads(args.backup.read_text(encoding="utf-8"))
allowed = {"custom_llama_cpp_path", "openai_api_auto_switch_overrides"}
if set(settings) != allowed:
    raise ValueError("This is not a llama.cpp P100 settings backup.")
run_as(OWNER, upsert_app_settings, settings)
print("Restored:", args.backup)
