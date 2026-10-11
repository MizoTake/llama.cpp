import contextlib
import io
import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch


class ConfigureUnslothTests(unittest.TestCase):
    def test_explicit_ram_cache_keeps_checkpoints_and_replaces_conflicting_extras(self):
        writes, _ = self.run_configure({"org/model-GGUF:Q4_K_M": {"llama_extra_args": ["--cache-ram=0", "--ctx-checkpoints", "0", "--no-cache-idle-slots", "--threads", "6"]}}, ["--cache-ram", "8192", "--ctx-checkpoints", "32"])
        config = writes.call_args.kwargs
        extras = writes.call_args.args[1]
        self.assertEqual(config["cache_ram"], 8192)
        self.assertEqual(config["ctx_checkpoints"], 32)
        self.assertNotIn("--cache-ram=0", extras)
        self.assertNotIn("--ctx-checkpoints", extras)
        self.assertNotIn("--no-cache-idle-slots", extras)
        self.assertIn("--cache-idle-slots", extras)
        self.assertIn("--threads", extras)

    def run_configure(self, configs, extra_args=()):
        writes = Mock()
        runtime_write = Mock()
        self.writes = writes
        self.runtime_write = runtime_write
        modules = {}
        exports = {
            "gguf": {"GGUFReader": Mock()},
            "core.inference.model_ids": {"hf_cache_repo_id": Mock()},
            "hub.utils.gguf": {"extract_quant_label": Mock()},
            "storage.studio_db": {"get_app_setting": Mock(return_value={})},
            "utils.account_context": {"OWNER": "owner", "run_as": lambda owner, fn, *a, **kw: fn(*a, **kw)},
            "utils.llama_cpp_path_settings": {"set_custom_llama_cpp_path": runtime_write},
            "utils.openai_auto_switch_settings": {
                "MODEL_OVERRIDES_SETTING_KEY": "overrides",
                "get_model_overrides": lambda: configs,
                "resolve_model_override_key": Mock(),
                "set_model_override": writes,
                "BATCH_SIZE_MIN": 1,
                "BATCH_SIZE_MAX": 65536,
                "MAX_SEQ_LENGTH_CEILING": 1048576,
            },
        }
        for name, values in exports.items():
            module = ModuleType(name)
            module.__dict__.update(values)
            modules[name] = module
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "llama-server.exe").touch()
            (root / "gguf-py").mkdir()
            (root / "Lib/site-packages/studio/backend").mkdir(parents=True)
            argv = ["configure_unsloth.py", "--runtime", str(root), *extra_args]
            with (
                patch.dict(sys.modules, modules),
                patch.object(sys, "argv", argv),
                patch.object(sys, "prefix", str(root)),
                patch.object(sys, "path", list(sys.path)),
                patch.object(Path, "home", return_value=root),
                patch.object(Path, "rglob", return_value=[]) as scan,
                patch("subprocess.check_output", return_value="0, NVIDIA GeForce RTX 4070, 8.9\n1, Tesla P100-PCIE-16GB, 6.0\n"),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.scan = scan
                runpy.run_path(str(Path(__file__).with_name("configure_unsloth.py")), run_name="__main__")
        return writes, runtime_write

    def test_existing_batch_and_context_survive_reconfiguration(self):
        writes, _ = self.run_configure({"org/model-GGUF:Q4_K_M": {"n_batch": 2048, "n_ubatch": 1024, "max_seq_length": 65536}})
        call = writes.call_args
        self.assertEqual(call.args[2], 65536)
        self.assertEqual(call.kwargs["n_batch"], 2048)
        self.assertEqual(call.kwargs["n_ubatch"], 1024)
        self.assertEqual(call.kwargs["gpu_ids"], [1, 0])

    def test_only_selected_model_receives_explicit_tuning(self):
        writes, _ = self.run_configure({"org/a-GGUF:Q4_K_M": {}, "org/b-GGUF:Q4_K_M": {}}, ["--model", "org/a-GGUF:Q4_K_M", "--batch-size", "2048", "--ubatch-size", "1024", "--context-length", "65536"])
        self.assertEqual(writes.call_count, 1)
        self.assertEqual(writes.call_args.args[0], "org/a-GGUF:Q4_K_M")
        self.assertEqual(writes.call_args.args[2], 65536)
        self.assertEqual(writes.call_args.kwargs["n_batch"], 2048)
        self.assertEqual(writes.call_args.kwargs["n_ubatch"], 1024)
        self.scan.assert_not_called()

    def test_dry_run_does_not_write_settings(self):
        writes, runtime_write = self.run_configure({"org/a-GGUF:Q4_K_M": {}}, ["--dry-run"])
        writes.assert_not_called()
        runtime_write.assert_not_called()

    def test_missing_target_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown GGUF model"):
            self.run_configure({"org/a-GGUF:Q4_K_M": {}}, ["--model", "org/missing-GGUF:Q4_K_M"])
        self.writes.assert_not_called()
        self.runtime_write.assert_not_called()

    def test_invalid_batch_relationship_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "ubatch"):
            self.run_configure({"org/a-GGUF:Q4_K_M": {}}, ["--batch-size", "512", "--ubatch-size", "1024"])
        self.writes.assert_not_called()
        self.runtime_write.assert_not_called()

    def test_out_of_range_context_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Context length"):
            self.run_configure({"org/a-GGUF:Q4_K_M": {}}, ["--context-length", "1048577"])
        self.writes.assert_not_called()
        self.runtime_write.assert_not_called()


if __name__ == "__main__":
    unittest.main()
