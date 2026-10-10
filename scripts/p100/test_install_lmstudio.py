import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from install_lmstudio import (
    FILES,
    NAME,
    VERSION,
    check_idle,
    install,
    restore,
    tuned_config,
    write_json,
)


class LmStudioInstallerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.home = self.root / "lmstudio"
        self.base = self.root / "base"
        self.package = self.root / "package"
        self.base.mkdir()
        self.package.mkdir()
        manifest = {
            "name": "stock", "version": "2.47.0", "gpu": {"targets": ["8.9"]},
            "engine_protocol_server": {"runtime_kind": "llama-server", "executable_relative_path": "llama-server.exe"},
        }
        (self.base / "backend-manifest.json").write_text(json.dumps(manifest))
        (self.base / "display-data.json").write_text(json.dumps([["en", {"displayName": "Stock"}]]))
        (self.base / "bridge.node").write_bytes(b"stock bridge")
        sums = []
        for name in FILES:
            value = name.encode()
            (self.package / name).write_bytes(value)
            sums.append(hashlib.sha256(value).hexdigest() + "  " + name)
        (self.package / "SHA256SUMS.txt").write_text("\n".join(sums))
        self.addCleanup(patch.stopall)
        patch("install_lmstudio.check_idle").start()
        patch("install_lmstudio.select_runtime").start()

    def test_bad_package_checksum_is_rejected_before_writing(self):
        (self.package / "llama-server.exe").write_bytes(b"changed")
        with self.assertRaisesRegex(RuntimeError, "checksum"):
            install(self.home, self.base, self.package, ["CUDA1", "CUDA0"])
        self.assertFalse(self.home.exists())

    def test_runtime_keeps_stock_bridge_and_uses_separate_server(self):
        record = install(self.home, self.base, self.package, ["CUDA1", "CUDA0"])
        target = Path(record["target"])
        manifest = json.loads((target / "backend-manifest.json").read_text())
        self.assertEqual(manifest["name"], NAME)
        self.assertEqual(manifest["version"], VERSION)
        self.assertEqual(manifest["engine_protocol_server"]["executable_relative_path"], "p100/llama-server.exe")
        self.assertEqual((target / "bridge.node").read_bytes(), b"stock bridge")
        self.assertEqual((target / "p100/llama-server.exe").read_bytes(), b"llama-server.exe")
        self.assertEqual(json.loads((self.base / "backend-manifest.json").read_text())["name"], "stock")

    def test_defaults_keep_other_fields_and_preserve_device_order(self):
        original = {"preset": "mine", "load": {"fields": [{"key": "unrelated", "value": 7}]}}
        result = tuned_config(original, ["CUDA1", "CUDA0"])
        fields = {item["key"]: item["value"] for item in result["load"]["fields"]}
        self.assertEqual(fields["unrelated"], 7)
        override = fields["llm.load.llama.argumentsOverride"]
        parameters = {item["key"]: item["value"] for item in override["overrideParameters"]}
        self.assertEqual(parameters["--device"], "CUDA1,CUDA0")
        self.assertEqual(parameters["--tensor-split"], "3,2")
        self.assertEqual(parameters["--fit"], "off")
        self.assertEqual(result["preset"], "mine")
        self.assertEqual(len(original["load"]["fields"]), 1)

    def test_dry_run_creates_no_files(self):
        install(self.home, self.base, self.package, ["CUDA1", "CUDA0"], dry_run=True)
        self.assertFalse(self.home.exists())

    def test_second_install_is_idempotent(self):
        first = install(self.home, self.base, self.package, ["CUDA1", "CUDA0"])
        second = install(self.home, self.base, self.package, ["CUDA1", "CUDA0"])
        self.assertEqual(first, second)

    def test_restore_preserves_original_model_file_bytes(self):
        model = self.home / ".internal/user-concrete-model-default-config/qwen/qwen3.5-9b.json"
        model.parent.mkdir(parents=True)
        original = b'{"preset":"mine","load":{"fields":[]}}\r\n'
        model.write_bytes(original)
        record = install(self.home, self.base, self.package, ["CUDA1", "CUDA0"])
        restore(self.home)
        self.assertEqual(model.read_bytes(), original)
        self.assertTrue(Path(record["target"]).is_dir())

    def test_restore_refuses_to_overwrite_manual_model_change(self):
        record = install(self.home, self.base, self.package, ["CUDA1", "CUDA0"])
        model = Path(record["model_file"])
        model.write_text('{"manual":true}')
        with self.assertRaisesRegex(RuntimeError, "changed"):
            restore(self.home)
        self.assertEqual(model.read_text(), '{"manual":true}')

    def test_restore_keeps_unrelated_hardware_edits(self):
        install(self.home, self.base, self.package, ["CUDA1", "CUDA0"])
        path = self.home / ".internal/hardware-config.json"
        data = json.loads(path.read_text())
        data["json"].append(["other", {"fields": [{"key": "mine", "value": True}]}])
        path.write_text(json.dumps(data))
        restore(self.home)
        data = json.loads(path.read_text())
        self.assertEqual(data["json"], [["other", {"fields": [{"key": "mine", "value": True}]}]])

    def test_failed_settings_write_rolls_back_model_and_keeps_runtime_for_inspection(self):
        def fail_hardware(path, value):
            if path.name == "hardware-config.json":
                raise OSError("write failed")
            return write_json(path, value)

        with patch("install_lmstudio.write_json", side_effect=fail_hardware), self.assertRaisesRegex(OSError, "write failed"):
            install(self.home, self.base, self.package, ["CUDA1", "CUDA0"])
        self.assertFalse((self.home / ".internal/user-concrete-model-default-config/qwen/qwen3.5-9b.json").exists())
        self.assertFalse((self.home / "p100-custom/installation.json").exists())
        self.assertEqual(len(list((self.home / "p100-custom").glob("failed-*"))), 1)

    def test_restore_refuses_a_different_version_selected_by_user(self):
        install(self.home, self.base, self.package, ["CUDA1", "CUDA0"])
        write_json(self.home / ".internal/backend-preferences-v1.json", [{"model_format": "gguf", "name": NAME, "version": "2.47.101"}])
        with self.assertRaisesRegex(RuntimeError, "selection changed"):
            restore(self.home)

    def test_loaded_models_prevent_configuration_changes(self):
        with patch("install_lmstudio.cli", return_value='[{"identifier":"existing"}]'), self.assertRaisesRegex(RuntimeError, "Unload"):
            check_idle(self.home)

    def test_reinstall_refuses_manual_gpu_changes(self):
        install(self.home, self.base, self.package, ["CUDA1", "CUDA0"])
        write_json(self.home / ".internal/hardware-config.json", {"json": [[NAME, {"fields": []}]]})
        with self.assertRaisesRegex(RuntimeError, "GPU settings changed"):
            install(self.home, self.base, self.package, ["CUDA1", "CUDA0"])

    def test_reinstall_rejects_a_different_build(self):
        install(self.home, self.base, self.package, ["CUDA1", "CUDA0"])
        (self.package / "llama-server.exe").write_bytes(b"new build")
        with self.assertRaisesRegex(RuntimeError, "Different package"):
            install(self.home, self.base, self.package, ["CUDA1", "CUDA0"])


if __name__ == "__main__":
    unittest.main()
