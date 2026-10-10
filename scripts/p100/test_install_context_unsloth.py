import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from install_context_unsloth import (
    MARKER,
    digest,
    install,
    patched_source,
    restore,
    update_shared,
)

LLAMA = "\n".join(
    f"def fit_{index}(self, thread_id, cancel_event):\n"
    "    return _fit_with_instruction_pins(\n"
    "        [],\n"
    "        context_length = self.context_length,\n"
    "    )\n"
    for index in range(6)
)
ROUTES = "\n".join(
    f"async def {name}(payload, llama_backend, cancel_event):\n"
    "    return await _build_openai_passthrough_body_async(\n"
    "        payload, backend_ctx = llama_backend.context_length, llama_backend = llama_backend\n"
    "    )\n"
    for name in ("_openai_passthrough_stream_admitted", "_openai_passthrough_non_streaming_upstream")
) + "\nasync def produce_openai_chat_completions(payload):\n    return payload\n"


class InstallerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.backend = self.root / "backend"
        self.backup = self.root / "backup"
        self.inputs = {"core/inference/llama_cpp.py": LLAMA, "routes/inference.py": ROUTES}
        for name, text in self.inputs.items():
            path = self.backend / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(text.encode())
        self.addon = self.root / "addon.py"
        self.addon.write_text("value = 1\n", encoding="utf-8")
        mock = patch("install_context_unsloth.SUPPORTED", {name: digest(text.encode()) for name, text in self.inputs.items()})
        mock.start()
        self.addCleanup(mock.stop)

    def test_dry_run_does_not_modify_files(self):
        install(self.backend, self.backup, self.addon, dry_run=True)
        self.assertFalse(self.backup.exists())
        self.assertEqual((self.backend / "core/inference/llama_cpp.py").read_text(), LLAMA)

    def test_install_and_restore_preserve_original_bytes(self):
        install(self.backend, self.backup, self.addon)
        self.assertIn(MARKER, (self.backend / "core/inference/llama_cpp.py").read_text())
        restore(self.backend, self.backup)
        for name, text in self.inputs.items():
            self.assertEqual((self.backend / name).read_bytes(), text.encode())
        self.assertFalse((self.backend / "core/inference/p100_context.py").exists())

    def test_unknown_desktop_revision_is_rejected_before_writing(self):
        (self.backend / "routes/inference.py").write_text("changed\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "Unsupported"):
            install(self.backend, self.backup, self.addon)
        self.assertFalse(self.backup.exists())

    def test_restore_does_not_overwrite_desktop_update(self):
        install(self.backend, self.backup, self.addon)
        (self.backend / "routes/inference.py").write_text("updated\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "overwrite"):
            restore(self.backend, self.backup)
        self.assertEqual((self.backend / "routes/inference.py").read_text(), "updated\n")

    def test_second_install_is_idempotent(self):
        install(self.backend, self.backup, self.addon)
        before = (self.backend / "routes/inference.py").read_bytes()
        install(self.backend, self.backup, self.addon)
        self.assertEqual((self.backend / "routes/inference.py").read_bytes(), before)

    def test_changed_addon_requires_restoring_first(self):
        install(self.backend, self.backup, self.addon)
        self.addon.write_text("value = 2\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "Restore"):
            install(self.backend, self.backup, self.addon)

    def test_changed_call_structure_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "integration calls"):
            patched_source(LLAMA.replace("_fit_with_instruction_pins", "another_fit"), "llama")

    def test_existing_newline_format_is_preserved(self):
        output = patched_source(LLAMA.replace("\n", "\r\n"), "llama")
        self.assertNotIn("\n", output.replace("\r\n", ""))

    def test_shared_update_keeps_context_and_route_wrappers(self):
        shared = self.root / "shared.py"
        shared.write_bytes(b"value = 1\n")
        install(self.backend, self.backup, self.addon, shared=shared)
        route = (self.backend / "routes/inference.py").read_bytes()
        context = (self.backend / "core/inference/p100_context.py").read_bytes()
        shared.write_bytes(b"value = 2\n")
        update_shared(self.backend, self.backup, shared)
        self.assertEqual((self.backend / "core/inference/p100_shared_memory.py").read_bytes(), b"value = 2\n")
        self.assertEqual((self.backend / "routes/inference.py").read_bytes(), route)
        self.assertEqual((self.backend / "core/inference/p100_context.py").read_bytes(), context)
        restore(self.backend, self.backup)

    def test_shared_update_rejects_manually_changed_integration(self):
        shared = self.root / "shared.py"
        shared.write_bytes(b"value = 1\n")
        install(self.backend, self.backup, self.addon, shared=shared)
        (self.backend / "routes/inference.py").write_bytes(b"manual = True\n")
        shared.write_bytes(b"value = 2\n")
        with self.assertRaises(RuntimeError):
            update_shared(self.backend, self.backup, shared)
        self.assertEqual((self.backend / "core/inference/p100_shared_memory.py").read_bytes(), b"value = 1\n")


if __name__ == "__main__":
    unittest.main()
