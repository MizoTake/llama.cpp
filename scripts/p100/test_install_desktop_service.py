import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from install_desktop_service import FILES, install, sha


class UpdateTests(unittest.TestCase):
    def test_helper_update_does_not_initialize_cuda_or_replace_the_model_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "desktop"
            source = Path(directory) / "source"
            source.mkdir()
            (root / "runtime").mkdir(parents=True)
            for name in FILES:
                (source / name).write_text("new helper\n", encoding="utf-8")
                (root / "runtime" / name).write_text("old helper\n", encoding="utf-8")
            record = {"files": {name: sha(root / "runtime" / name) for name in FILES}}
            (root / "installation.json").write_text(json.dumps(record), encoding="utf-8")
            config = b'{"context_limit":65536,"memory_backend":"memorix"}\n'
            (root / "config.json").write_bytes(config)
            with patch("install_desktop_service.Path.is_file", return_value=True), patch("install_desktop_service.discover_devices", side_effect=RuntimeError("GPU occupied")) as discover, patch("install_desktop_service.gpu_samples") as sample:
                install(root, Path(directory) / "lmstudio", source, update=True)
            discover.assert_not_called()
            sample.assert_not_called()
            self.assertEqual((root / "config.json").read_bytes(), config)
            self.assertEqual(json.loads((root / "installation.json").read_text(encoding="utf-8"))["files"], {name: sha(source / name) for name in FILES})
