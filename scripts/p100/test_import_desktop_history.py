import json
import tempfile
import unittest
from pathlib import Path

from import_desktop_history import import_lmstudio
from shared_memory import MemoryStore


class ImportTests(unittest.TestCase):
    def test_folder_conversations_keep_distinct_identity_and_selected_version(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = MemoryStore(root / "memory.sqlite3")
            for folder, text in (("one", "current"), ("two", "different")):
                path = root / "conversations" / folder / "1.conversation.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps({"messages": [{"currentlySelected": 1, "versions": [
                    {"type": "singleStep", "role": "user", "content": [{"type": "text", "text": "old"}]},
                    {"type": "singleStep", "role": "user", "content": [{"type": "text", "text": text}]},
                ]}]}), encoding="utf-8")
            self.assertEqual(import_lmstudio(root, store), 2)
            rows = store.list()
            self.assertEqual({row["conversation"] for row in rows}, {"one/1.conversation.json", "two/1.conversation.json"})
            self.assertEqual({store.get(row["id"])["messages"][0]["content"] for row in rows}, {"current", "different"})


if __name__ == "__main__":
    unittest.main()
