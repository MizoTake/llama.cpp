import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from context_unsloth import archive_passthrough_response
from shared_memory import CompletionCapture, MemoryStore, conversation_key


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.store = MemoryStore(self.root / "memory.sqlite3")
        config = self.root / ".llama.cpp/desktop/config.json"
        config.parent.mkdir(parents=True)
        config.write_text('{"share_memory":true}', encoding="utf-8")
        self.owner = "owner"
        account = ModuleType("utils.account_context")
        setattr(account, "current_account_id", lambda: self.owner)
        shared = ModuleType("core.inference.p100_shared_memory")
        setattr(shared, "CompletionCapture", CompletionCapture)
        setattr(shared, "MemoryStore", lambda: self.store)
        setattr(shared, "conversation_key", conversation_key)
        self.modules = patch.dict(sys.modules, {"utils.account_context": account, "core.inference.p100_shared_memory": shared})
        self.home = patch("context_unsloth.Path.home", return_value=self.root)
        self.modules.start()
        self.home.start()
        self.addCleanup(self.modules.stop)
        self.addCleanup(self.home.stop)
        self.payload = SimpleNamespace(messages=[{"role": "user", "content": "current"}], thread_id="one")

    def test_completed_reply_is_saved_immediately_and_response_is_unchanged(self):
        value = {"choices": [{"message": {"role": "assistant", "content": "42"}, "finish_reason": "stop"}]}
        response = SimpleNamespace(status_code=200, body=json.dumps(value).encode())
        result = asyncio.run(archive_passthrough_response(response, self.payload))
        self.assertIs(result, response)
        row = self.store.get(self.store.list()[0]["id"])
        self.assertEqual(row["messages"][-1], value["choices"][0]["message"])

    def test_cancelled_stream_closes_upstream_and_does_not_archive_an_answer(self):
        closed = []

        async def original():
            try:
                yield b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
                raise asyncio.CancelledError
            finally:
                closed.append(True)

        async def consume():
            response = SimpleNamespace(body_iterator=original())
            await archive_passthrough_response(response, self.payload)
            async for _chunk in response.body_iterator:
                pass

        with self.assertRaises(asyncio.CancelledError):
            asyncio.run(consume())
        self.assertEqual(closed, [True])
        self.assertEqual(self.store.list(), [])

    def test_non_owner_reply_never_enters_shared_memory(self):
        self.owner = "another-account"
        response = {"choices": [{"message": {"role": "assistant", "content": "private"}, "finish_reason": "stop"}]}
        self.assertIs(asyncio.run(archive_passthrough_response(response, self.payload)), response)
        self.assertEqual(self.store.list(), [])


if __name__ == "__main__":
    unittest.main()
