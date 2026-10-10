import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

from context_unsloth import compact_context, model_scope


def count(messages):
    return sum(len(json.dumps(message).split()) + 4 for message in messages)


def history():
    messages = [{"role": "system", "content": "Keep original system rules."}]
    for index in range(12):
        messages.extend([
            {"role": "user", "content": f"turn {index} " + "detail " * 100},
            {"role": "assistant", "content": f"result {index}"},
        ])
    messages.append({"role": "user", "content": "What remains?"})
    return messages


class ContextSummaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.calls = []
        self.config = {"trigger_tokens": 900, "recent_turns": 4, "summary_tokens": 128, "safety_tokens": 16}

    def summarize(self, messages, max_tokens):
        self.calls.append(copy.deepcopy(messages))
        return "Goal: finish task. Constraints: preserve quality. Pending: verification."

    def fit(self, messages, **kwargs):
        return compact_context(
            messages, context_length=2048, max_tokens=128, count_tokens=count,
            summarize=self.summarize, config=self.config, state_dir=self.root,
            scope="account-model", **kwargs,
        )

    def test_short_conversation_does_not_generate_or_write(self):
        messages = history()[:3]
        self.assertIsNone(self.fit(messages))
        self.assertEqual(self.calls, [])
        self.assertEqual(list(self.root.iterdir()), [])

    def test_summary_keeps_system_and_recent_four_user_turns_verbatim(self):
        messages = history()
        before = copy.deepcopy(messages)
        fitted, event = self.fit(messages)
        self.assertEqual(messages, before)
        self.assertIs(fitted[0], messages[0])
        self.assertEqual(fitted[-7:], messages[-7:])
        self.assertEqual(fitted[1]["role"], "assistant")
        self.assertIn("Pending: verification", fitted[1]["content"])
        self.assertTrue(event["fits"])
        self.assertLessEqual(count(fitted), event["prompt_target"])
        archived = json.loads(next(self.root.glob("archive-*.json")).read_text())
        self.assertEqual(archived["messages"], messages[1:-7])

    def test_full_history_replay_reuses_summary_without_another_generation(self):
        messages = history()
        first, _ = self.fit(messages)
        calls = len(self.calls)
        second, event = self.fit(copy.deepcopy(messages))
        self.assertEqual(first, second)
        self.assertEqual(len(self.calls), calls)
        self.assertTrue(event["summary_reused"])

    def test_branch_change_does_not_reuse_old_summary(self):
        messages = history()
        self.fit(messages)
        calls = len(self.calls)
        messages[1] = {"role": "user", "content": "Different branch " + "detail " * 100}
        self.fit(messages)
        self.assertGreater(len(self.calls), calls)

    def test_account_scope_does_not_share_a_summary(self):
        messages = history()
        self.fit(messages)
        calls = len(self.calls)
        compact_context(
            messages, context_length=2048, max_tokens=128, count_tokens=count,
            summarize=self.summarize, config=self.config, state_dir=self.root,
            scope="another-account-model",
        )
        self.assertGreater(len(self.calls), calls)

    def test_tool_exchange_is_kept_as_a_complete_turn(self):
        messages = history()
        messages[-3:-3] = [
            {"role": "assistant", "content": "", "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "read", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "call-1", "content": "file content"},
        ]
        fitted, _ = self.fit(messages)
        self.assertIn(messages[-5], fitted)
        self.assertIn(messages[-4], fitted)

    def test_latest_oversize_message_is_not_cut_or_summarized(self):
        messages = history() + [{"role": "user", "content": "large " * 2500}]
        self.assertIsNone(self.fit(messages))
        self.assertEqual(self.calls, [])

    def test_summary_failure_preserves_original_request(self):
        messages = history()
        before = copy.deepcopy(messages)
        with patch.object(self, "summarize", side_effect=RuntimeError("offline")):
            self.assertIsNone(self.fit(messages))
        self.assertEqual(messages, before)
        self.assertFalse(list(self.root.glob("state-*.json")))

    def test_oversize_summary_is_not_committed(self):
        with patch.object(self, "summarize", return_value="too long " * 2000):
            self.assertIsNone(self.fit(history()))
        self.assertFalse(list(self.root.glob("state-*.json")))

    def test_protected_old_message_is_not_removed(self):
        messages = history()
        self.assertIsNone(self.fit(messages, protected_message_ids={id(messages[1])}))

    def test_cache_cannot_remove_a_newly_protected_message(self):
        messages = history()
        self.fit(messages)
        self.assertIsNone(self.fit(messages, protected_message_ids={id(messages[1])}))

    def test_extended_history_merges_previous_summary_with_new_turns(self):
        messages = history()
        self.fit(messages)
        calls = len(self.calls)
        for index in range(12):
            messages.extend([
                {"role": "assistant", "content": "continued"},
                {"role": "user", "content": f"new {index} " + "detail " * 100},
            ])
        fitted, event = self.fit(messages)
        self.assertGreater(len(self.calls), calls)
        self.assertIn("Pending: verification", self.calls[calls][1]["content"])
        self.assertLessEqual(count(fitted), event["prompt_target"])

    def test_corrupt_state_is_rebuilt_from_original_history(self):
        messages = history()
        self.fit(messages)
        next(self.root.glob("state-*.json")).write_text("[]", encoding="utf-8")
        self.assertTrue(self.fit(messages)[1]["fits"])

    def test_summary_input_is_chunked_under_model_budget(self):
        messages = history()
        messages[1] = {"role": "user", "content": "old log " * 3000}
        fitted, _ = self.fit(messages)
        self.assertGreater(len(self.calls), 1)
        self.assertTrue(all(count(prompt) <= 2048 - 128 - 16 for prompt in self.calls))
        self.assertLessEqual(count(fitted), 900)

    def test_cache_replay_also_leaves_requested_output_room(self):
        messages = history()
        self.fit(messages)
        fitted, event = compact_context(
            messages, context_length=900, max_tokens=400, count_tokens=count,
            summarize=self.summarize, config=self.config, state_dir=self.root,
            scope="account-model",
        )
        self.assertLessEqual(count(fitted) + 400 + 16, 900)
        self.assertTrue(event["fits"])

    def test_extremely_large_history_has_a_bounded_number_of_summary_passes(self):
        messages = history()
        messages[1] = {"role": "user", "content": "old log " * 20000}
        self.assertIsNone(self.fit(messages))
        self.assertEqual(len(self.calls), 8)
        self.assertFalse(list(self.root.glob("state-*.json")))

    def test_gpu_reload_keeps_cache_identity_for_the_same_model_file(self):
        path = self.root / "model.gguf"
        path.write_bytes(b"fixture")
        first = SimpleNamespace(gguf_path=str(path), model_identifier="model", base_url="http://127.0.0.1:1")
        second = SimpleNamespace(gguf_path=str(path), model_identifier="model", base_url="http://127.0.0.1:2")
        self.assertEqual(model_scope(first), model_scope(second))

    def test_changed_model_file_does_not_reuse_cache_identity(self):
        path = self.root / "model.gguf"
        path.write_bytes(b"fixture")
        backend = SimpleNamespace(gguf_path=str(path), model_identifier="model")
        before = model_scope(backend)
        path.write_bytes(b"new fixture")
        self.assertNotEqual(before, model_scope(backend))


if __name__ == "__main__":
    unittest.main()
