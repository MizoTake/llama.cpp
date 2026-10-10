import tempfile
import unittest
from pathlib import Path

from shared_memory import MemoryStore, add_shared_memory


class SharedMemoryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.store = MemoryStore(Path(self.temporary.name) / "memory.sqlite3")

    def test_other_app_summary_is_available_without_current_conversation(self):
        self.store.remember("unsloth", "one", [{"role": "user", "content": "Keep F16"}], "Use F16 KV.")
        self.store.remember("lmstudio", "two", [{"role": "user", "content": "Current"}], "Current context")
        rows = self.store.select("lmstudio", "two", "F16")
        self.assertEqual([row["conversation"] for row in rows], ["one"])

    def test_manual_summary_and_exclusion_survive_automatic_updates(self):
        record = self.store.remember("unsloth", "one", [{"role": "user", "content": "old"}], "old")
        self.store.edit(record, "Corrected fact", shared=False)
        self.store.remember("unsloth", "one", [{"role": "user", "content": "new"}], "new")
        row = self.store.get(record)
        self.assertEqual(row["summary"], "Corrected fact")
        self.assertEqual(row["shared"], 0)
        self.assertEqual(self.store.select("lmstudio", "two", "fact"), [])

    def test_memory_is_reference_data_and_respects_budget(self):
        self.store.remember("unsloth", "one", [{"role": "user", "content": "x"}], "reference " * 100)
        messages = [{"role": "system", "content": "Current instruction"}, {"role": "user", "content": "new"}]
        fitted, ids = add_shared_memory(messages, self.store, "lmstudio", "two", lambda m: sum(len(x["content"]) for x in m), 500)
        self.assertEqual(ids, [])
        self.assertEqual(fitted, messages)
        fitted, ids = add_shared_memory(messages, self.store, "lmstudio", "two", lambda m: sum(len(x["content"]) for x in m), 2048)
        self.assertEqual(fitted[0], messages[0])
        self.assertEqual(fitted[-1], messages[-1])
        self.assertEqual(fitted[1]["role"], "assistant")
        self.assertIn("reference only", fitted[1]["content"])
        self.assertEqual(len(ids), 1)

    def test_short_conversation_has_automatic_extract_and_original_history(self):
        messages = [{"role": "user", "content": "Keep P100 first"}]
        record = self.store.remember("lmstudio", "one", messages)
        row = self.store.get(record)
        self.assertIn("P100 first", row["summary"])
        self.assertEqual(row["messages"], messages)

    def test_changed_history_retains_prior_summary_versions(self):
        record = self.store.remember("lmstudio", "one", [{"role": "user", "content": "first"}], "first summary")
        self.store.remember("lmstudio", "one", [{"role": "user", "content": "second"}], "second summary")
        self.assertEqual(len(self.store.get(record)["versions"]), 2)

    def test_background_result_cannot_replace_newer_history(self):
        record = self.store.remember("lmstudio", "one", [{"role": "user", "content": "old"}])
        _, digest, _messages = self.store.pending()
        self.store.remember("lmstudio", "one", [{"role": "user", "content": "new"}])
        self.assertFalse(self.store.finish_summary(record, digest, "Stale summary"))
        self.assertIn("new", self.store.get(record)["summary"])

    def test_background_summary_is_shared_and_manual_edits_are_protected(self):
        record = self.store.remember("unsloth", "one", [{"role": "user", "content": "F16"}])
        _, digest, _messages = self.store.pending()
        self.assertTrue(self.store.finish_summary(record, digest, "Keep F16 KV"))
        self.assertIsNone(self.store.pending())
        self.store.edit(record, "Manual correction")
        self.assertFalse(self.store.finish_summary(record, digest, "Overwrite"))

    def test_excluded_memory_is_not_summarized_in_background(self):
        record = self.store.remember("lmstudio", "one", [{"role": "user", "content": "private"}])
        self.store.edit(record, shared=False)
        self.assertIsNone(self.store.pending())

    def test_reimport_does_not_change_order_or_replace_completed_summary(self):
        messages = [{"role": "user", "content": "original"}]
        record = self.store.remember("lmstudio", "one", messages)
        _, digest, _ = self.store.pending()
        self.store.finish_summary(record, digest, "Completed summary")
        before = self.store.get(record)
        self.store.remember("lmstudio", "one", messages)
        self.assertEqual(self.store.get(record), before)

    def test_sharing_can_be_disabled_without_losing_context_compaction(self):
        from shared_memory import prepare_context

        self.store.remember("unsloth", "one", [{"role": "user", "content": "other"}], "Shared fact")
        messages = [{"role": "user", "content": "current"}]
        fitted, event = prepare_context(messages, app="lmstudio", thread="two", context_length=2048, max_tokens=128,
            count_tokens=lambda items: sum(len(item["content"].split()) for item in items), summarize=lambda *_: "summary",
            config={"trigger_tokens": 1000, "recent_turns": 4, "summary_tokens": 128, "safety_tokens": 16},
            state_dir=Path(self.temporary.name), scope="two", store=self.store, share_memory=False)
        self.assertEqual(fitted, messages)
        self.assertEqual(event["shared_memory_ids"], [])

    def test_rare_subject_terms_rank_before_common_transcript_words(self):
        useful = self.store.remember("lmstudio", "one", [{"role": "user", "content": "reference"}], "Validation crossapp fixture code SHARED9172.")
        self.store.remember("unsloth", "two", [{"role": "user", "content": "other"}], "What user conversation replied with the previous code is only the current content. " * 30)
        rows = self.store.select("unsloth", "current", "What is the validation crossapp fixture code? Reply only with the code.")
        self.assertEqual(rows[0]["id"], useful)

    def test_stream_completion_retains_split_unicode_and_tool_arguments(self):
        from shared_memory import CompletionCapture

        capture = CompletionCapture()
        first = 'data: {"choices":[{"delta":{"content":"接続"}}]}\n\n'.encode()
        capture.feed(first[:40])
        capture.feed(first[40:])
        self.assertIsNone(capture.result())
        capture.add({"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "one", "function": {"name": "check", "arguments": '{"key":'}}]}}]})
        capture.add({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '42}'}}]}, "finish_reason": "tool_calls"}]})
        message = capture.result()
        self.assertEqual(message["content"], "接続")
        self.assertEqual(message["tool_calls"][0]["function"], {"name": "check", "arguments": '{"key":42}'})

    def test_agent_search_returns_project_and_global_memories(self):
        project = self.store.remember("codex", "one", [], "Keep F16", project="project-a", sharing_scope="project")
        common = self.store.remember("claude-code", "two", [], "Use Japanese", sharing_scope="global")
        self.store.remember("codex", "three", [], "Other project", project="project-b", sharing_scope="project")
        self.store.remember("lmstudio", "four", [], "Unclassified history")
        rows = self.store.search("F16", project="project-a")
        self.assertEqual({row["id"] for row in rows}, {project, common})
        self.assertEqual(len(self.store.search("", scope="all")), 4)

    def test_manual_project_binding_survives_import(self):
        record = self.store.remember("codex", "one", [], "first", project="a", sharing_scope="project")
        self.store.bind(record, project="b", sharing_scope="project")
        self.store.remember("codex", "one", [{"role": "user", "content": "new"}], project="a", sharing_scope="project")
        self.assertEqual(self.store.get(record)["project"], "b")

    def test_expired_memory_is_not_shared_or_summarized(self):
        self.store.remember("codex", "one", [], "Old state", expires="2000-01-01T00:00:00+00:00")
        self.assertEqual(self.store.search("", scope="all"), [])
        self.assertEqual(self.store.select("lmstudio", "current", "state"), [])
        self.assertIsNone(self.store.pending())

    def test_correction_keeps_previous_content_and_source(self):
        record = self.store.remember("claude-code", "one", [], "Original", provenance={"turn": "turn-a"})
        self.store.correct(record, "Corrected", "The user corrected the assumption")
        value = self.store.get(record)
        self.assertEqual(value["summary"], "Corrected")
        self.assertEqual(value["corrections"][0]["previous"], "Original")
        self.assertEqual(value["provenance"], {"turn": "turn-a"})

    def test_invalid_scope_and_source_are_rejected(self):
        for fields in ({"app": "unknown"}, {"sharing_scope": "public"}, {"kind": "permission"}, {"expires": "invalid"}):
            with self.assertRaises(ValueError):
                self.store.remember(**{"app": "codex", "conversation": "one", "messages": [], **fields})


if __name__ == "__main__":
    unittest.main()
