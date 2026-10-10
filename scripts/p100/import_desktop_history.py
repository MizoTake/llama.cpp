import argparse
import json
from collections import defaultdict
from pathlib import Path

from shared_memory import MemoryStore, conversation_key
from thermal_unsloth import StudioApi


def lm_messages(value):
    messages = []
    if value.get("systemPrompt"):
        messages.append({"role": "system", "content": value["systemPrompt"]})
    for message in value.get("messages", []):
        versions = message.get("versions", [])
        index = message.get("currentlySelected", 0)
        if type(index) is not int or not 0 <= index < len(versions):
            continue
        selected = versions[index]
        role = selected.get("role")
        if role not in ("user", "assistant", "system", "tool"):
            continue
        parts = selected.get("content", []) if selected.get("type") == "singleStep" else [
            part for step in selected.get("steps", []) if step.get("style", {}).get("type") != "thinking" and step.get("shouldIncludeInContext", step.get("defaultShouldIncludeInContext", True))
            for part in step.get("content", [])
        ]
        text = "\n".join(part["text"] for part in parts if part.get("type") == "text" and isinstance(part.get("text"), str))
        if text:
            messages.append({"role": role, "content": text, "_original": message})
    return messages


def import_lmstudio(home, store):
    count = 0
    directory = home / "conversations"
    for file in sorted(directory.rglob("*.conversation.json")):
        if file.stat().st_size > 32 * 1024 * 1024:
            continue
        try:
            value = json.loads(file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        messages = lm_messages(value)
        if any(message["role"] == "user" for message in messages):
            canonical = [{"role": message["role"], "content": message["content"]} for message in messages]
            identifier = file.relative_to(directory).as_posix()
            with store.connect() as db:
                old = conversation_key(canonical)
                if not db.execute("SELECT id FROM conversations WHERE app='lmstudio' AND conversation=?", (identifier,)).fetchone():
                    db.execute("UPDATE conversations SET conversation=? WHERE app='lmstudio' AND conversation=?", (identifier, old))
            store.remember("lmstudio", identifier, messages)
            count += 1
    return count


def import_unsloth(api, store):
    export = api.request("/api/chat/export")
    if export.get("version") != 1:
        raise ValueError("Unknown Unsloth export schema")
    grouped = defaultdict(list)
    for message in sorted(export["messages"], key=lambda item: item["createdAt"]):
        grouped[message["threadId"]].append({"role": message["role"], "content": message["content"], "_original": message})
    count = 0
    for thread in export["threads"]:
        messages = grouped[thread["id"]]
        if messages:
            store.remember("unsloth", thread["id"], messages)
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--app", choices=("both", "lmstudio", "unsloth"), default="both")
    args = parser.parse_args()
    store = MemoryStore()
    result = {}
    if args.app in ("both", "lmstudio"):
        result["lmstudio"] = import_lmstudio(Path.home() / ".lmstudio", store)
    if args.app in ("both", "unsloth"):
        result["unsloth"] = import_unsloth(StudioApi("http://127.0.0.1:8888"), store)
    print(json.dumps({"imported_conversations": result}))


if __name__ == "__main__":
    main()
