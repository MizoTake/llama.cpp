import argparse
import ast
import hashlib
import json
import sys
import uuid
from pathlib import Path

from context_unsloth import DEFAULTS, ROOT, _write

SUPPORTED = {
    "core/inference/llama_cpp.py": "37cb66179cf205ebd9d70deb63a084ce3c75cc732ee33dec5da53fc924a443ea",
    "routes/inference.py": "f72902b3b925280838f638f92403d27cb0e18c2786956463d474420cd95f70aa",
}
MARKER = "# P100 local context summary integration v1"
LLAMA_WRAPPER = '''
# P100 local context summary integration v1
_p100_original_fit = _fit_with_instruction_pins


def _fit_with_instruction_pins(messages, *, summary_backend=None, summary_thread_id=None, summary_cancel_event=None, **kwargs):
    from core.inference.p100_context import fit_summary_context

    result = fit_summary_context(
        messages, backend=summary_backend, thread_id=summary_thread_id,
        cancel_event=summary_cancel_event, **kwargs,
    )
    if result is not None:
        return result
    return _p100_original_fit(messages, **kwargs)
'''
ROUTE_WRAPPER = '''
# P100 local context summary integration v1
_p100_original_overflow_policy = _overflow_truncation_policy
_p100_original_passthrough_body = _build_openai_passthrough_body_async


def _overflow_truncation_policy(payload):
    from core.inference.p100_context import enabled

    if not (getattr(payload, "provider_id", None) or getattr(payload, "provider_type", None)) and enabled():
        return "truncate_oldest"
    return _p100_original_overflow_policy(payload)


async def _build_openai_passthrough_body_async(payload, backend_ctx=None, llama_backend=None, summary_cancel_event=None):
    body = await _p100_original_passthrough_body(payload, backend_ctx=backend_ctx, llama_backend=llama_backend)
    if summary_cancel_event is not None:
        from core.inference.p100_context import fit_passthrough

        body = await asyncio.to_thread(fit_passthrough, body, llama_backend, payload.thread_id, summary_cancel_event)
    return body


def _p100_archive_response(original):
    from functools import wraps
    from inspect import signature

    @wraps(original)
    async def wrapped(*args, **kwargs):
        response = await original(*args, **kwargs)
        from core.inference.p100_context import archive_passthrough_response

        payload = signature(original).bind(*args, **kwargs).arguments["payload"]
        return await archive_passthrough_response(response, payload)

    return wrapped


produce_openai_chat_completions = _p100_archive_response(produce_openai_chat_completions)
'''


def digest(data):
    return hashlib.sha256(data).hexdigest()


def patched_source(source, kind):
    if MARKER in source:
        raise ValueError("Integration already exists")
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    changes = []
    for function in ast.walk(tree):
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if kind == "routes" and function.name not in (
            "_openai_passthrough_stream_admitted", "_openai_passthrough_non_streaming_upstream",
        ):
            continue
        for call in ast.walk(function):
            if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
                continue
            if kind == "llama" and call.func.id == "_fit_with_instruction_pins" and any(
                isinstance(node, ast.Name) and node.id == "self" for node in ast.walk(call)
            ):
                addition = "summary_backend = self, summary_thread_id = thread_id, summary_cancel_event = cancel_event,"
            elif kind == "routes" and call.func.id == "_build_openai_passthrough_body_async":
                addition = "summary_cancel_event = cancel_event,"
            else:
                continue
            # Nested function walks can encounter the same call more than once.
            first_keyword = call.keywords[0]
            position = (first_keyword.lineno - 1, first_keyword.col_offset)
            if any(location == position for location, _ in changes):
                continue
            newline = "\r\n" if lines[call.lineno - 1].endswith("\r\n") else "\n"
            changes.append((position, addition + newline + " " * first_keyword.col_offset))
    expected = 6 if kind == "llama" else 2
    if len(changes) != expected:
        raise ValueError(f"Expected {expected} integration calls; found {len(changes)}")
    for (line, column), addition in sorted(changes, reverse=True):
        lines[line] = lines[line][:column] + addition + lines[line][column:]
    newline = "\r\n" if "\r\n" in source else "\n"
    wrapper = LLAMA_WRAPPER if kind == "llama" else ROUTE_WRAPPER
    output = "".join(lines) + wrapper.replace("\n", newline)
    compile(output, "<Unsloth context integration>", "exec")
    return output


def install(backend, root, addon, dry_run=False, shared=None):
    manifest_path = root / "installation.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if Path(manifest["backend"]).resolve() != backend.resolve():
            raise RuntimeError("The integration belongs to another Desktop environment")
        if all(digest((backend / entry["path"]).read_bytes()) == entry["installed"] for entry in manifest["files"]):
            if manifest["addon_hash"] != digest(addon.read_bytes()):
                raise RuntimeError("Restore the old integration before installing changed source")
            if shared is not None and manifest.get("shared_hash") != digest(shared.read_bytes()):
                raise RuntimeError("Restore the old integration before installing changed shared helper")
            print("CONTEXT_ALREADY_INSTALLED")
            return
        raise RuntimeError("Installed files changed; review the Desktop update before reinstalling")
    planned = []
    for name, expected in SUPPORTED.items():
        before = (backend / name).read_bytes()
        if digest(before) != expected:
            raise RuntimeError("Unsupported Desktop source revision: " + name)
        text = before.decode("utf-8")
        kind = "llama" if name.startswith("core/") else "routes"
        planned.append((name, before, patched_source(text, kind).encode("utf-8")))
    name = "core/inference/p100_context.py"
    if (backend / name).exists():
        raise RuntimeError("An unmanaged context helper already exists")
    data = addon.read_bytes()
    compile(data, str(addon), "exec")
    planned.append((name, None, data))
    if shared is not None:
        shared_name = "core/inference/p100_shared_memory.py"
        if (backend / shared_name).exists():
            raise RuntimeError("An unmanaged shared memory helper already exists")
        shared_data = shared.read_bytes()
        compile(shared_data, str(shared), "exec")
        planned.append((shared_name, None, shared_data))
    if dry_run:
        print("CONTEXT_INSTALL_DRY_RUN", len(planned))
        return
    root.mkdir(parents=True, exist_ok=True)
    entries = []
    for index, (name, before, after) in enumerate(planned):
        backup = f"original-{index}.py" if before is not None else None
        if before is not None:
            (root / backup).write_bytes(before)
        entries.append({"path": name, "backup": backup, "original": digest(before) if before is not None else None, "installed": digest(after)})
    _write(manifest_path, {"backend": str(backend), "addon_hash": digest(data), "shared_hash": digest(shared.read_bytes()) if shared is not None else None, "files": entries})
    try:
        for name, _, after in planned:
            (backend / name).write_bytes(after)
        if not (root / "config.json").exists():
            _write(root / "config.json", DEFAULTS)
    except Exception:
        for name, before, _ in planned:
            if before is None:
                (backend / name).unlink(missing_ok=True)
            else:
                (backend / name).write_bytes(before)
        manifest_path.unlink()
        raise
    print("CONTEXT_INSTALLED", manifest_path)


def restore(backend, root, dry_run=False):
    manifest_path = root / "installation.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if Path(manifest["backend"]).resolve() != backend.resolve():
        raise RuntimeError("The backup belongs to another Desktop environment")
    for entry in manifest["files"]:
        if digest((backend / entry["path"]).read_bytes()) != entry["installed"]:
            raise RuntimeError("Refusing to overwrite a changed Desktop file: " + entry["path"])
        if entry["backup"] and digest((root / entry["backup"]).read_bytes()) != entry["original"]:
            raise RuntimeError("Backup hash mismatch")
    if dry_run:
        print("CONTEXT_RESTORE_DRY_RUN")
        return
    for entry in manifest["files"]:
        path = backend / entry["path"]
        if entry["backup"]:
            path.write_bytes((root / entry["backup"]).read_bytes())
        else:
            path.unlink()
    manifest_path.unlink()
    print("CONTEXT_RESTORED")


def update_shared(backend, root, shared, dry_run=False):
    manifest_path = root / "installation.json"
    before_manifest = manifest_path.read_bytes()
    manifest = json.loads(before_manifest)
    if Path(manifest["backend"]).resolve() != backend.resolve():
        raise RuntimeError("The integration belongs to another Desktop environment")
    for entry in manifest["files"]:
        if digest((backend / entry["path"]).read_bytes()) != entry["installed"]:
            raise RuntimeError("Installed integration changed: " + entry["path"])
    name = "core/inference/p100_shared_memory.py"
    entry = next((entry for entry in manifest["files"] if entry["path"] == name), None)
    if entry is None:
        raise RuntimeError("Install the shared helper before updating it")
    data = shared.read_bytes()
    compile(data, str(shared), "exec")
    if digest(data) == entry["installed"]:
        print("CONTEXT_SHARED_ALREADY_UPDATED")
        return
    if dry_run:
        print("CONTEXT_SHARED_UPDATE_DRY_RUN")
        return
    target = backend / name
    before = target.read_bytes()
    backup = root / ("shared-before-" + uuid.uuid4().hex + ".py")
    backup.write_bytes(before)
    entry["installed"] = digest(data)
    manifest["shared_hash"] = digest(data)
    temporary = target.with_name(target.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_bytes(data)
        temporary.replace(target)
        _write(manifest_path, manifest)
    except Exception:
        target.write_bytes(before)
        manifest_path.write_bytes(before_manifest)
        raise
    finally:
        temporary.unlink(missing_ok=True)
    print("CONTEXT_SHARED_UPDATED_RELOAD_REQUIRED")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--restore", action="store_true")
    parser.add_argument("--update-shared", action="store_true")
    args = parser.parse_args()
    backend = Path(sys.prefix) / "Lib/site-packages/studio/backend"
    if not backend.is_dir():
        raise RuntimeError("Use the Unsloth Desktop Python environment")
    if args.update_shared:
        update_shared(backend, ROOT, Path(__file__).with_name("shared_memory.py"), args.dry_run)
    elif args.restore:
        restore(backend, ROOT, args.dry_run)
    else:
        shared = Path(__file__).with_name("shared_memory.py")
        install(backend, ROOT, Path(__file__).with_name("context_unsloth.py"), args.dry_run, shared if shared.exists() else None)


if __name__ == "__main__":
    main()
