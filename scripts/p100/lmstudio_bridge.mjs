import { execFileSync } from "node:child_process";
import { readFile, writeFile } from "node:fs/promises";
import { pathToFileURL } from "node:url";
import { win32 } from "node:path";

const input = JSON.parse(await new Promise(resolve => {
    let data = "";
    process.stdin.setEncoding("utf8");
    process.stdin.on("data", chunk => { data += chunk; });
    process.stdin.on("end", () => resolve(data));
}));
const { LMStudioClient } = await import(pathToFileURL(input.sdk).href);
const client = new LMStudioClient();

function engines() {
    const text = execFileSync("powershell.exe", ["-NoProfile", "-Command", "Get-CimInstance Win32_Process -Filter \"Name='llama-server.exe'\" | Select-Object ProcessId,ExecutablePath,CommandLine | ConvertTo-Json -Compress"], { encoding: "utf8", windowsHide: true });
    const data = text.trim() ? JSON.parse(text) : [];
    return (Array.isArray(data) ? data : [data]).filter(row => row.ExecutablePath?.toLowerCase().startsWith(input.runtime.toLowerCase() + "\\"));
}

function argument(command, key) {
    return command.match(new RegExp("(?:^|\\s)" + key + "\\s+(?:\"([^\"]*)\"|(\\S+))"))?.slice(1).find(value => value !== undefined);
}

function idle(identifier) {
    const rows = JSON.parse(execFileSync(input.lms, ["ps", "--json"], { encoding: "utf8", windowsHide: true }));
    return rows.length === 1 && rows[0].identifier === identifier && rows[0].status === "idle" && rows[0].queued === 0;
}

function verifiedEngine(identifier) {
    const matches = engines();
    if (matches.length !== 1) throw new Error("Custom engine identity is ambiguous");
    const engine = matches[0];
    const expected = input.models[identifier]?.model_file;
    const actual = argument(engine.CommandLine, "--model");
    if (!expected || !actual || win32.normalize(actual).toLowerCase() !== win32.normalize(expected).toLowerCase()) throw new Error("Native model identity does not match the requested GGUF file");
    return engine;
}

function configured(normal, identifier) {
    const profile = input.models[identifier];
    const config = structuredClone(normal);
    const limit = input.hot ? profile.fallback_context_limit : profile.context_limit;
    config.contextLength = Math.min(normal.contextLength, limit || input.contextLimit || 65536);
    Object.assign(config, input.hot ? profile.fallback_load : profile.primary_load);
    config.llamaCppArgumentsOverride = {
        enabled: true, excludeAllConfig: false,
        disabledParameters: ["--device", "--tensor-split", "--split-mode", "--main-gpu", "--fit", "--cache-ram", "-cram", "--cache-idle-slots", "--no-cache-idle-slots"],
        overrideParameters: [
            { key: "--device", value: input.devices },
            { key: "--tensor-split", value: input.hot ? "1" : "3,2" },
            { key: "--split-mode", value: input.hot ? "none" : "layer" },
            { key: "--main-gpu", value: "0" },
            { key: "--fit", value: "off" },
            { key: "--cache-ram", value: String(input.ramCacheMiB ?? 8192) },
            { key: "--cache-idle-slots", value: "" },
        ],
    };
    return config;
}

try {
    let models = await client.llm.listLoaded();
    if (models.length > 1) throw new Error("Multiple LM Studio models are loaded");
    let model = models[0];
    if (!model && input.action === "load" && models.length === 0) {
        const normal = JSON.parse(await readFile(input.profiles[input.identifier], "utf8"));
        model = await client.llm.load(input.models[input.identifier].modelKey, { identifier: input.identifier, config: configured(normal, input.identifier), verbose: false });
        models = await client.llm.listLoaded();
    }
    if (!model) throw new Error("Model not loaded");
    const profile = input.models[model.identifier];
    if (!profile || profile.modelKey !== model.modelKey) throw new Error("Unregistered LM Studio model");
    if (input.action === "reload" || input.action === "unload" || input.action === "select") {
        const before = model.instanceReference;
        const originalIdentifier = model.identifier;
        const identifier = input.action === "select" ? input.identifier : originalIdentifier;
        const target = input.models[identifier];
        if (!target) throw new Error("Unregistered LM Studio model");
        if (!idle(originalIdentifier)) throw new Error("Model busy");
        const restore = await model.getLoadConfig();
        let normal;
        try { normal = JSON.parse(await readFile(input.profiles[identifier], "utf8")); }
        catch (error) {
            if (error.code !== "ENOENT") throw error;
            if (identifier !== originalIdentifier) throw new Error("Requested model profile not captured");
            normal = await model.getLoadConfig();
            await writeFile(input.profiles[identifier], JSON.stringify(normal), { encoding: "utf8", flag: "wx" });
        }
        const config = configured(normal, identifier);
        const engine = verifiedEngine(originalIdentifier);
        const port = argument(engine.CommandLine, "--port");
        const key = argument(engine.CommandLine, "--api-key");
        const slotsResponse = await fetch("http://127.0.0.1:" + port + "/slots", { headers: { Authorization: "Bearer " + key }, signal: AbortSignal.timeout(5000) });
        if (!slotsResponse.ok) throw new Error("Cannot check native engine slots");
        const slots = await slotsResponse.json();
        if (!Array.isArray(slots) || slots.some(slot => slot.is_processing)) throw new Error("Native engine busy");
        models = await client.llm.listLoaded();
        model = models.find(item => item.identifier === originalIdentifier);
        if (!model || model.instanceReference !== before || !idle(originalIdentifier) || models.length !== 1) throw new Error("Model state changed");
        await model.unload();
        if (input.action === "unload") {
            process.stdout.write(JSON.stringify({ unloaded: true, identifier }) + "\n");
            process.exit(0);
        }
        try {
            model = await client.llm.load(target.modelKey, { identifier, config, verbose: false });
        } catch (error) {
            await client.llm.load(profile.modelKey, { identifier: originalIdentifier, config: restore, verbose: false });
            throw error;
        }
    }
    const engine = verifiedEngine(model.identifier);
    const port = argument(engine.CommandLine, "--port");
    if (!/^\d+$/.test(port) || +port < 1 || +port > 65535) throw new Error("Invalid engine port");
    const key = argument(engine.CommandLine, "--api-key");
    const slotsResponse = await fetch("http://127.0.0.1:" + port + "/slots", { headers: { Authorization: "Bearer " + key }, signal: AbortSignal.timeout(5000) });
    if (!slotsResponse.ok) throw new Error("Cannot check native engine slots");
    const slots = await slotsResponse.json();
    process.stdout.write(JSON.stringify({ port: +port, key, pid: engine.ProcessId, native_model: argument(engine.CommandLine, "--model"), devices: argument(engine.CommandLine, "--device"), cache_ram_mib: +(argument(engine.CommandLine, "--cache-ram") ?? argument(engine.CommandLine, "-cram") ?? 8192), context: await model.getContextLength(), identifier: model.identifier, modelKey: model.modelKey, idle: idle(model.identifier) && Array.isArray(slots) && slots.every(slot => !slot.is_processing) }) + "\n");
} catch (error) {
    process.stdout.write(JSON.stringify({ error: error.message }) + "\n");
    process.exitCode = 1;
}
process.exit(process.exitCode || 0);
