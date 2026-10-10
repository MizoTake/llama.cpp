import { Chat, type PluginContext, type PredictionLoopHandlerController } from "@lmstudio/sdk";
import { readFile } from "node:fs/promises";
import { homedir } from "node:os";
import { join, resolve, relative, isAbsolute } from "node:path";

async function request(path: string, body: unknown, signal?: AbortSignal): Promise<any> {
    const root = join(homedir(), ".llama.cpp", "desktop");
    const config = JSON.parse(await readFile(join(root, "config.json"), "utf8"));
    const key = (await readFile(join(root, "api-key.txt"), "utf8")).trim();
    const response = await fetch("http://127.0.0.1:" + config.port + path, {
        method: "POST", headers: { "Content-Type": "application/json", Authorization: "Bearer " + key },
        body: JSON.stringify(body), signal,
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || data.error || "Local desktop service unavailable");
    return data;
}

function serialize(chat: Chat) {
    return chat.map(message => {
        const result: any = { role: message.getRole(), content: message.getText() };
        const calls = message.getToolCallRequests();
        if (calls.length) result.tool_calls = calls.map(call => ({ id: call.id, type: "function", function: { name: call.name, arguments: JSON.stringify(call.arguments || {}) } }));
        const responses = message.getToolCallResults();
        if (responses.length) {
            result.content = responses.map(response => response.content).join("\n");
            result.tool_call_id = responses[0].toolCallId;
        }
        if (message.hasFiles()) result.content += "\n[Attachments retained in original LM Studio chat]";
        return result;
    });
}

async function conversationId(chat: Chat): Promise<string | undefined> {
    try {
        const root = join(homedir(), ".lmstudio");
        const config = JSON.parse(await readFile(join(root, ".internal", "conversation-config.json"), "utf8"));
        const identifier = config.selectedConversation;
        if (typeof identifier !== "string" || !identifier.endsWith(".conversation.json")) return undefined;
        const directory = join(root, "conversations");
        const path = resolve(directory, identifier);
        const name = relative(directory, path);
        if (isAbsolute(name) || name.startsWith("..")) return undefined;
        const saved = JSON.parse(await readFile(path, "utf8"));
        const first = saved.messages.map((message: any) => message.versions[message.currentlySelected]).find((message: any) => message?.role === "user");
        const current = chat.getMessagesArray().find(message => message.getRole() === "user");
        const text = first?.content.filter((part: any) => part.type === "text").map((part: any) => part.text).join("");
        if (current && current.getText() === text) return name.replaceAll("\\", "/");
    } catch {
        // API-only histories use an explicit thread_id or a content fingerprint.
    }
    return undefined;
}

export async function predict(ctl: PredictionLoopHandlerController) {
    const status = ctl.createStatus({ status: "loading", text: "Preparing local model and conversation memory" });
    const history = await ctl.pullHistory();
    const original = history.getMessagesArray();
    const source = await ctl.tokenSource();
    if (!("identifier" in source)) throw new Error("Select a native LM Studio model for this processor");
    const session = await ctl.startToolUseSession();
    let lease: string | undefined;
    let heartbeat: ReturnType<typeof setInterval> | undefined;
    try {
        const prepared = await request("/lmstudio/prepare", {
            model: source.identifier, messages: serialize(history), thread_id: await conversationId(history),
            protected_indices: original.flatMap((message, index) => message.hasFiles() ? [index] : []),
            reserve_tokens: original.filter(message => message.hasFiles()).length * 4096 + (session.tools.length ? 4096 : 0),
        }, ctl.abortSignal);
        lease = prepared.lease;
        heartbeat = setInterval(() => { request("/lmstudio/heartbeat", { lease }).catch(() => {}); }, 15000);
        let cut = prepared.event.dropped_messages;
        const fitted = Chat.empty();
        for (const message of original.filter(message => message.getRole() === "system")) fitted.append(message);
        for (const message of prepared.shared) fitted.append("assistant", message.content);
        if (prepared.summary) fitted.append("assistant", prepared.summary);
        for (const message of original.filter(message => message.getRole() !== "system")) {
            if (cut > 0) { cut--; continue; }
            fitted.append(message);
        }
        ctl.guardAbort();
        status.setText("Local model ready");
        status.setState({ status: "done", text: "Conversation memory prepared" });
        const model = await ctl.tokenSource();
        let block = ctl.createContentBlock();
        let thinking: ReturnType<typeof ctl.createContentBlock> | undefined;
        const fragmentGenerated = (fragment: any) => {
            if (fragment.reasoningType === "reasoning") {
                if (!thinking) thinking = ctl.createContentBlock({ style: { type: "thinking", ended: false }, includeInContext: false });
                thinking.appendText(fragment.content);
            } else if (fragment.reasoningType === "none") {
                if (thinking) thinking.setStyle({ type: "thinking", ended: true });
                block.appendText(fragment.content, { tokensCount: fragment.tokensCount, fromDraftModel: fragment.containsDrafted, isStructural: fragment.isStructural });
            }
        };
        if (!session.tools.length) {
            const prediction = model.respond(fitted);
            const abort = () => { prediction.cancel(); };
            ctl.abortSignal.addEventListener("abort", abort, { once: true });
            try {
                for await (const fragment of prediction) fragmentGenerated(fragment);
                const result = await prediction;
                block.attachGenInfo({ indexedModelIdentifier: result.modelInfo.path, identifier: result.modelInfo.identifier, loadModelConfig: result.loadConfig, predictionConfig: result.predictionConfig, stats: result.stats });
                if (!ctl.abortSignal.aborted) {
                    history.append("assistant", result.nonReasoningContent);
                    await request("/lmstudio/complete", { lease, messages: serialize(history) });
                    lease = undefined;
                }
            } finally {
                ctl.abortSignal.removeEventListener("abort", abort);
            }
        } else {
            await model.act(fitted, session.tools, {
                signal: ctl.abortSignal,
                onRoundStart: round => { if (round) { block = ctl.createContentBlock(); thinking = undefined; } },
                onPredictionFragment: fragmentGenerated,
                onMessage: message => { history.append(message); },
                onToolCallRequestFinalized: (_round, callId, info) => {
                    const call = info.toolCallRequest;
                    const tool = session.tools.find(tool => tool.name === call.name);
                    block.appendToolRequest({ callId, toolCallRequestId: call.id, name: call.name,
                        parameters: call.arguments || {}, pluginIdentifier: tool && "pluginIdentifier" in tool ? String(tool.pluginIdentifier) : undefined });
                },
                onToolCallResult: (_round, callId, result) => {
                    block.appendToolResult({ callId, toolCallRequestId: result.toolCallId, name: result.name, content: result.content });
                },
                guardToolCall: async (_round, callId, guard) => {
                    const tool = guard.tool;
                    if (!("pluginIdentifier" in tool)) { guard.deny("Tool origin unavailable"); return; }
                    const result = await ctl.requestConfirmToolCall({ callId, pluginIdentifier: String(tool.pluginIdentifier), name: guard.toolCallRequest.name, parameters: guard.toolCallRequest.arguments || {} });
                    if (result.type === "allow" && result.toolArgsOverride) guard.allowAndOverrideParameters(result.toolArgsOverride);
                    else if (result.type === "allow") guard.allow();
                    else guard.deny(result.denyReason || "Tool not approved");
                },
            });
            if (!ctl.abortSignal.aborted) {
                await request("/lmstudio/complete", { lease, messages: serialize(history) });
                lease = undefined;
            }
        }
    } finally {
        if (heartbeat) clearInterval(heartbeat);
        if (lease) await request("/lmstudio/complete", { lease }).catch(() => {});
        session[Symbol.dispose]();
    }
}

export async function main(context: PluginContext) {
    context.withPredictionLoopHandler(predict);
}
