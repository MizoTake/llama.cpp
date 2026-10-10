const fs = require("node:fs");
const path = require("node:path");
const os = require("node:os");
const { createRequire } = require("node:module");

async function main() {
    const home = os.homedir();
    const modules = path.join(home, "AppData/Local/Programs/DeepSeek Harness/resources/app.asar/dsh/node_modules");
    const dsh = createRequire(path.join(modules, "@deepseek-ai/dsh-desktop-host/lib/cli.js"));
    const yaml = dsh("yaml");
    const patch = yaml.parse(fs.readFileSync(path.join(home, ".dsh/profiles/desktop/cordis.patch.yml"), "utf8"));
    const providers = patch.find(row => row.id === "llm-pi-ai").config.providers;
    const root = path.join(home, ".dsh/storages/session_projcache/sessions");
    const workspace = JSON.parse(fs.readFileSync(path.join(home, ".dsh/storages/workspace.json"), "utf8"));
    const sessions = await Promise.all(Object.values(workspace.tables.workspaces).flatMap(w => w.sessionIds).map(async session => {
        const file = path.join(root, session + ".json");
        const cached = fs.existsSync(file) ? JSON.parse(fs.readFileSync(file, "utf8")).record.rows.modelSelection?.val : undefined;
        const directory = Object.values(workspace.tables.workspaces).find(w => w.sessionIds.includes(session))?.path;
        const encoded = "--" + directory?.replace(/[:\\/]+/g, "-") + "--";
        const eventsPath = path.join(home, ".dsh/sessions", encoded, session, "session.v4.jsonl.zstd");
        const store = dsh("@deepseek-ai/dsh-session-persistence-jsonl").default;
        const events = fs.existsSync(eventsPath) ? (await store.prototype.readZstdPrefix.call({}, fs.readFileSync(eventsPath))).events : [];
        const selected = events.findLast(event => event.type === "model/selection");
        const used = events.findLast(event => event.type === "request/header");
        const last = used?.data.header.config;
        const selection = cached || { lastUsed: last ? { provider: last.provider, model: last.model } : null, pending: selected && (!used || selected.seq > used.seq) ? selected.data : null };
        const requests = events.filter(event => ["request/header", "model/selection", "turn/start", "turn/end"].includes(event.type)).slice(-18).map(event => {
            const value = event.data?.header?.config || event.data;
            return { type: event.type, seq: event.seq, time: event.time, provider: value?.provider, model: value?.model,
                ...(event.type === "turn/end" ? { reason: { kind: event.data.reason?.kind, errorCode: event.data.reason?.error?.code } } : {}) };
        });
        const start = events.findLast(event => event.type === "turn/start");
        const end = events.findLast(event => event.type === "turn/end");
        const activeTurn = !!start && (!end || start.seq > end.seq);
        const timings = events.filter(event => event.type === "assistant/message" && event.data.stream).slice(-3).map(event => {
            const request = events.findLast(item => item.type === "request/header" && item.seq < event.seq);
            const stream = event.data.stream;
            const first = stream.find(item => item.time || item.time0);
            return { seq: event.seq, provider: request?.data.header.config.provider, model: request?.data.header.config.model,
                firstChunkAfterHeaderMs: first && request ? (first.time || first.time0) - request.time : null,
                usage: event.data.usage, blocks: stream.filter(item => item.dt).map(item => ({ type: item.type, chunks: item.dt.length, elapsedMs: item.dt.reduce((sum, value) => sum + value, 0) })) };
        });
        return { session, selection, requests, timings, activeTurn, validPending: !selection?.pending || providers[selection.pending.provider]?.models.some(model => model.id === selection.pending.model) };
    }));
    const report = { defaults: patch.find(row => row.id === "agent-default-model").config,
        compaction: patch.find(row => row.id === "compaction-basic").config,
        providers: Object.fromEntries(Object.entries(providers).map(([id, p]) => [id, { endpoint: p.baseURL, models: p.models.map(m => ({ id: m.id, context: m.contextWindow, maxTokens: m.maxTokens })) }])), sessions };
    if (process.argv.includes("--check-idle")) {
        const active = sessions.filter(session => session.activeTurn);
        if (active.length) throw new Error("Wait for active DSH turns before changing providers: " + active.map(session => session.session).join(", "));
        console.log(JSON.stringify({ idle: true, sessions: sessions.length, validSelections: sessions.every(session => session.validPending),
            nextProviders: [...new Set(sessions.map(session => session.selection.pending?.provider || session.selection.lastUsed?.provider).filter(Boolean))] }));
    } else {
        fs.writeFileSync(path.join(home, ".llama.cpp/desktop/model-routing-audit-result.json"), JSON.stringify(report, null, 2) + "\n");
        console.log(JSON.stringify(report));
    }

}
main().catch(error => { console.error(error.message); process.exitCode = 1; });
