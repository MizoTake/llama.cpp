const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const { createRequire } = require("node:module");

async function main() {
    const home = os.homedir();
    const dsh = createRequire(path.join(home, "AppData/Local/Programs/DeepSeek Harness/resources/app.asar/dsh/node_modules/@deepseek-ai/dsh-desktop-host/lib/cli.js"));
    const patch = dsh("yaml").parse(fs.readFileSync(path.join(home, ".dsh/profiles/desktop/cordis.patch.yml"), "utf8"));
    const config = patch.find(row => row.id === "compaction-basic").config;
    const providers = patch.find(row => row.id === "llm-pi-ai").config.providers;
    const { BasicCompactionEngine } = dsh("@deepseek-ai/dsh-compaction-basic");
    const calls = [];
    const ctx = { llm: { async *stream(options) {
        calls.push({ provider: options.provider, model: options.model, purpose: options.purpose });
        yield { type: "block-start", index: 0, blockType: "text" };
        yield { type: "text-delta", index: 0, text: "Preserve F16 KV and P100 priority." };
        yield { type: "block-end", index: 0, block: { type: "text", text: "Preserve F16 KV and P100 priority." } };
        yield { type: "finish", reason: { kind: "stop" } };
    } } };
    for (const model of providers.lmstudio.models) {
        const agent = { options: { provider: "lmstudio", model: config.summarizationModel }, session: {
            id: "routing-regression-fixture", requestHeader: () => ({ config: { provider: "lmstudio", model: model.id } }), toolHistory: () => [],
        } };
        await BasicCompactionEngine.prototype.summarize.call({ ctx, config }, { messages: [{ role: "user", content: [{ type: "text", text: "Keep F16 KV." }] }] }, agent);
        assert.deepEqual(calls.at(-1), { provider: "lmstudio", model: config.summarizationModel, purpose: "compaction" }, "A prior 9B header cannot redirect 27B compaction to 9B");
    }
    console.log(JSON.stringify({ installedCompactionChecked: true, modelRoutes: calls, generationMocked: true }));
}

main().catch(error => { console.error(error.message); process.exitCode = 1; });
