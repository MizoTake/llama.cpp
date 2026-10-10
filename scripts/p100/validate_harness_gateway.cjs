const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const os = require("node:os");
const { createRequire } = require("node:module");

async function main() {
    const home = os.homedir();
    const app = path.join(home, "AppData", "Local", "Programs", "DeepSeek Harness", "resources", "app.asar", "dsh", "node_modules", "@deepseek-ai", "dsh-desktop-host", "lib", "cli.js");
    const dsh = createRequire(app);
    const patch = dsh("yaml").parse(fs.readFileSync(path.join(home, ".dsh", "profiles", "desktop", "cordis.patch.yml"), "utf8"));
    const defaults = patch.find(row => row.id === "agent-default-model").config;
    const providerArgument = process.argv.indexOf("--provider");
    if (providerArgument !== -1) defaults.provider = process.argv[providerArgument + 1];
    const compaction = patch.find(row => row.id === "compaction-basic").config;
    const providers = patch.find(row => row.id === "llm-pi-ai").config.providers;
    const config = JSON.parse(fs.readFileSync(path.join(home, ".llama.cpp", "desktop", "config.json"), "utf8"));
    const modelArgument = process.argv.indexOf("--model");
    if (modelArgument !== -1) defaults.model = process.argv[modelArgument + 1];
    else if (!providers[defaults.provider]?.models.some(model => model.id === defaults.model)) defaults.model = config.lmstudio.identifier;
    assert.ok(providers[defaults.provider]?.models.some(model => model.id === defaults.model), "The tested model is registered");
    const policy = { ...compaction, ...compaction.modelPolicies?.find(policy => policy.provider === defaults.provider && policy.model === defaults.model) };
    const endpoint = "http://127.0.0.1:" + config.port + "/v1";
    assert.equal(providers["lmstudio-native"], undefined, "The unused legacy provider is removed");
    for (const name of ["lmstudio", "unslothdesktop"]) {
        assert.equal(providers[name].baseURL, endpoint, "Every desktop route uses the shared gateway");
        assert.equal(providers[name].apiKeyEnv, "LLAMA_DESKTOP_API_KEY");
    }
    const ctx = new (dsh("@deepseek-ai/cordis").Context)();
    const requests = [];
    const originalFetch = globalThis.fetch;
    globalThis.fetch = async (input, init) => {
        const url = typeof input === "string" ? input : input.url;
        let request;
        if (url.endsWith("/chat/completions")) {
            assert.equal(url, endpoint + "/chat/completions", "Generation cannot bypass the gateway");
            const body = JSON.parse(typeof init?.body === "string" ? init.body : await input.clone().text());
            request = { model: body.model, maxTokens: body.max_tokens };
            requests.push(request);
        }
        const response = await originalFetch(input, init);
        if (request) {
            request.httpStatus = response.status;
            if (!response.ok) request.error = await response.clone().json();
        }
        return response;
    };
    try {
        await ctx.plugin(dsh("@deepseek-ai/dsh-credentials-local").LocalCredentialProvider, { path: path.join(home, ".dsh", ".credentials.yaml"), watch: false });
        await ctx.plugin(dsh("@deepseek-ai/dsh-llm").LlmRuntime);
        await ctx.plugin(dsh("@deepseek-ai/dsh-llm-pi-ai"), { providers });
        const prepared = await ctx.llm.prepareCall(defaults);
        const summary = await ctx.llm.prepareCall({ provider: policy.summarizationProvider, model: policy.summarizationModel });
        assert.equal(summary.config.model, defaults.model, "LM Studio compaction cannot inherit the previous 9B route");
        const start = Date.now();
        let text = "";
        let finish;
        let firstChunkMs;
        for await (const chunk of ctx.llm.stream({ ...prepared.config, maxTokens: 128,
            messages: [{ role: "user", content: [{ type: "text", text: "What is 6 times 7? Reply only 42." }] }],
            temperature: 0, signal: AbortSignal.timeout(180000),
        })) {
            if (firstChunkMs === undefined && ["text-delta", "reasoning-delta", "block-start"].includes(chunk.type)) firstChunkMs = Date.now() - start;
            if (chunk.type === "text-delta") text += chunk.text ?? chunk.delta ?? "";
            if (chunk.type === "finish") finish = chunk;
        }
        assert.equal(text.trim(), "42", "The configured native model answers through the gateway; finish=" + JSON.stringify(finish?.reason) + "; requests=" + JSON.stringify(requests));
        assert.ok(finish && requests.length > 0, "The generation finishes through the configured route");
        assert.ok(requests.every(request => request.model === defaults.model), "Every request uses the selected model");
        assert.equal(finish.replayState.response.responseModel, config.lmstudio.models[defaults.model].model_file, "The native response identifies the selected GGUF file");
        const report = { endpoint, defaultModel: defaults, summarizationModel: summary.config.model, configuredMaxTokens: prepared.config.maxTokens, requests,
            elapsedMs: Date.now() - start, firstChunkMs, text, finish, fullOutputBudgetTested: false };
        fs.writeFileSync(path.join(home, ".llama.cpp", "desktop", "gateway-harness-smoke-result.json"), JSON.stringify(report, null, 2) + "\n");
        console.log(JSON.stringify(report));
    } finally {
        await ctx.fiber.dispose();
        globalThis.fetch = originalFetch;
    }
}

main().catch(error => { console.error(error.name + ": " + error.message); process.exitCode = 1; });
