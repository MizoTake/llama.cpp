const assert = require("node:assert/strict");
const test = require("node:test");
const { configureRoutes } = require("./harness_routes.cjs");

function fixture() {
    return [
        { id: "agent-default-model", config: { provider: "unslothdesktop", model: "UF27" } },
        { id: "llm-pi-ai", config: { providers: {
            lmstudio: { models: [{ id: "LM9", contextWindow: 131072, maxTokens: 8192 }] },
            "lmstudio-native": { apiKey: "fixture-key", models: [{ id: "LM27", contextWindow: 65536, maxTokens: 49152 }] },
            unslothdesktop: { models: [{ id: "UF27", contextWindow: 65536, maxTokens: 8192 }] },
        } } },
        { id: "compaction-basic", config: { auto: true, headroomTokens: 16384, maxTokens: 4096,
            summarizationProvider: "lmstudio-native", summarizationModel: "LM27",
            modelPolicies: [{ provider: "lmstudio-native", model: "LM27", headroomTokens: 0 }] } },
    ];
}

test("Removing a legacy provider retains the 27B budget and routes its summary to 27B", () => {
    const source = fixture();
    const result = configureRoutes(source, "http://127.0.0.1:8877/v1");
    const providers = result[1].config.providers;
    const compaction = result[2].config;
    assert.equal(providers["lmstudio-native"], undefined);
    assert.deepEqual(providers.lmstudio.models.find(model => model.id === "LM27"), { id: "LM27", contextWindow: 65536, maxTokens: 49152 });
    assert.equal(compaction.summarizationProvider, "lmstudio");
    assert.deepEqual(compaction.modelPolicies.find(policy => policy.model === "LM27"), {
        provider: "lmstudio", model: "LM27", headroomTokens: 0, summarizationProvider: "lmstudio", summarizationModel: "LM27",
    });
    assert.equal(compaction.modelPolicies.find(policy => policy.model === "LM9"), undefined);
    assert.equal(compaction.summarizationModel, "LM27");
    assert.deepEqual(result[0], source[0]);
    assert.ok(source[1].config.providers["lmstudio-native"]);
    assert.deepEqual(configureRoutes(result, "http://127.0.0.1:8877/v1"), result);
});

test("A legacy default moves to the canonical provider without changing its model", () => {
    const source = fixture();
    source[0].config = { provider: "lmstudio-native", model: "LM27" };
    assert.deepEqual(configureRoutes(source, "http://127.0.0.1:8877/v1")[0].config, { provider: "lmstudio", model: "LM27" });
});

test("Canonical budgets and policies take precedence over duplicate legacy entries", () => {
    const source = fixture();
    source[1].config.providers.lmstudio.models.push({ id: "LM27", contextWindow: 65536, maxTokens: 32768 });
    source[2].config.modelPolicies.push({ provider: "lmstudio", model: "LM27", headroomTokens: 512 });
    const result = configureRoutes(source, "http://127.0.0.1:8877/v1");
    assert.equal(result[1].config.providers.lmstudio.models.find(model => model.id === "LM27").maxTokens, 32768);
    assert.equal(result[2].config.modelPolicies.find(policy => policy.model === "LM27").headroomTokens, 512);
});
