const assert = require("node:assert/strict");

const ALIASES = { "lmstudio-native": "lmstudio" };

function canonicalProvider(provider) {
    return ALIASES[provider] || provider;
}

function configureRoutes(patch, endpoint) {
    const result = structuredClone(patch);
    const providers = result.find(row => row.id === "llm-pi-ai").config.providers;
    const defaults = result.find(row => row.id === "agent-default-model").config;
    const compaction = result.find(row => row.id === "compaction-basic").config;
    for (const [alias, name] of Object.entries(ALIASES)) {
        if (!providers[alias]) continue;
        const current = providers[name] || providers[alias];
        const models = new Map(current.models.map(model => [model.id, model]));
        for (const model of providers[alias].models) if (!models.has(model.id)) models.set(model.id, model);
        providers[name] = { ...current, models: [...models.values()] };
        delete providers[alias];
    }
    defaults.provider = canonicalProvider(defaults.provider);
    compaction.summarizationProvider = canonicalProvider(compaction.summarizationProvider);
    const policies = new Map();
    for (const policy of compaction.modelPolicies || []) {
        const provider = canonicalProvider(policy.provider);
        const key = JSON.stringify([provider, policy.model]);
        const previous = policies.get(key) || {};
        policies.set(key, policy.provider === provider ? { ...previous, ...policy, provider } : { ...policy, ...previous, provider });
    }
    for (const name of ["lmstudio", "unslothdesktop"].filter(name => providers[name])) {
        const route = providers[name];
        Object.assign(route, { baseURL: endpoint, apiKeyEnv: "LLAMA_DESKTOP_API_KEY" });
        delete route.apiKey;
        if (name === "lmstudio") route.displayName = "LM Studio (Shared)";
        for (const model of route.models) {
            const key = JSON.stringify([name, model.id]);
            if (name === "lmstudio") {
                const policy = policies.get(key);
                const summaryModel = compaction.summarizationModel;
                assert.equal(compaction.summarizationProvider, name, "LM Studio uses one explicit summary route before the next request header exists");
                if (model.id !== summaryModel && (!policy || Object.keys(policy).every(key => ["provider", "model", "summarizationProvider", "summarizationModel"].includes(key)))) {
                    policies.delete(key);
                    continue;
                }
                policies.set(key, { ...policy, provider: name, model: model.id, summarizationProvider: name, summarizationModel: summaryModel });
            } else {
                policies.set(key, { ...policies.get(key), provider: name, model: model.id, summarizationProvider: name, summarizationModel: model.id });
            }
        }
    }
    compaction.modelPolicies = [...policies.values()];
    assert.ok(providers[defaults.provider]?.models.some(model => model.id === defaults.model), "The default model remains registered");
    assert.ok(providers[compaction.summarizationProvider]?.models.some(model => model.id === compaction.summarizationModel), "The summary model remains registered");
    for (const policy of compaction.modelPolicies) {
        assert.ok(providers[policy.provider]?.models.some(model => model.id === policy.model), "Every model policy refers to a registered model");
    }
    return result;
}

module.exports = { canonicalProvider, configureRoutes };
