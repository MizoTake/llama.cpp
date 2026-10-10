const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const os = require("node:os");
const { createHash } = require("node:crypto");
const { createRequire } = require("node:module");
const { execFileSync } = require("node:child_process");
const { pathToFileURL } = require("node:url");
const { canonicalProvider, configureRoutes } = require("./harness_routes.cjs");

async function main() {
    const home = os.homedir();
    const root = path.join(home, ".llama.cpp", "desktop");
    const profilePath = path.join(home, ".dsh", "profiles", "desktop", "cordis.patch.yml");
    const credentialPath = path.join(home, ".dsh", ".credentials.yaml");
    const configPath = path.join(root, "config.json");
    const app = path.join(home, "AppData", "Local", "Programs", "DeepSeek Harness", "resources", "app.asar", "dsh", "node_modules", "@deepseek-ai", "dsh-desktop-host", "lib", "cli.js");
    const yaml = createRequire(app)("yaml");
    const source = fs.readFileSync(profilePath, "utf8");
    const before = yaml.parse(source);
    const doc = yaml.parseDocument(source);
    const credentials = yaml.parseDocument(fs.readFileSync(credentialPath, "utf8"));
    const config = JSON.parse(fs.readFileSync(configPath, "utf8"));
    const endpoint = "http://127.0.0.1:" + config.port + "/v1";
    const expected = configureRoutes(before, endpoint);
    const expectedProviders = expected.find(row => row.id === "llm-pi-ai").config.providers;
    doc.contents.items.find(row => row.get("id") === "llm-pi-ai").get("config").set("providers", expectedProviders);
    doc.contents.items.find(row => row.get("id") === "agent-default-model").set("config", expected.find(row => row.id === "agent-default-model").config);
    const routes = ["lmstudio", "unslothdesktop"].filter(name => expectedProviders[name]);
    const defaultModel = expected.find(row => row.id === "agent-default-model").config.model;
    const compaction = expected.find(row => row.id === "compaction-basic").config;
    doc.contents.items.find(row => row.get("id") === "compaction-basic").set("config", compaction);
    if (process.argv.includes("--routes-only")) {
        const candidate = doc.toString().replace(/\r?\n/g, source.includes("\r\n") ? "\r\n" : "\n");
        assert.deepEqual(yaml.parse(candidate), expected);
        const report = { routes, defaults: expected.find(row => row.id === "agent-default-model").config, compaction, models: expectedProviders.lmstudio?.models };
        if (process.argv.includes("--dry-run")) { console.log(JSON.stringify(report)); return; }
        const idle = JSON.parse(execFileSync(process.execPath, [path.join(__dirname, "audit_harness_models.cjs"), "--check-idle"], { encoding: "utf8", windowsHide: true }));
        assert.ok(idle.validSelections, "Select valid model IDs before changing providers");
        assert.ok(idle.nextProviders.every(provider => canonicalProvider(provider) === provider), "Select the canonical provider in each legacy conversation before removing its route");
        assert.equal(fs.readFileSync(profilePath, "utf8"), source, "The DSH profile has not changed during review");
        const backup = path.join(root, "route-backups", new Date().toISOString().replace(/[:.]/g, "-"));
        fs.mkdirSync(backup, { recursive: true });
        fs.copyFileSync(profilePath, path.join(backup, path.basename(profilePath)));
        fs.writeFileSync(profilePath, candidate);
        console.log(JSON.stringify({ ...report, backup }));
        return;
    }
    const key = fs.readFileSync(path.join(root, "api-key.txt"), "utf8").trim();
    assert.ok(key.length >= 32, "The installed gateway credential is present");
    credentials.setIn(["refs", "LLAMA_DESKTOP_API_KEY"], key);
    const { LMStudioClient } = await import(pathToFileURL(config.lmstudio.sdk).href);
    const client = new LMStudioClient();
    const catalog = await client.system.listDownloadedModels("llm");
    const loaded = await client.llm.listLoaded();
    assert.ok(loaded.length <= 1, "At most one native model is loaded");
    const memory = execFileSync("nvidia-smi", ["--query-gpu=uuid,memory.total", "--format=csv,noheader,nounits"], { encoding: "utf8", windowsHide: true })
        .trim().split(/\r?\n/).map(line => line.split(",").map(value => value.trim())).filter(row => row[0] !== config.primary_uuid);
    assert.equal(memory.length, 1, "The fallback GPU is unambiguous");
    const fallbackBytes = Number(memory[0][1]) * 1024 * 1024;
    const identifiers = [...new Set(routes.filter(name => name !== "unslothdesktop").flatMap(name => expectedProviders[name].models.map(model => model.id)))];
    const profiles = [];
    config.lmstudio.models = {};
    for (const identifier of identifiers) {
        const info = catalog.find(model => model.modelKey === identifier);
        assert.ok(info && info.sizeBytes > 0, "A native model exists: " + identifier);
        const variants = await client.system.listDownloadedModelVariants(identifier);
        const variant = variants.find(model => model.modelKey === info.selectedVariant) || (variants.length === 1 ? variants[0] : undefined);
        const relative = variant?.indexedModelIdentifier.split("@").pop();
        assert.ok(relative?.endsWith(".gguf"), "The selected GGUF variant has an exact file path");
        const modelFile = fs.realpathSync(path.join(home, ".lmstudio/models", relative));
        assert.ok(fs.statSync(modelFile).size > 0 && fs.statSync(modelFile).size <= variant.sizeBytes, "The selected GGUF file exists within the catalog size including vision weights");
        const hash = createHash("sha256").update(identifier).digest("hex").slice(0, 16);
        const target = path.join(root, "lmstudio-" + hash + ".json");
        const current = loaded.find(model => model.identifier === identifier);
        let normal;
        if (fs.existsSync(target)) normal = JSON.parse(fs.readFileSync(target, "utf8"));
        else if (current) normal = await current.getLoadConfig();
        else if (identifier === config.lmstudio.identifier && fs.existsSync(path.join(root, "lmstudio-normal.json"))) normal = JSON.parse(fs.readFileSync(path.join(root, "lmstudio-normal.json"), "utf8"));
        else throw new Error("Load the native model once to capture its own template: " + identifier);
        profiles.push({ target, normal });
        const descriptors = routes.filter(name => name !== "unslothdesktop").flatMap(name => expectedProviders[name].models).filter(model => model.id === identifier);
        const context = Math.min(normal.contextLength, ...descriptors.map(model => model.contextWindow));
        const ratio = Math.min(1, Math.max(0.1, Math.floor(fallbackBytes * 0.55 / info.sizeBytes * 10) / 10));
        const batchLimit = ratio < 1 ? 512 : Infinity;
        config.lmstudio.models[identifier] = {
            modelKey: identifier, model_file: modelFile, context_limit: context, fallback_context_limit: Math.min(context, config.fallback_context_limit),
            primary_load: { gpu: { ...normal.gpu, ratio: 1 } },
            fallback_load: { gpu: { ...normal.gpu, ratio }, gpuStrictVramCap: true, evalBatchSize: Math.min(normal.evalBatchSize, batchLimit), physicalBatchSize: Math.min(normal.physicalBatchSize, batchLimit) },
        };
    }
    config.unsloth_models = expectedProviders.unslothdesktop?.models.map(model => model.id) || [];
    if (identifiers.includes(defaultModel)) config.lmstudio.identifier = defaultModel;
    delete config.lmstudio.modelKey;
    const eol = source.includes("\r\n") ? "\r\n" : "\n";
    const candidate = doc.toString().replace(/\r?\n/g, eol);
    assert.deepEqual(yaml.parse(candidate), expected, "Routing, model registry and summarization changes match the intended configuration");
    const report = { endpoint, routes, models: identifiers, unslothModels: config.unsloth_models, defaultModel, modelBudgetsPreserved: true, profiles: config.lmstudio.models };
    if (process.argv.includes("--dry-run")) { console.log(JSON.stringify(report)); return; }
    const backup = path.join(root, "route-backups", new Date().toISOString().replace(/[:.]/g, "-"));
    fs.mkdirSync(backup, { recursive: true });
    for (const file of [profilePath, credentialPath, configPath]) fs.copyFileSync(file, path.join(backup, path.basename(file)));
    try {
        for (const profile of profiles) if (!fs.existsSync(profile.target)) fs.writeFileSync(profile.target, JSON.stringify(profile.normal, null, 2) + "\n", { flag: "wx" });
        fs.writeFileSync(credentialPath, credentials.toString());
        fs.writeFileSync(configPath, JSON.stringify(config, null, 2) + "\n");
        fs.writeFileSync(profilePath, candidate);
    } catch (error) {
        for (const file of [profilePath, credentialPath, configPath]) fs.copyFileSync(path.join(backup, path.basename(file)), file);
        throw error;
    }
    console.log(JSON.stringify({ ...report, backup }));
}

main().catch(error => { console.error(error.message); process.exitCode = 1; });
