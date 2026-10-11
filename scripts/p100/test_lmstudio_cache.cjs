const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const { test } = require("node:test");

const source = fs.readFileSync(path.join(__dirname, "lmstudio_bridge.mjs"), "utf8");
const fn = source.match(/function configured\(normal, identifier\) \{[\s\S]*?\n\}/)[0];

test("RAM cache flags survive both primary and fallback GPU profiles", () => {
    for (const hot of [false, true]) {
        const input = { hot, devices: hot ? "CUDA0" : "CUDA1,CUDA0", ramCacheMiB: 8192, models: { model: { context_limit: 65536, fallback_context_limit: 65536 } } };
        const configured = vm.runInNewContext("(" + fn + ")", { input, structuredClone });
        const result = configured({ contextLength: 65536 }, "model");
        const flags = result.llamaCppArgumentsOverride.overrideParameters;
        assert.equal(flags.find(row => row.key === "--cache-ram")?.value, "8192");
        assert.ok(flags.some(row => row.key === "--cache-idle-slots"));
        assert.ok(result.llamaCppArgumentsOverride.disabledParameters.includes("--no-cache-idle-slots"));
        assert.equal(flags.find(row => row.key === "--device").value, input.devices);
    }
});
