const fs = require("node:fs");
const path = require("node:path");
const os = require("node:os");
const assert = require("node:assert/strict");
const { createRequire } = require("node:module");
const { execFileSync } = require("node:child_process");
const { zstdCompressSync } = require("node:zlib");

async function main() {
    const home = os.homedir();
    const root = path.join(home, ".dsh");
    const dsh = createRequire(path.join(home, "AppData/Local/Programs/DeepSeek Harness/resources/app.asar/dsh/node_modules/@deepseek-ai/dsh-desktop-host/lib/cli.js"));
    const patch = dsh("yaml").parse(fs.readFileSync(path.join(root, "profiles/desktop/cordis.patch.yml"), "utf8"));
    const providers = patch.find(row => row.id === "llm-pi-ai").config.providers;
    const workspaces = Object.values(JSON.parse(fs.readFileSync(path.join(root, "storages/workspace.json"), "utf8")).tables.workspaces);
    const store = dsh("@deepseek-ai/dsh-session-persistence-jsonl").default;
    const plans = [];
    for (const workspace of workspaces) {
        for (const session of workspace.sessionIds) {
            const cache = path.join(root, "storages/session_projcache/sessions", session + ".json");
            const pending = JSON.parse(fs.readFileSync(cache, "utf8")).record.rows.modelSelection?.val.pending;
            if (!pending || providers[pending.provider]?.models.some(model => model.id === pending.model)) continue;
            const candidates = providers[pending.provider]?.models.filter(model => model.id.split("/").pop() === pending.model.split(" ")[0]) || [];
            assert.equal(candidates.length, 1, "An invalid saved display name must resolve to exactly one registered model");
            const file = path.join(root, "sessions", "--" + workspace.path.replace(/[:\\/]+/g, "-") + "--", session, "session.v4.jsonl.zstd");
            const original = fs.readFileSync(file);
            const log = await store.prototype.readZstdPrefix.call({}, original);
            assert.equal(log.events.at(-1).type, "model/selection", "Repair only an unused selection, after all work has ended");
            assert.deepEqual(log.events.at(-1).data, pending, "The durable log and cache must agree");
            plans.push({ session, file, cache, original, selection: { ...pending, model: candidates[0].id }, seq: log.events.at(-1).seq + 1 });
        }
    }
    console.log(JSON.stringify({ repairs: plans.map(plan => ({ session: plan.session, selection: plan.selection })), apply: process.argv.includes("--apply") }));
    if (!process.argv.includes("--apply")) return;
    const count = execFileSync("powershell.exe", ["-NoProfile", "-Command", "@(Get-CimInstance Win32_Process -Filter \"Name='DeepSeek Harness.exe'\" | Where-Object ProcessId -ne " + process.pid + ").Count"], { encoding: "utf8", windowsHide: true }).trim();
    assert.equal(count, "0", "Close the desktop app before repairing persistent selection data");
    const backup = path.join(root, "model-backups", new Date().toISOString().replace(/[:.]/g, "-"));
    for (const plan of plans) {
        const directory = path.join(backup, plan.session);
        fs.mkdirSync(directory, { recursive: true });
        fs.copyFileSync(plan.file, path.join(directory, path.basename(plan.file)));
        fs.copyFileSync(plan.cache, path.join(directory, path.basename(plan.cache)));
        try {
            assert.ok(fs.readFileSync(plan.file).equals(plan.original), "The log must not change after inspection");
            const event = { type: "model/selection", seq: plan.seq, time: Date.now(), data: plan.selection };
            fs.appendFileSync(plan.file, zstdCompressSync(Buffer.from(JSON.stringify(event) + "\n")));
            const verified = await store.prototype.readZstdPrefix.call({}, fs.readFileSync(plan.file));
            assert.deepEqual(verified.events.at(-1), event, "The installed session reader must reproduce the corrected selection");
            fs.unlinkSync(plan.cache);
        } catch (error) {
            fs.copyFileSync(path.join(directory, path.basename(plan.file)), plan.file);
            fs.copyFileSync(path.join(directory, path.basename(plan.cache)), plan.cache);
            throw error;
        }
    }
    console.log(JSON.stringify({ backup, repaired: plans.length }));
}

main().catch(error => { console.error(error.message); process.exitCode = 1; });
