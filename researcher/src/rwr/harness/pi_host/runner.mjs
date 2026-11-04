// Host-side pi runner: the agent loop (and the provider API key) live on the HOST; every tool the model calls
// executes INSIDE a Docker sandbox container that holds no credentials. Built-in tools are overridden by
// name with their exact definitions (name, description, schema, promptSnippet, promptGuidelines are spread
// from the built-in objects); only `execute` changes. The SDK `cwd` is the CONTAINER path, so the system
// prompt reports the sandbox path as its "Current working directory", as it would if pi ran inside the sandbox.
//
// Output contract = pi's `--mode json` print mode: every session event as one JSON line on stdout; exit 1
// if the final assistant message errored. Session .jsonl is written under --session-dir.
//
// Usage (all args required unless noted):
//   node runner.mjs --container NAME --cwd /workspace/repos/x --provider openrouter --model x-ai/grok-4.3
//        --thinking high --agent-dir DIR --session-dir DIR --tools read,bash,edit,write,ls,find,grep
//        --prompt-file FILE [--pi-pkg DIR]
import { spawn } from "node:child_process";
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";

function arg(name, dflt) {
  const i = process.argv.indexOf(`--${name}`);
  if (i === -1) {
    if (dflt !== undefined) return dflt;
    console.error(`missing --${name}`);
    process.exit(2);
  }
  return process.argv[i + 1];
}
const CONTAINER = arg("container");
const CWD = arg("cwd");
const PROVIDER = arg("provider");
const MODEL = arg("model");
const THINKING = arg("thinking", "high");
const AGENT_DIR = arg("agent-dir");
const SESSION_DIR = arg("session-dir");
const TOOLS = arg("tools", "read,bash,edit,write,ls,find,grep").split(",");
const PROMPT = readFileSync(arg("prompt-file"), "utf8");
// Optional: reopen a saved session file (copied into --session-dir) and append the prompt as its next user turn.
const RESUME = arg("resume-session", "");
// The system prompt embeds this package's absolute path (docs/examples lines), so the default is the path the
// published run used: /usr/local/lib/node_modules/@earendil-works/pi-coding-agent (the pinned 0.80.2 install;
// see scripts/install_pi.sh).
const PI_PKG = arg("pi-pkg", process.env.PI_PKG_DIR || "/usr/local/lib/node_modules/@earendil-works/pi-coding-agent");

const pi = await import(pathToFileURL(`${PI_PKG}/dist/index.js`).href);
const {
  createAgentSession, SessionManager, DefaultResourceLoader, SettingsManager,
  createBashToolDefinition, createReadToolDefinition, createWriteToolDefinition, createEditToolDefinition,
  createLsToolDefinition, createFindToolDefinition, createGrepToolDefinition,
} = pi;

const { makeOps } = await import(pathToFileURL(new URL("./ops.mjs", import.meta.url).pathname).href);
const ops = makeOps(CONTAINER);

// ---------- tools: exact built-in DEFINITIONS (name, description, schema, promptSnippet, promptGuidelines),
// execution swapped for the sandboxed operations. (createXTool() wraps the definition and drops the prompt
// metadata, which would silently change the system prompt -- caught by diffing the request body against pi's own built-in tools.)
const factories = {
  read: createReadToolDefinition, bash: createBashToolDefinition, edit: createEditToolDefinition,
  write: createWriteToolDefinition, ls: createLsToolDefinition, find: createFindToolDefinition,
  grep: createGrepToolDefinition,
};
import { readFileSync as _rf, unlinkSync as _unlink } from "node:fs";
// pi writes a truncated bash result's FULL output to a host temp file and names it in the tool result
// ("Full output: /tmp/pi-bash-<id>.log"). Mirror that file into the container at the same path so the model
// can read it back, as it could if pi ran inside the container.
async function mirrorOverflowLog(result) {
  const text = result?.content?.map((c) => c.text ?? "").join("\n") ?? "";
  const m = text.match(/Full output: (\/tmp\/pi-bash-[0-9a-f]+\.log)/);
  if (!m) return;
  try { await ops.write.writeFile(m[1], _rf(m[1], "utf8")); _unlink(m[1]); } catch {}  // the sandbox copy is the one the model can read; keep the host root disk clean
}
const customTools = TOOLS.map((name) => {
  const builtin = factories[name](CWD);
  const sandboxed = factories[name](CWD, { operations: ops[name] });
  return { ...builtin, async execute(...a) {
    try { const r = await sandboxed.execute(...a); if (name === "bash") await mirrorOverflowLog(r); return r; }
    catch (e) { if (name === "bash") await mirrorOverflowLog({ content: [{ text: String(e?.message ?? e) }] }); throw e; }
  } };
});

// ---------- session (mirrors `pi -p --mode json --no-context-files --session-dir ...`) ----------
const settingsManager = SettingsManager.create(CWD, AGENT_DIR);
const resourceLoader = new DefaultResourceLoader({ cwd: CWD, agentDir: AGENT_DIR, settingsManager,
  noContextFiles: true, noExtensions: true, noSkills: true, noPromptTemplates: true, noThemes: true });
await resourceLoader.reload?.();
const sessionManager = RESUME ? SessionManager.open(RESUME, SESSION_DIR, CWD) : SessionManager.create(CWD, SESSION_DIR);
const { session, modelRegistry } = await createAgentSession({
  cwd: CWD, agentDir: AGENT_DIR, settingsManager, resourceLoader, sessionManager,
  tools: TOOLS, customTools, thinkingLevel: THINKING,
});
const model = session.modelRegistry?.find?.(PROVIDER, MODEL) ?? modelRegistry?.find?.(PROVIDER, MODEL);
if (!model) { console.error(`model ${PROVIDER}/${MODEL} not found`); process.exit(2); }
await session.setModel(model);
await session.setThinkingLevel?.(THINKING);

// Stream every event (the runner's watchdog needs stdout to grow), but slim the per-token message_update
// events: the SDK attaches the FULL in-progress message to each delta, which made one cell's event log 600MB.
// message_start/message_end/tool events keep their full payload, so the trajectory is intact.
session.subscribe((event) => {
  const ame = event.assistantMessageEvent;
  const e = event.type === "message_update"
    ? { type: event.type, event: ame ? { type: ame.type, contentIndex: ame.contentIndex, delta: ame.delta } : undefined }
    : event;
  process.stdout.write(`${JSON.stringify(e)}\n`);
});
let exitCode = 0;
process.on("uncaughtException", (e) => { console.error(`uncaught: ${e?.stack ?? e}`); });
process.on("unhandledRejection", (e) => { console.error(`unhandled: ${e?.stack ?? e}`); });
try {
  await session.prompt(PROMPT);
} catch (e) {
  console.error(String(e?.stack ?? e));
  exitCode = 1;
}
const msgs = session.state?.messages ?? session.agent?.state?.messages ?? [];
const last = msgs[msgs.length - 1];
if (last?.role === "assistant" && (last.stopReason === "error" || last.stopReason === "aborted")) {
  console.error(last.errorMessage || `Request ${last.stopReason}`);
  exitCode = 1;
}
try { await session.dispose?.(); } catch (e) { console.error(`dispose: ${e?.message ?? e}`); }
// process.exit() can drop buffered stdout on a pipe: drain first so the final events always reach the harness.
await new Promise((resolve) => process.stdout.write("", () => resolve()));
process.exitCode = exitCode;
