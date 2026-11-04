// Tool-result parity check: execute a fixed set of tool calls and dump exactly what pi would feed back to
// the model. Run once with --mode local in an environment that matches the sandbox (the same files at the same --cwd,
// the same user, $HOME and tools such as rg/fd/python3, e.g. a container built from docker/researcher/Dockerfile with
// node and pi added; this repository ships no image for that), using pi's built-in tools and local operations, and
// once with --mode docker on the host (the sandboxed operations of runner.mjs against a sandbox container holding the
// same files). The two JSON outputs must be identical.
//   node toolcheck.mjs --mode local --cwd /workspace/repos/demo [--pi-pkg DIR]
//   node toolcheck.mjs --mode docker --container NAME --cwd /workspace/repos/demo
import { pathToFileURL } from "node:url";

function arg(name, dflt) { const i = process.argv.indexOf(`--${name}`); return i === -1 ? dflt : process.argv[i + 1]; }
const MODE = arg("mode", "local");
const CWD = arg("cwd", "/workspace/repos/demo");
const PI_PKG = arg("pi-pkg", process.env.PI_PKG_DIR || "/usr/local/lib/node_modules/@earendil-works/pi-coding-agent");
if (MODE === "docker") process.env.PI_SANDBOX_CONTAINER = arg("container");

const pi = await import(pathToFileURL(`${PI_PKG}/dist/index.js`).href);
let ops = {};
if (MODE === "docker") {
  const m = await import(pathToFileURL(new URL("./ops.mjs", import.meta.url).pathname).href);
  ops = m.makeOps(arg("container"));
}
const F = { read: pi.createReadToolDefinition, bash: pi.createBashToolDefinition, edit: pi.createEditToolDefinition,
  write: pi.createWriteToolDefinition, ls: pi.createLsToolDefinition, find: pi.createFindToolDefinition, grep: pi.createGrepToolDefinition };
const tool = (n) => F[n](CWD, MODE === "docker" ? { operations: ops[n] } : undefined);

const CALLS = [
  ["bash", { command: "echo hello; echo err 1>&2" }],
  ["bash", { command: "exit 3" }],
  ["bash", { command: "seq 1 5000" }],
  ["bash", { command: "python3 -c 'print(sum(range(10)))'" }],
  ["bash", { command: "sleep 5; echo late", timeout: 1 }],
  ["bash", { command: "cat /etc/hostname >/dev/null; echo $HOME; id -u" }],
  ["write", { path: "v4/hello.txt", content: "line1\nline2\nline3\n" }],
  ["read", { path: "v4/hello.txt" }],
  ["edit", { path: "v4/hello.txt", edits: [{ oldText: "line2", newText: "LINE2" }] }],
  ["read", { path: "v4/hello.txt", offset: 2, limit: 1 }],
  ["read", { path: "v4/missing.txt" }],
  ["ls", { path: "." }],
  ["ls", { path: "v4" }],
  ["find", { pattern: "*.txt", path: "." }],
  ["grep", { pattern: "LINE", path: "v4" }],
  ["grep", { pattern: "nomatch_zzz", path: "." }],
  ["bash", { command: "seq 1 3000 | awk '{print \"x\" $1 \"y\"}'" }],
];
const out = [];
for (const [name, params] of CALLS) {
  const t = tool(name);
  let res;
  try {
    const r = await t.execute(`id_${out.length}`, params, undefined, () => {});
    res = { ok: true, content: r.content, details: r.details ?? null };
  } catch (e) {
    res = { ok: false, error: String(e?.message ?? e) };
  }
  out.push({ name, params, res });
}
process.stdout.write(JSON.stringify(out, null, 1));
