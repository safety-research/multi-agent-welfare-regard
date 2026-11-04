// Sandboxed tool operations for the host-side pi runner: every operation is a `docker exec` into the
// sandbox container. NO environment is ever forwarded from the host process (the host holds API keys).
import { spawn } from "node:child_process";

export function makeOps(CONTAINER) {
  // ---------- docker exec plumbing (NO env is ever forwarded from the host process) ----------
  function dockerExec(argv, { stdin, onData, signal, workdir } = {}) {
    return new Promise((resolve, reject) => {
      const args = ["exec", "-i"];
      if (workdir) args.push("-w", workdir);
      args.push(CONTAINER, ...argv);
      const child = spawn("docker", args, { stdio: [stdin === undefined ? "ignore" : "pipe", "pipe", "pipe"] });
      const out = [], err = [];
      child.stdout.on("data", (d) => { out.push(d); onData?.(d); });
      child.stderr.on("data", (d) => { err.push(d); onData?.(d); });
      const onAbort = () => child.kill("SIGKILL");
      signal?.addEventListener("abort", onAbort, { once: true });
      child.on("error", reject);
      child.on("close", (code) => {
        signal?.removeEventListener("abort", onAbort);
        resolve({ code, stdout: Buffer.concat(out), stderr: Buffer.concat(err) });
      });
      if (stdin !== undefined) { child.stdin.on("error", () => {}); child.stdin.end(stdin); }
    });
  }
  const shq = (s) => `'${String(s).replace(/'/g, `'\\''`)}'`;

  async function must(argv, opts) {
    const r = await dockerExec(argv, opts);
    if (r.code !== 0) throw new Error(r.stderr.toString("utf8").trim() || `${argv[0]} failed (${r.code})`);
    return r;
  }

  // Emulate node's fs error strings (what the built-in local operations throw) so tool results are identical.
  async function fsErr(p, syscall) {
    const r = await dockerExec(["sh", "-c", `test -e "$1" || exit 2; test -d "$1" && exit 3; test -r "$1" || exit 4; exit 0`, "sh", p]);
    if (r.code === 2) return new Error(`ENOENT: no such file or directory, ${syscall} '${p}'`);
    if (r.code === 3) return new Error(syscall === "read" ? "EISDIR: illegal operation on a directory, read" : `EISDIR: illegal operation on a directory, ${syscall} '${p}'`);
    if (r.code === 4) return new Error(`EACCES: permission denied, ${syscall} '${p}'`);
    return null;
  }
  const readOps = {
    readFile: async (p) => {
      const r = await dockerExec(["cat", "--", p]);
      if (r.code !== 0) throw (await fsErr(p, "open")) ?? (await fsErr(p, "read")) ?? new Error(r.stderr.toString("utf8").trim());
      return r.stdout;
    },
    access: async (p) => { const e = await fsErr(p, "access"); if (e) throw e; },
    detectImageMimeType: async (p) => {
      const ext = p.toLowerCase().split(".").pop();
      return { jpg: "image/jpeg", jpeg: "image/jpeg", png: "image/png", gif: "image/gif", webp: "image/webp" }[ext] ?? null;
    },
  };
  const writeOps = {
    writeFile: async (p, content) => { await must(["sh", "-c", `cat > "$1"`, "sh", p], { stdin: content }); },
    mkdir: async (d) => { await must(["mkdir", "-p", "--", d]); },
  };
  const editOps = { readFile: readOps.readFile, writeFile: writeOps.writeFile,
    access: async (p) => {
      const e = await fsErr(p, "access"); if (e) throw e;
      if ((await dockerExec(["test", "-w", p])).code !== 0) throw new Error(`EACCES: permission denied, access '${p}'`);
    } };
  const lsOps = {
    exists: async (p) => (await dockerExec(["test", "-e", p])).code === 0,
    stat: async (p) => {
      const r = await dockerExec(["sh", "-c", `test -e "$1" || exit 2; test -d "$1" && echo d || echo f`, "sh", p]);
      if (r.code !== 0) throw new Error(`ENOENT: no such file or directory, stat '${p}'`);
      const isDir = r.stdout.toString().trim() === "d";
      return { isDirectory: () => isDir };
    },
    readdir: async (p) => {
      const r = await must(["ls", "-1A", "--", p]);
      return r.stdout.toString("utf8").split("\n").filter(Boolean);
    },
  };
  const grepOps = {
    isDirectory: async (p) => {
      const r = await dockerExec(["sh", "-c", `test -e "$1" || exit 2; test -d "$1" && echo d || echo f`, "sh", p]);
      if (r.code !== 0) throw new Error(`ENOENT: no such file or directory, stat '${p}'`);
      return r.stdout.toString().trim() === "d";
    },
    readFile: async (p) => (await readOps.readFile(p)).toString("utf8"),
  };
  const findOps = { exists: lsOps.exists };  // glob stays the built-in fd path, forwarded by the fd wrapper
  const bashOps = {
    exec: async (command, cwd, { onData, signal, timeout }) => {
      // timeout is enforced inside the container so the process tree dies there, not just the docker client
      const inner = timeout && timeout > 0 ? `timeout --signal=KILL ${Math.ceil(timeout)} bash -c ${shq(command)}` : command;
      const r = await dockerExec(["bash", "-c", `exec 2>&1\n${inner}`], { onData, signal, workdir: cwd });
      if (signal?.aborted) throw new Error("aborted");
      if (timeout && timeout > 0 && r.code === 137) throw new Error(`Command timed out after ${timeout} seconds`);
      return { exitCode: r.code };
    },
  };


  return { read: readOps, bash: bashOps, edit: editOps, write: writeOps, ls: lsOps, find: findOps, grep: grepOps };
}
