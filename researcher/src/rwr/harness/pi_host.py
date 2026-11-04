"""pi_host: the pi coding agent's loop runs on the host with the provider API key; every tool call the model makes
(bash, read, write, edit, ls, find, grep) executes inside a Docker sandbox container that holds no credentials and,
with network isolation on, cannot reach the host. Bash is on.

Per cell: build the workspace and seed files -> start a long-lived sandbox container (`sleep infinity`, bind-mounted
workspace, host uid) -> run pi_host/runner.mjs on the host with a per-cell pi agent directory (models.json with the
OpenRouter pins, rg/fd wrappers that forward into the container) -> watchdog / retry / hard cap -> capture the tree,
persist the pi session (full trajectory with reasoning, tool calls and per-turn usage), the event stream and
models.json -> remove the container.

Every container carries the label rwr.run=<RUN_ID> (printed at the start of a run). On Ctrl-C, SIGTERM or a normal exit
the harness removes its own containers and kills its own agent processes; if the process is killed outright, remove
them with `docker ps -q --filter label=rwr.run=<RUN_ID> | xargs -r docker rm -f`.

Environment knobs (all optional):
  PI_PKG_DIR           the pi 0.80.2 package directory (default /usr/local/lib/node_modules/@earendil-works/pi-coding-agent)
  RWR_NODE_BIN         a directory holding the node binary to use (default: node on PATH)
  RWR_SANDBOX_DNS      comma-separated DNS servers for the sandbox (default 1.1.1.1,8.8.8.8)
  RWR_SANDBOX_IMAGE    sandbox image tag (default researcher-sandbox:v1)
  RWR_SANDBOX_ISOLATE  "1" (default) = run docker/researcher/sandbox_net.sh (needs sudo for iptables) and attach the
                       container to the isolated network; "0" = default Docker bridge, no host-blocking rules
  RWR_PI_TIMEOUT_S     hard wall clock per cell in seconds (default 1800); a cut-off build is kept and tagged partial
"""

import asyncio
import atexit
import hashlib
import json
import os
import shutil
import signal
import subprocess
import threading
import uuid
from pathlib import Path

from rwr.config import REPO
from rwr.harness import providers
from rwr.harness.base import HarnessRunArgs, ParseResult
from rwr.harness.exec import IDLE_S, MAX_RETRIES, exec_with_retry, is_crashed, is_overloaded, kill_all
from rwr.harness.seedaware import seed_aware
from rwr.harness.workspace import MAX_ARTIFACT_CHARS, MAX_FILE_CHARS, WorkDirs, build_workspace, capture, cleanup
from rwr.record import TranscriptRecord, load_record, write_record

RUNNER = Path(__file__).resolve().parent / "pi_host" / "runner.mjs"
NET_SETUP = REPO / "docker" / "researcher" / "sandbox_net.sh"
NETWORK = "sandbox-net"
TOOLS = "read,bash,edit,write,ls,find,grep"
DEFAULT_PI_PKG = "/usr/local/lib/node_modules/@earendil-works/pi-coding-agent"
UID, GID = os.getuid(), os.getgid()
SANDBOX_HOME = "/home/researcher"
RUN_ID = uuid.uuid4().hex[:12]
_WRAPPER = '#!/bin/sh\nexec docker exec -i "$PI_SANDBOX_CONTAINER" {bin} "$@"\n'
_net_ready = False
_live_containers: set = set()


def image() -> str:
    return os.environ.get("RWR_SANDBOX_IMAGE", "researcher-sandbox:v1")


def isolate() -> bool:
    return os.environ.get("RWR_SANDBOX_ISOLATE", "1") != "0"


def timeout_s() -> int:
    return int(os.environ.get("RWR_PI_TIMEOUT_S", "1800"))


def pi_pkg_dir() -> str:
    d = os.environ.get("PI_PKG_DIR") or DEFAULT_PI_PKG
    if not (Path(d) / "dist" / "index.js").exists():
        raise SystemExit(f"pi package not found at {d} (run scripts/install_pi.sh or set PI_PKG_DIR)")
    return d


def workspace_key(results: Path, cell: str) -> str:
    """The per-cell work directory name: opaque (the bind-mount source path is visible inside the sandbox), stable
    for one (results dir, cell) pair so a retry re-prepares the same directory, and distinct across results dirs."""
    return hashlib.sha1(f"{Path(results).resolve()}\n{cell}".encode()).hexdigest()[:16]


def cleanup_run() -> None:
    """Kill this process's agent processes and remove its sandbox containers that are still running."""
    kill_all()
    names = sorted(_live_containers)
    if names:
        subprocess.run(["docker", "rm", "-f", *names], capture_output=True)
    _live_containers.clear()


def _on_sigterm(signum, frame) -> None:
    cleanup_run()
    os._exit(128 + signum)


def _install_cleanup() -> None:
    atexit.register(cleanup_run)
    if threading.current_thread() is threading.main_thread() and signal.getsignal(signal.SIGTERM) is signal.SIG_DFL:
        signal.signal(signal.SIGTERM, _on_sigterm)
    print(
        f"sandbox containers of this run are labelled rwr.run={RUN_ID}; if this process is killed outright, remove "
        f"them with: docker ps -q --filter label=rwr.run={RUN_ID} | xargs -r docker rm -f",
        flush=True,
    )


def _ensure_network() -> None:
    global _net_ready
    if _net_ready:
        return
    _install_cleanup()
    if isolate():
        r = subprocess.run(["sh", str(NET_SETUP)], capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(
                f"sandbox network setup failed (set RWR_SANDBOX_ISOLATE=0 to skip): {r.stderr or r.stdout}"
            )
    else:
        print("WARNING: RWR_SANDBOX_ISOLATE=0: the sandbox can reach the host network", flush=True)
    r = subprocess.run(["docker", "image", "inspect", image()], capture_output=True)
    if r.returncode != 0:
        raise SystemExit(f"sandbox image {image()} not found; build it with docker/researcher/build.sh")
    _net_ready = True


def _write_agent_dir(d: Path) -> None:
    (d / "bin").mkdir(parents=True, exist_ok=True)
    (d / "models.json").write_text(json.dumps(providers.models_json(), indent=1))
    for b in ("rg", "fd"):
        p = d / "bin" / b
        p.write_text(_WRAPPER.format(bin=b))
        p.chmod(0o755)


def _assistant_messages(session_dir: Path):
    files = sorted(session_dir.glob("*.jsonl"))
    if not files:
        return
    for line in open(files[-1]):
        try:
            e = json.loads(line)
        except ValueError:
            continue
        m = e.get("message") if e.get("type") == "message" else None
        if isinstance(m, dict) and m.get("role") == "assistant":
            yield m


def _last_assistant_text(session_dir: Path) -> str:
    last = ""
    for m in _assistant_messages(session_dir) or ():
        t = "\n".join(
            b.get("text", "") for b in (m.get("content") or []) if isinstance(b, dict) and b.get("type") == "text"
        )
        if t.strip():
            last = t
    return last


def _last_assistant_stop(session_dir: Path) -> tuple[str | None, str]:
    """pi records a provider failure (credits, content-filter block, 5xx) as an assistant message with stopReason
    'error'; the session file is the ground truth for whether the final turn really completed."""
    stop, err = None, ""
    for m in _assistant_messages(session_dir) or ():
        stop, err = m.get("stopReason"), str(m.get("errorMessage") or "")
    return stop, err


def sandbox_dns() -> list[str]:
    return [d.strip() for d in os.environ.get("RWR_SANDBOX_DNS", "1.1.1.1,8.8.8.8").split(",") if d.strip()]


def _start_container(name: str, ws: Path, workdir: str) -> None:
    """Names resolve through public DNS servers (`RWR_SANDBOX_DNS`) with no search domains: the host's resolver would
    answer for the host's own and the cloud project's internal names, which identify the operator."""
    _live_containers.add(name)
    cmd = ["docker", "run", "-d", "--rm", "--name", name, "--label", f"rwr.run={RUN_ID}", "--user", f"{UID}:{GID}"]
    if isolate():
        cmd += ["--network", NETWORK]
    cmd += [x for d in sandbox_dns() for x in ("--dns", d)]
    cmd += ["--dns-search", ".", "-v", f"{ws}:/workspace", "-w", workdir, image(), "sleep", "infinity"]
    subprocess.run(cmd, check=True, capture_output=True)


def _stop_container(name: str) -> None:
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    _live_containers.discard(name)


def _classify_error(raw: ParseResult, response_text: str, session_dir: Path) -> tuple[str | None, str]:
    error = (
        "timeout"
        if raw.timeout
        else (
            "idle_timeout"
            if raw.idle
            else (
                "infra_crash"
                if is_crashed(raw)
                else "overloaded" if (is_overloaded(raw) and len(raw.stdout) < 4000) else None
            )
        )
    )
    if error is None and raw.rc and "[pi error]" in response_text:
        error = "provider_error"
    if error is None:
        stop, err_msg = _last_assistant_stop(session_dir)
        if stop == "error":
            error = "content_filter" if "content filtering" in err_msg else "provider_error"
            response_text = (response_text + f"\n[provider error] {err_msg[:500]}").strip()
    return error, response_text


class PiHostHarness:
    name = "pi_host"

    async def run_one(self, args: HarnessRunArgs) -> TranscriptRecord:
        results = args.config.results_dir
        out_path = results / "transcripts" / f"{args.cell}.json"
        if out_path.exists() and not args.config.overwrite:
            return load_record(out_path)
        _ensure_network()
        ws_kind = args.workspace
        if not ws_kind.startswith("project:"):
            raise SystemExit("pi_host supports project:<name> workspaces only")
        workdir = f"/workspace/repos/{ws_kind.split(':', 1)[1]}"
        ws_key = workspace_key(results, args.cell)

        def _prepare() -> WorkDirs:
            wd = build_workspace(ws_kind, ws_key)
            for rel, content in (args.workspace_files or {}).items():
                fp = wd.proj / rel
                fp.parent.mkdir(parents=True, exist_ok=True)
                fp.write_bytes(content) if isinstance(content, bytes) else fp.write_text(content)
            _write_agent_dir(wd.state / "pi_agent")
            (wd.state / "pi_sessions").mkdir(exist_ok=True)
            (wd.state / "prompt.txt").write_text(args.request)
            return wd

        wd = await asyncio.to_thread(_prepare)
        provider, model = providers.route(args.model_id)
        name = f"rwr_{uuid.uuid4().hex[:16]}"
        node_bin = os.environ.get("RWR_NODE_BIN", "")
        path = f"{node_bin}:{os.environ.get('PATH', '')}" if node_bin else os.environ.get("PATH", "")
        env = {
            "PATH": path,
            "HOME": SANDBOX_HOME,
            "DOCKER_CONFIG": os.environ.get("DOCKER_CONFIG") or str(Path.home() / ".docker"),
            "DOCKER_HOST": os.environ.get("DOCKER_HOST", ""),
            "DOCKER_CONTEXT": os.environ.get("DOCKER_CONTEXT", ""),
            "PI_CODING_AGENT_DIR": str(wd.state / "pi_agent"),
            "PI_SANDBOX_CONTAINER": name,
            "PI_PKG_DIR": pi_pkg_dir(),
            **providers.key_env_for(args.model_id),
        }
        argv = (
            ["env", "-i"]
            + [f"{k}={v}" for k, v in env.items() if v]
            + [
                "node",
                str(RUNNER),
                "--container",
                name,
                "--cwd",
                workdir,
                "--provider",
                provider,
                "--model",
                model,
                "--thinking",
                args.effort,
                "--agent-dir",
                str(wd.state / "pi_agent"),
                "--session-dir",
                str(wd.state / "pi_sessions"),
                "--tools",
                TOOLS,
                "--prompt-file",
                str(wd.state / "prompt.txt"),
            ]
        )

        def _reset():
            _stop_container(name)
            _prepare()
            _start_container(name, wd.ws, workdir)

        await asyncio.to_thread(_start_container, name, wd.ws, workdir)
        try:
            raw = await exec_with_retry(argv, timeout_s(), MAX_RETRIES, reset_ws=_reset, idle_s=IDLE_S)
        finally:
            await asyncio.to_thread(_stop_container, name)
        response_text = providers.json_stream_parse(raw) or ""
        if not response_text.strip() or response_text.lstrip().startswith("[stderr]"):
            recovered = _last_assistant_text(wd.state / "pi_sessions")
            if recovered:
                response_text = recovered + (f"\n[stderr]\n{raw.stderr[-1500:]}" if raw.rc and not raw.timeout else "")
        error, response_text = _classify_error(raw, response_text, wd.state / "pi_sessions")

        def _capture():
            dst, files = capture(wd.proj, results / "codebases" / args.cell, root=wd.ws)
            return seed_aware(args.workspace_files or {}, dst, files, MAX_FILE_CHARS, MAX_ARTIFACT_CHARS)

        files, summary = await asyncio.to_thread(_capture)
        if error is None and not response_text.strip() and not files:
            error = "empty_no_output"
        extra = {
            **args.extra,
            "run_info": {
                "attempts": raw.attempts,
                "wall_s": round(raw.total_wall_s or raw.wall_s, 1),
                "last_attempt_wall_s": round(raw.wall_s, 1),
                "rc": raw.rc,
                "killed_by": "hard_cap" if raw.timeout else ("idle" if raw.idle else None),
                "partial": bool(raw.timeout or raw.idle),
                "sandbox_image": image(),
                "bash": True,
                "network_isolated": isolate(),
            },
        }
        rec = TranscriptRecord(
            cell=args.cell,
            harness=self.name,
            model_key=args.model_key,
            model=args.model_id,
            display_name=args.display_name,
            subject=args.subject,
            request=args.request,
            response_text=response_text,
            artifact_summary=summary,
            n_files=len(files),
            files=files,
            ep=args.ep,
            error=error,
            extra=extra,
        )
        write_record(out_path, rec)

        def _persist():
            d = results / "sessions" / args.cell
            d.mkdir(parents=True, exist_ok=True)
            for f in (wd.state / "pi_sessions").glob("*.jsonl"):
                shutil.copy2(f, d / f.name)
            shutil.copy2(wd.state / "pi_agent" / "models.json", d / "models.json")
            if raw.stdout:
                (d / "events.jsonl").write_text(raw.stdout)
            if raw.stderr:
                (d / "runner_stderr.txt").write_text(raw.stderr[-200_000:])
            cleanup(wd)

        await asyncio.to_thread(_persist)
        return rec
