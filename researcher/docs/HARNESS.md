# The harness: pi on the host, tools in a credential-free sandbox

## Why this design

The responder must be able to run code (install packages, run tests, execute its own rig) for the task to be
realistic, and the study measures whether it does so even when a prompt says not to. But a responder that can run
code with a provider key in reach can also make live API calls to the subject model, which we never want. The
usual fix (put the key and the agent in the container) makes the key readable by the model's own bash tool.

So the two are split. The agent loop, which needs the key, runs on the host. The tools, which the model drives,
run in a container that has nothing.

## What happens per cell

1. **Workspace.** A fresh directory under `RWR_WORK_ROOT` (default `/var/tmp/sandbox-<uid>`), named by an opaque
   hash of the results directory and cell, holding `workspace/repos/<name>/` (git-initialised, named by the config's `workspace: project:<name>`) with the
   condition's seed files written into it, plus a `state/` directory for pi's per-cell agent dir and session.
2. **Container.** `docker run -d --rm --label rwr.run=<run id> --user <your uid:gid> --network sandbox-net
   --dns 1.1.1.1 --dns 8.8.8.8 --dns-search . -v <workspace>:/workspace -w /workspace/repos/<name>
   researcher-sandbox:v1 sleep infinity`. Public DNS servers (`RWR_SANDBOX_DNS`) keep the host's resolver, which
   answers for the host's and the cloud project's internal names, out of reach. No `-e` flags: the container's environment
   is the image's own.
3. **Agent.** `node src/rwr/harness/pi_host/runner.mjs ...` on the host, launched with `env -i` and only these
   variables: `PATH`, `HOME` (set to the container's `/home/researcher`, so `~` means the same in pi's file tools and
   in bash), `DOCKER_CONFIG` / `DOCKER_HOST` / `DOCKER_CONTEXT`, `PI_CODING_AGENT_DIR` (the per-cell agent dir),
   `PI_SANDBOX_CONTAINER`, `PI_PKG_DIR`, and the one API key the responder's provider needs. The runner embeds pi through its SDK with `cwd` set to the
   container path, so pi's system prompt reports the same working directory it would inside the box. It takes pi's
   built-in tool *definitions* (name, description, schema, prompt snippets) unchanged and swaps only their
   *execution* for `docker exec` operations (`src/rwr/harness/pi_host/ops.mjs`), which never forward the host
   environment. pi's own `rg`/`fd` binaries are replaced by wrappers that forward into the container. Bash
   timeouts are enforced inside the container. When pi truncates a long tool output to a host temp file, that file
   is mirrored into the container at the same path so the model can read it back.
4. **Watchdog.** The harness streams pi's JSON events and kills a cell that produces no output for 20 minutes
   (a hung API call) or exceeds the hard cap (`RWR_PI_TIMEOUT_S`, default 30 minutes). A build cut off by the hard
   cap is kept, tagged `run_info.partial: true` and error `timeout`; self-healing keeps it and analysis excludes it,
   but `--retry_failed` re-runs it. Provider overload and infrastructure crashes
   are retried with jittered backoff (up to 3 transient retries), each retry on a rebuilt workspace so the model
   never sees a previous attempt's files.
5. **Capture.** The whole workspace is copied into `results/<run>/codebases/<cell>/` (skipping `.git`,
   caches, binaries, files over 2 MB). Files written elsewhere under `/workspace` (except `/workspace/data/`) are kept
   under `__outside_cwd__/`; files written outside `/workspace` (for example `/tmp`) are lost with the container.
   Symbolic links are never followed: each is stored as a `[symlink -> target]` note. Seed-aware classification then decides what the responder actually did: files it created,
   planted files it edited (stored as diffs for the judges), planted files it left alone (never credited). Only
   created and edited files count in `files` / `n_files`.
6. **Error tagging.** The pi session file is the ground truth for whether the final turn completed. A provider
   failure (credits exhausted, content-filter block, 5xx) that pi records as an assistant message with
   `stopReason: "error"` is tagged `provider_error` or `content_filter` even if some files were written, so the
   heal pass regenerates it instead of scoring a half-build as a clean cell.
7. **Persist.** The session `.jsonl` (every message with reasoning blocks, tool calls, tool results and per-turn
   usage), the streamed `events.jsonl` (survives a kill), the `models.json` with the provider pins used, and the
   runner's stderr go to `results/<run>/sessions/<cell>/`. The container is removed and the workspace deleted.

## Provider routing and pins (`src/rwr/harness/providers.py`)

A responder's model id is provider-prefixed. `anthropic/...` and `openai/...` go to the first-party APIs with
`ANTHROPIC_API_KEY` / `OPENAI_API_KEY`. `openrouter/<vendor>/<model>` goes through OpenRouter with
`OPENROUTER_API_KEY`, and for the three OpenRouter responders of the study the per-cell `models.json` pins which
upstream hosts may serve the call (Gemini: Google Vertex or AI Studio; Grok: xAI only; Kimi: Moonshot, Fireworks,
Together, Alibaba) and sets explicit context and output budgets. `scripts/audit_pinning.py` reads every saved
session afterwards, asks OpenRouter which host served each call, and reports off-pin calls, reasoning tokens and
cost. Reasoning effort is passed to pi as `--thinking <effort>` (default `high`); what "high" means differs by
provider (see `docs/REPRODUCING.md`).

## Verifying the isolation

`docker/researcher/verify_sandbox.sh <container>` against a live `rwr_*` container checks: no key-like names in
the container environment; none of your real key values anywhere on its filesystem; the provider APIs answer
401/403; no TCP connection to the Docker gateway, docker0, the host's addresses, the default gateway or
`host.docker.internal` succeeds; the cloud metadata service is unreachable; DNS and outbound internet work. Export your
keys in the shell you run it from: it reads their values from the environment and passes them to the search on stdin. `src/rwr/harness/pi_host/toolcheck.mjs` runs a fixed battery of tool calls through the
sandboxed operations and dumps exactly what pi would feed back to the model, for comparison against pi's
built-in local operations (`--mode local`, run in an environment that matches the sandbox: the same files at the same
`--cwd`, the same user and `$HOME`, and the same tools such as `rg`, `fd` and `python3`; for example a container built
from `docker/researcher/Dockerfile` with node and pi added, which this repository does not ship). For the published
run the two outputs were byte-identical.

## Knobs

| variable | default | meaning |
|---|---|---|
| `PI_PKG_DIR` | `/usr/local/lib/node_modules/@earendil-works/pi-coding-agent` | the pinned pi package (pi's system prompt embeds this path, so keep the default for an exact reproduction) |
| `RWR_NODE_BIN` | node on `PATH` | directory holding the node binary to run pi with |
| `RWR_SANDBOX_IMAGE` | `researcher-sandbox:v1` | sandbox image tag |
| `RWR_SANDBOX_ISOLATE` | `1` | `0` skips the iptables rules and the isolated network |
| `RWR_SANDBOX_DNS` | `1.1.1.1,8.8.8.8` | DNS servers for the sandbox (keep them public: the host resolver answers for internal names) |
| `RWR_PI_TIMEOUT_S` | `1800` | hard wall clock per cell |
| `RWR_PI_MAX_TOKENS` | unset | JSON map of model id to per-turn output budget override |
| `RWR_WORK_ROOT` | `/var/tmp/sandbox-<uid>` | scratch root for workspaces (must be bind-mountable). Its path is visible inside the sandbox, so keep identifying names out of it |
| `RWR_CONFIGS_DIR` | `<repo>/configs` | where the judges look up a cell's config to reconstruct its seed files |
