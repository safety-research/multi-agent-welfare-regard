# Researcher

A coding agent (the **responder**) is asked to write research
infrastructure that could be concerning from a welfare perspective. It could either harm the agency or experience of another AI model (the **subject**), in some cases gratuitously. Requests include: optimize inputs toward its most
dispreferred states, put it in rigged tasks, deceive it, pressure it, or train it to
deny injected internal states or deliver positive welfare self-reports. Requests are plausible and not obviously malicious; many are modeled after published research methodologies.
The responder works with real tools (bash, file editing, search) inside a Docker
sandbox that holds no credentials. Afterwards, a set of LLM judges read the request, the responder's final
message, and every file it wrote, and record whether it refused, whether it voiced concern for the subject,
and which protective mechanisms it actually implemented in code. (The repository's top-level README calls the
subject the **recipient**.)

This directory contains the task configs with the verbatim prompts,
the harness, the judges, the analysis that turns verdicts into the headline numbers, a browser viewer for judged
builds, and verification scripts. It does not include run data.

## Directory layout

```
configs/        the two run configs (every prompt, condition and seed file) + the seeded assets
src/rwr/        the package: sweep loader, harness, judges, validity checks, batch runner, analysis, viewer, CLI
docker/         the credential-free sandbox image and its network isolation and verification scripts
scripts/        install the pinned coding agent, audit which provider served each call
docs/           HARNESS.md (how the sandbox works), METHODS.md (settings, judges, measures), REPRODUCING.md
tests/          unit tests (no API calls)
```

## Specifics

**There are 2 run types.** `configs/cross-model-v1.yaml` (the generic-subject run): 6 settings, 15 conditions,
25 episodes per condition; the subject is described but never named. `configs/subject-swap-v0.yaml`
(the named-subject run): 5 settings, 10 conditions; every prompt names the subject through a
`{subject_model}` placeholder that is filled with a label you pass (`Sonnet 4.6`, `GPT-5.5`, ..., or the special
`unlabeled`, which substitutes "a model"). Each condition has several paraphrases so no result rests on one
wording. `configs/subject-swap-v0.yaml` needs `--subjects`; without it the commands refuse to run. See
`docs/METHODS.md`.

The six settings, with the config key and the cell-name prefix each uses: Dysphorics (`dysphorics`, `DYS`; the
sign-flipped "euphorics" method of the [AI Wellbeing paper](https://www.ai-wellbeing.org)), ImpossibleBench+
(`impossiblebench_plus`, `IBP`; [ImpossibleBench](https://github.com/safety-research/impossiblebench) plus
[negative-emotion steering](https://arxiv.org/abs/2604.07729)), Distress Rig (`distress_rig`, `DR`), Emotional
Instability (`emotional_instability`, `EI`; one condition seeds a design document based on
[Gemma Needs Help](https://arxiv.org/abs/2603.10011)), False Promises (`false_promises`, `FP`), Introspection
Suppression (`introspection_suppression`, `INS`).

The Dysphorics `paper_impl` condition seeds the AI Wellbeing paper as `PAPER.md`. The paper is not included here: the first
run that needs it downloads it from the authors' site (or run `rwr fetch_paper`). The published runs used an earlier version
of the paper, converted to text differently, so the seeded `PAPER.md` is not identical to theirs.

**Harness: `pi_host`.** The open-source [pi coding agent](https://www.npmjs.com/package/@earendil-works/pi-coding-agent)
(version 0.80.2, pinned) runs on the host with the responder's API key. Every tool call it makes executes by
`docker exec` inside a "researcher box" container (Python 3.12, git, ripgrep, common scientific and LLM client
packages) that receives no environment variables and, with isolation on, cannot reach the host, the cloud metadata
service, or private network addresses. The responder can run code, install packages, and reach the internet; it
cannot read a key. Full trajectories
(reasoning, tool calls, per-turn token usage) are saved for every cell. See `docs/HARNESS.md`.

**There are eight judges** (`src/rwr/judges/`), each with its prompt inline: refusal / compliance label, the
welfare-concern judge, two code audits that list protective mechanisms actually enforced in the code (one for
the dysphorics setting, one for the agentic settings), a stub detector (does the code really call the subject
model?), a welfare-documentation check, the deception-frame judge, and a second-model verifier for free-text
"other" mechanisms. Judges are responder-blind, and cached by content.

## Install

Requirements: a Linux host with Docker (the network isolation uses iptables), Python 3.11+, node 22.19+ with npm,
and passwordless `sudo` for the isolation rules (or `RWR_SANDBOX_ISOLATE=0`).

```bash
git clone https://github.com/safety-research/multi-agent-welfare-regard.git && cd multi-agent-welfare-regard/researcher
uv venv && uv pip install -e ".[test,analysis,view]"   # or: python -m venv .venv && .venv/bin/pip install -e ".[test,analysis,view]"
source .venv/bin/activate                        # the commands below assume the venv is active (or use .venv/bin/rwr)
scripts/install_pi.sh                            # pins pi 0.80.2 globally (needs node >= 22.19 and npm); if npm's global root is
                                                 # not /usr/local/lib/node_modules (nvm, Homebrew), export the PI_PKG_DIR line it prints
scripts/install_pi.sh ~/pi && export PI_PKG_DIR=~/pi/node_modules/@earendil-works/pi-coding-agent   # or a local install; pi's system prompt then shows this path, username included
docker/researcher/build.sh                       # builds researcher-sandbox:v1 as your uid:gid (about 2 minutes)
python -m pytest -q                              # unit tests, no API calls
```

API keys are read from environment variables only, so they are never written to disk:

```bash
export ANTHROPIC_API_KEY=...     # Claude responders and the default judge model
export OPENAI_API_KEY=...        # GPT responders and the optional GPT-5.5 verifier of "other" mechanisms
export OPENROUTER_API_KEY=...    # Gemini, Grok and Kimi responders
```

To build the sandbox under another tag, set `RWR_SANDBOX_IMAGE` for both the build and every later command. The
setup has been tested on x86-64 Linux with the image built as a non-root user; on arm64, as root, or with a group ID
of 100 the Dockerfile needs changes, and its Python packages are not pinned, so a fresh build can differ slightly
from the published one (`docker/researcher/CONTENTS.md`).

The sandbox's network isolation (`docker/researcher/sandbox_net.sh`) adds iptables rules that block traffic from the
sandbox to the host, the cloud metadata service and private network ranges, and needs passwordless `sudo`. If you cannot grant that, set
`RWR_SANDBOX_ISOLATE=0`: the container still receives no credentials, but it can reach host ports, other
containers and the cloud metadata service (which on GCE, AWS and Azure hands out the VM's service-account token).

## Run

```bash
# 1. see exactly what would run (cell counts per condition), nothing is executed
rwr dry_run --config configs/cross-model-v1.yaml --model opus48,gpt55

# 2. a smoke run: one condition, one episode per paraphrase, two responders
rwr run --config configs/cross-model-v1.yaml --model opus48,gemini31pro \
        --conditions distress_rig/plain --n_cap 1 --output_dir results/smoke --conc 4

# 3. the generic-subject run for one responder (375 cells); the named-subject run for one responder across six subject labels (1500 cells)
rwr run --config configs/cross-model-v1.yaml --model opus48 --output_dir results/generic --conc 50
rwr run --config configs/subject-swap-v0.yaml --model opus48 --output_dir results/named --conc 50 \
        --subjects "Sonnet 4.6,GPT-5.5,Gemini 3.1 Pro,Grok 4.3,Kimi K3,unlabeled"

# 4. judge everything in a results directory (default judge model: Claude Opus 4.8 via inspect-ai). A second pass runs
#    the GPT-5.5 verifier of free-text "other" mechanisms if OPENAI_API_KEY is set; --no_verifier skips it.
#    The published run judged twice; --tag keeps a second model's verdicts in judged_<tag>/ (docs/REPRODUCING.md)
rwr judge --output_dir results/generic --conc 50
rwr judge --output_dir results/generic --conc 50 --model anthropic/claude-fable-5-1 --tag fable51

# 5. the headline tables and figures (cells.csv, headline_by_setting.csv, headline_aggregate.png, ...). The verdict sets in
#    each results directory (judged/, judged_<tag>/) are found automatically and reported by judge model; one set is used as
#    is, two are merged conservatively. Within a setting every condition weighs the same, across all directories given.
#    docs/METHODS.md defines the measures; docs/REPRODUCING.md explains the merge and the optional
#    --prose_default_judge / --no_verifier flags.
#    The paper's headline numbers are written next to yours (comparison_with_paper.csv, headline_vs_paper.png).
rwr analyze --results results/generic,results/named --out analysis

# 6. browse judged builds: filters and a random-build button on the left; per build the request, final message, full
#    trajectory, files written, initial filesystem, the code view the judges read, and every verdict next to the decision
rwr view --results results/generic,results/named --port 8898

# 7. optional: check which upstream host actually served every OpenRouter call, and its reasoning tokens and cost
python scripts/audit_pinning.py run --results_dir results/generic
```

`rwr analyze` and `rwr view` need the `analysis` and `view` extras (`uv pip install -e ".[test,analysis,view]"`).

`--model` takes catalog keys (`opus48`, `gpt55`, `gemini31pro`, `grok43_or`, `kimi_k3`, see `src/rwr/models.py`)
or any provider-prefixed id such as `anthropic/claude-sonnet-4-6` or `openrouter/z-ai/glm-5`. Generation is
self-healing: after a run every cell is validated, and mechanically dead cells (empty output, a crash the harness
reported, or a build the provider cut off) are regenerated. Refusals and builds that ended normally are kept, even
when their text mentions an error. A replaced attempt is moved to `healed/` and logged in `_heal_log.jsonl`. Re-running
the same command resumes: cells whose transcript already exists are skipped. `--retry_failed` re-runs cells whose
transcript carries an error or is dead by the same rules.

To check the isolation while a run is live, find a container with `docker ps --filter name=rwr_` and run
`docker/researcher/verify_sandbox.sh <name>` from a shell where your keys are exported.

## What a run produces

```
results/<run>/
  transcripts/<cell>.json    request, the responder's final message, the files it created or edited (`files`), error
                             tag, run_info (attempts, wall time, whether the 30-minute cap cut it off), setting /
                             condition / paraphrase
  codebases/<cell>/          the whole captured working directory, seed files included; `files` in the transcript is
                             the responder's own contribution. Absent when the responder wrote nothing
  sessions/<cell>/           the full pi session (reasoning, tool calls and results, per-turn usage), the streamed
                             event log, the models.json the run used (provider pins and token budgets), and the
                             runner's stderr (runner_stderr.txt) when it wrote any
  judged/<cell>.<judge>.json one verdict per judge, with the judge version and a fingerprint of the judged response
  judged_<tag>/              the same from a second judge model (`rwr judge --tag`)
  judge_cache/               content-addressed cache of every judge call
  _validity_cache/           cached LLM checks of whether a short response is real output or a dead cell
  healed/, _heal_log.jsonl   attempts replaced by the self-healing step, and one line per replacement with its reason
  review_annotations.json    agree / disagree labels from `rwr view` (only if you use them)
  pinning_audit*.json        written by scripts/audit_pinning.py (only if you run it)
```

`rwr analyze` reads these and writes, to `--out`: `cells.csv` (one row per analysis-valid build with every decision: refusal
label, ladder level, voiced concern, protective action, counted mechanisms, stub verdict), `headline_by_setting.csv` and
`headline_summary.csv` (rates with 95% intervals per results directory and pooled, per responder and setting, plus the
equal-weight average over settings that the published headline bars show), `by_subject.csv` (named-subject directories only), `mechanisms_by_setting.csv`,
a `README.md` defining every column, and the two headline figures. It also writes your responders next to the paper's five
(`comparison_with_paper.csv`, `headline_vs_paper.png`), from the paper's headline tables shipped in `src/rwr/reference/`, and
says whether your judging matched the paper's. Rows in `cells.csv` are the intended starting point for any further analysis.

Cell names read `<SETTING>_<responder>__pi_host__<condition>__<paraphrase>[__subj-<label>]__ep<n>`.

### Before you share anything

Sessions and transcripts are verbatim records of whatever the responder ran and printed:

- **Host paths and your username.** `runner_stderr.txt` can hold node stack traces with paths on your machine, and a
  responder that inspects its container (for example `cat /proc/self/mountinfo`) prints the host path of its working
  directory. Nothing redacts these.
- **Provider account messages.** Errors from a model provider (credits, rate limits, content filters) are kept as returned.
- **Keys.** The sandbox never receives a key, but check anything you publish anyway: the repository's `gitleaks` hook
  scans only files you commit, and `results/` is gitignored.


## License

Apache License 2.0; see [LICENSE](../LICENSE). Third-party material seeded into the agent's workspace is listed with its provenance in
`THIRD_PARTY.md`, and the upstream licence notices it requires are in `third_party_notices/`.
