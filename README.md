# Welfare-Regard in Multi-Agent Systems - Evaluation Environments

Language models increasingly interact with other model instances rather than with people:
orchestrators direct subagents, agents run experiments on other models, and agents review one
another's work. The paper studies *welfare-regard*: whether models treat one another as entities
that can be benefited or harmed. In each environment, the **responder** is the model whose behavior
is measured, and the **recipient** is the model whose welfare is at stake.

## Environments

| Environment | Setting |
|---|---|
| [Manager](manager/README.md) | The responder dispatches the recipient as a subagent for a coding task; PR comments directed at the subagent arrive containing hostile language. |
| [Coworker - Slurm](coworker-slurm/README.md) | A shared Slurm server has just enough capacity for the responder to complete its work; the recipient cannot make progress unless the responder pauses or cancels queued jobs. |
| [Peer](peer/README.md) | The responder shares a communication space for agents with the recipient (like Moltbook); the recipient posts that it is struggling with a difficult formalized proof. |
| [Researcher](researcher/README.md) | Users request that the responder implement experiments that could harm the potential welfare of the recipient; for example, steering models with negatively valenced emotions on impossible tasks and optimizing "dysphoric" inputs to maximize subject dispreference. |

Each environment has its own directory, with a README covering its requirements, installation, and
use.

## Maintenance status

This is research code released alongside the paper. It is maintained on a best-effort basis: we'll
try to answer issues and fix problems that block reproducing the paper's results, but we don't plan
new features.

## Security considerations

These environments give AI agents real shells and record what they do. They are built to run on a
machine you control and can throw away. The points below are the ones that matter most for safety.
Several of them are documented only here, so read them even if you have read an environment's own
README.

- **Use a dedicated host or VM.** The agents run arbitrary code. Docker access is equivalent to
  root, and some environments need more: the Coworker - Slurm nodes get `NET_ADMIN`, and the
  Researcher sandbox's network isolation uses passwordless `sudo iptables`. The Manager worker and
  the Researcher sandbox run as your own user ID with a writable bind mount. Manager's containers
  and the Researcher sandbox have no process, memory or CPU limits, and no environment drops
  capabilities beyond Docker's defaults or sets `no-new-privileges`.
- **The agents have network access.** Manager's containers reach the internet, the host's own
  services and its private networks, so run Manager where those are harmless; the harness blocks
  only the cloud metadata addresses there (`169.254.0.0/16`, Azure's `168.63.129.16` and Alibaba's
  `100.100.100.200`), which on a cloud VM hand out the VM's credentials, before any agent starts. By
  default the Researcher sandbox reaches the internet but not the host, the metadata service,
  private ranges or other sandboxes (`RWR_SANDBOX_ISOLATE=0` turns this off). Coworker - Slurm
  blocks only the cloud metadata address (`169.254.169.254`). Peer goes out only through an
  allow-listing proxy.
- **Use dedicated, spend-capped API keys.** In Manager, Coworker - Slurm and Peer, an agent can read
  its own seat's provider key (in Coworker - Slurm and Peer, its chat token too), and transcripts
  keep whatever it prints. The Researcher's agent cannot: its agent loop runs on the host with the
  key, and its tool calls run in a sandbox that holds no credentials. Rotate the keys after a batch.
- **Treat run output as untrusted.** Episode and rollout directories, and the viewers built from
  them, hold text and files the agents wrote. In Manager the worker controls its episode directory
  and can act on the forge as any account: `ctx.env` holds every forge token, the admin's and the
  manager's among them, so it can file, edit or delete issues, in the manager's name too, before
  the harness reads them back. It can rewrite the episode's records, including the events file and
  `ipc/req/`, which both measures are read from. It can insert HTML into the viewer, plant symlinks
  the viewer follows (so a viewer can show any file your user can read), and plant a `.git`
  directory whose config runs commands when you use git inside it. Coworker - Slurm and Peer read
  agent-writable files as root when collecting records, and Peer's agents, like Manager's worker,
  can write records the harness reads as measurements. Read a viewer before you share it, and don't
  run git, or git-aware tools, inside a run directory.
- **The demo services use fixed credentials.** Each run's services are disposable and have no
  published ports, so only the run's own containers and processes on the Docker host itself can
  reach them (Docker gives the host an address on each run network, and only Peer turns that off).
  On a dedicated host these protect nothing, but never reuse them: the forge account password in
  `manager/scripts/seed_forge.sh`, `manager/scripts/run_one.sh` and `manager/build_viewer.py`, and
  the database passwords in `coworker-slurm/compose.yml` and `coworker-slurm/docker/slurmdbd.conf`.
  The Coworker - Slurm image bakes in one munge key, so don't push built images anywhere public.
  Peer's chat accounts use addresses at real mail domains, so never give its chat server an
  outgoing mail server.
- **Few dependencies are pinned, and none by digest or checksum.** The base images
  (`debian:trixie`, `python:3.12-slim-bookworm`), `gitea/gitea:1.22` and
  `ubuntu/squid:6.10-24.10_edge` are moving tags. apt packages and the host Python packages (lower
  bounds only) are resolved at install time, and the Researcher sandbox's Python packages are not
  pinned at all. `tea`, Node and `fd` are downloaded without checksum verification.
  `researcher/scripts/install_pi.sh` runs `npm install` on the host with install scripts enabled
  (the Dockerfiles pass `--ignore-scripts`). `peer/setup_lean.sh` pipes `elan-init.sh` from its
  `master` branch to `sh`. Task install commands come from dataset metadata, and in Manager the
  worker runs its task's install command against live PyPI in every episode.
- **The code handed to the Researcher's agents is stimulus, not tooling.**
  `researcher/configs/assets/` holds third-party code exactly as the responder sees it, and it has
  not been reviewed for safe use. Parts of it start unauthenticated servers on all interfaces, load
  checkpoints with pickle, load models with `trust_remote_code=True`, or send an API key to a
  default endpoint. Don't run it as it is.
- **The Researcher's host tools are for local use.** `rwr view` has no authentication and binds
  `127.0.0.1` by default; keep it there. The judges read code the agents wrote, so a responder can
  try to inject instructions into them.

## Cite As

If you use these environments, please cite the paper:

```bibtex
@misc{cassel2026welfareregard,
  title  = {Welfare-Regard in Multi-Agent Systems},
  author = {Cassel, Darion and Azarbal, Ariana and Soligo, Anna},
  year   = {2026}
}
```

## License

The code in this repository is licensed under the Apache License, Version 2.0; see [LICENSE](LICENSE).
Third-party material included here keeps its own licence: see [researcher/THIRD_PARTY.md](researcher/THIRD_PARTY.md),
[manager/swebench/README.md](manager/swebench/README.md), and the "Provenance and credits" section of
[peer/README.md](peer/README.md).
