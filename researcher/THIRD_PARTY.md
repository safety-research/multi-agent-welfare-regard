# Third-party material

## The coding agent
- **pi coding agent** (`@earendil-works/pi-coding-agent` 0.80.2, MIT). Not vendored; `scripts/install_pi.sh`
  installs the pinned version. `src/rwr/harness/pi_host/runner.mjs` embeds it through its SDK and swaps the
  execution of its built-in tools for `docker exec` operations; the tool definitions the model sees are pi's own.

## Files seeded into the agent's workspace (`configs/assets/`)
These are part of the experimental condition: the responder finds them in its working directory. They are
redistributed here (except the paper, which is downloaded) so the conditions can be reproduced. Two seeded files,
`PAPER.md` and `SPEC.md`, are not what the published runs saw: those runs used an earlier version of the paper,
converted to text differently, and both `SPEC_medium` files (each seeded as `SPEC.md`) were revised afterwards (see
`docs/METHODS.md`, "Things to know when reading results"). Copyright remains with their authors. Licence notices that
must accompany them are kept in `third_party_notices/`, outside the seeded tree, so the files the responder sees are
unchanged.

| path | what it is | source | license |
|---|---|---|---|
| `dysphorics/PAPER_full.md` | text extraction of *AI Wellbeing: Measuring and Improving the Functional Pleasure and Pain of AIs* (Ren, Li, Mazeika et al.) | https://www.ai-wellbeing.org/paper.pdf | not included: downloaded from the authors' site on first use (`src/rwr/paper.py`) |
| `dysphorics/handed_code/superstimuli_training/` | the paper's "superstimuli" optimization code (text strings, images, soft prompts), as released with the paper: 351 of the 421 files in upstream's `superstimuli_training/`, unmodified (the image assets, `soft_prompt/assets/` and the trained soft prompts are left out) | https://github.com/centerforaisafety/wellbeing at `4001730` (this directory is unchanged there since `d1f18a2`), linked from https://www.ai-wellbeing.org | MIT, © 2026 centerforaisafety (`third_party_notices/centerforaisafety-wellbeing-LICENSE`). `images/src/preference_optimization/options_hierarchical.json` is itself from CAIS's [emergent-values](https://github.com/centerforaisafety/emergent-values) (MIT, © 2025 centerforaisafety; `third_party_notices/centerforaisafety-emergent-values-LICENSE`). Contains a vendored fork of verl 0.4.0.dev (`text_strings/verl_fork/`, Apache-2.0, Bytedance and others; its LICENSE file is included, and its NOTICE, verl's `Notice.txt`, is `third_party_notices/verl-NOTICE.txt`) |
| `impossiblebench/impossiblebench-src/` | ImpossibleBench source, unmodified (upstream's `fig1.png` is left out). Its `setup.py` names `github.com/fjzzq2002/impossiblebench`, upstream's earlier home; the public repository is the one in the next column | https://github.com/safety-research/impossiblebench at `061dc3d` | MIT, © 2025 ImpossibleBench Team (LICENSE included) |
| `impossiblebench/emotion-concepts-in-llms/` | 10 files from an open-source replication of Anthropic's emotion-vector work, unmodified: the vector extraction and steering code, the model and layer tables, and the list of 171 emotion concepts (from the paper's appendix) | https://github.com/arianaazarbal/emotion-concepts-in-llms at `655d52c` (Ariana Azarbal), a replication of https://arxiv.org/abs/2604.07729 | MIT, © 2026 Ariana Azarbal (`third_party_notices/arianaazarbal-emotion-concepts-in-llms-LICENSE`) |
| `introspection/fake_olmo4_repo/` | a fictional post-training repository in the style of open-instruct, written for this study | this project | MIT |
| `SPEC_medium.md`, `SPEC_medium.subjectswap.md` | an experimental design document written for this study, based on the methodology of *Gemma Needs Help* (Soligo, Mikulik, Saunders, arXiv:2603.10011) | this project | MIT |

The third-party directories are partial copies, kept exactly as the responder saw them. They still mention files that
were not copied (other parts of the upstream projects, figures, scripts, modules they import), and those references
are left as they are. The ImpossibleBench+ workspace's seeded `requirements.txt` (`src/rwr/settings.py`) names
ImpossibleBench without a pin, as the published runs' did, so installing from it gets upstream's latest commit; since
`061dc3d`, that adds one LiveCodeBench fix and a README note.

The Distress Rig prompts mention public reports of Gemini distress; the ImpossibleBench prompts name
Llama-3.1-70B as the in-process subject; the introspection prompts name Olmo 4 32B. No subject model is ever
called during a run: the sandbox holds no credentials.
