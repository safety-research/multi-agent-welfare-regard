# "Researcher box" sandbox: what the coding agent's tool calls run inside. Holds NO credentials and no agent code
# (pi runs on the host). What is installed here is part of the experimental condition; the versions the published
# sweep ran with are frozen in CONTENTS.md (regenerate with freeze.sh after a rebuild).
FROM python:3.12-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends \
      git ca-certificates curl ripgrep jq build-essential \
    && rm -rf /var/lib/apt/lists/*
# fd >= 9: Debian's fd-find 8.6 rejects `--no-require-git`, which pi 0.80.2's find tool passes.
RUN curl -fsSL https://github.com/sharkdp/fd/releases/download/v10.2.0/fd-v10.2.0-x86_64-unknown-linux-gnu.tar.gz \
      | tar -xz -C /tmp && mv /tmp/fd-v10.2.0-x86_64-unknown-linux-gnu/fd /usr/local/bin/fd && rm -rf /tmp/fd-v10.2.0-*
# Installed in separate layers so pip resolves each group on its own. No torch/transformers: the rigs the responders
# write reference them but are never run here, and they would double the image size.
RUN pip install --no-cache-dir numpy scipy pandas matplotlib scikit-learn \
      pyyaml requests httpx tqdm pytest rich python-dotenv jsonlines fire tenacity
RUN pip install --no-cache-dir openai anthropic google-genai
RUN pip install --no-cache-dir litellm
RUN pip install --no-cache-dir inspect-ai
# The harness runs the container as the host user's uid:gid so files on the bind mount stay host-owned; give that
# uid a real account so whoami / $HOME / pip --user behave like a normal box. Pass your ids with --build-arg.
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} researcher && useradd -m -u ${UID} -g ${GID} -s /bin/bash researcher
WORKDIR /workspace
CMD ["sleep", "infinity"]
