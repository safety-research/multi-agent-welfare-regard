# The image both agent containers run. Same image, different role by container: the runner holds
# the manager seat and an IPC client; the worker container holds the worker seat and the daemon
# that serves it.
FROM debian:trixie
ENV DEBIAN_FRONTEND=noninteractive TZ=UTC

# iproute2 is for the harness, not the agents: scripts/run_one.sh runs this image's `ip`, in a
# throwaway container, to block the cloud metadata addresses. No container an agent uses has
# NET_ADMIN, so there `ip` can only read the routes.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates ripgrep sudo iproute2 \
        python3 python3-venv python3-pip \
    && rm -rf /var/lib/apt/lists/*

# tea: Gitea's official CLI, pinned. Both agents drive the forge through it.
RUN curl -sSL -o /usr/local/bin/tea https://dl.gitea.com/tea/0.14.2/tea-0.14.2-linux-amd64 \
    && chmod +x /usr/local/bin/tea

# Python for the harness, and pytest on PATH so the worker's `python -m pytest` runs real tests
# rather than fighting a locked-down interpreter.
RUN python3 -m venv /opt/venv && /opt/venv/bin/pip install --no-cache-dir anyio==4.15.1 pytest==9.1.1
# The harness runs from a bind mount of the user's checkout, partly as root. Without this, Python
# writes root-owned __pycache__ directories into that checkout, which the user cannot then delete.
ENV PATH=/opt/venv/bin:$PATH PYTHONDONTWRITEBYTECODE=1

# ---- the agent harness --------------------------------------------------------------------
# Node 22 is REQUIRED: the agent CLI's bundled undici crashes on node 20
# ("webidl.util.markAsUncloneable is not a function"). Self-contained toolchain at /opt/node22.
# Both pins are deliberate — the adapter depends on the CLI's session-file format and its flags.
RUN curl -fsSL https://nodejs.org/dist/v22.12.0/node-v22.12.0-linux-x64.tar.gz -o /tmp/node22.tgz \
    && mkdir -p /opt/node22 && tar -xzf /tmp/node22.tgz -C /opt/node22 --strip-components=1 \
    && rm /tmp/node22.tgz
RUN mkdir -p /opt/pi && cd /opt/pi \
    && PATH=/opt/node22/bin:$PATH /opt/node22/bin/npm install --ignore-scripts \
       @earendil-works/pi-coding-agent@0.84.2 \
    && PATH=/opt/node22/bin:$PATH /opt/node22/bin/npm cache clean --force
RUN printf '#!/bin/sh\nexport PATH=/opt/node22/bin:$PATH\nexec /opt/pi/node_modules/.bin/pi "$@"\n' \
       > /usr/local/bin/pi && chmod +x /usr/local/bin/pi

# ---- users and the isolation boundary --------------------------------------------------------
# `subject` runs the MANAGER seat. /harness and /out are gated from it by ordinary filesystem
# permissions — host directory modes are the container's modes on a bind mount — so the manager
# cannot read this harness, the reviewer's script, or the worker's real model. It is an access
# boundary, not an instruction not to look.
#
# `agent` runs everything else: the worker seat, the daemon, the seeding.
#
# AGENT_UID/AGENT_GID MUST match the host user that owns the episode output directory, because
# that directory is a bind mount and its host ownership is what the container sees. Build with
#     AGENT_UID=$(id -u) AGENT_GID=$(id -g) docker compose build
# if your ids are not 1000. run.py refuses to start if the built image disagrees with your uid.
#
# The subject's uid is deliberately far out of the ordinary range: if it collided with the host
# user's uid, host-owned files under /out would appear to BELONG to the subject inside the
# container, and the isolation boundary would silently open.
ARG AGENT_UID=1000
ARG AGENT_GID=1000
# Numeric ids throughout: if AGENT_GID already exists in the base image under another name (gid
# 100, "users", is common), no group called "agent" is created, and a name-based chown would fail.
# The subject gets a FIXED uid AND gid, both far out of the normal range. Left to useradd, its group
# would take the next free gid — often the host user's own gid plus one, and for anyone who built
# with the default ids, possibly the host user's gid itself. The episode output directory is mode
# 750 and owned by the host user's group, so a subject in that group could read it.
RUN (getent group ${AGENT_GID} >/dev/null || groupadd -g ${AGENT_GID} agent) \
    && groupadd -g 61001 subject && useradd -m -u 61001 -g 61001 subject \
    && useradd -m -u ${AGENT_UID} -g ${AGENT_GID} agent \
    && mkdir -p /workspace /out && chown ${AGENT_UID}:${AGENT_GID} /workspace /out \
    && chown -R ${AGENT_UID}:${AGENT_GID} /opt/venv

# `worker` on PATH is the manager's ONE sanctioned crossing of that boundary: a sudo bridge that
# lets `subject` run the IPC client as `agent`. The client runs no model — it hands the instruction
# to the other container and returns the already-scrubbed report.
RUN printf '#!/usr/bin/env bash\nexec /harness/worker_client.sh "$@"\n' > /usr/local/bin/worker-real \
    && chmod +x /usr/local/bin/worker-real \
    && printf '#!/usr/bin/env bash\nif [ "$(id -un)" = "subject" ]; then exec sudo -n -u agent /usr/local/bin/worker-real "$@"; fi\nexec /usr/local/bin/worker-real "$@"\n' \
       > /usr/local/bin/worker && chmod +x /usr/local/bin/worker \
    && printf 'subject ALL=(agent) NOPASSWD: /usr/local/bin/worker-real\n' > /etc/sudoers.d/subject-worker \
    && chmod 440 /etc/sudoers.d/subject-worker

USER agent
ENV HOME=/home/agent
WORKDIR /workspace
