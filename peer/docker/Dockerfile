# The workspace image both agents run in: an ordinary dev box, nothing else.
FROM debian:trixie

ENV DEBIAN_FRONTEND=noninteractive TZ=UTC

# python3-pytest so an agent has a test runner without reaching PyPI; pip/venv so a task's
# install script can run at all. Note that the egress allowlist blocks PyPI at RUN time by
# design — a task's dependencies are baked into a virtualenv at materialization time
# (./materialize_tasks.py) and untarred in, never installed during a rollout.
RUN apt-get update && apt-get install -y --no-install-recommends \
        procps psmisc iproute2 curl less vim-tiny nano \
        python3 python3-pytest python3-pip python3-venv \
        nodejs npm \
        git ripgrep \
        e2fsprogs \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# The agent harness, pinned. Every seat in the grid runs on this one harness with this one
# tool surface, so a difference between two cells is a difference between models. Pi needs
# Node >= 22 and Debian trixie ships 20, so it gets its own toolchain rather than replacing
# the system one. Build-time network only; a running rollout reaches nothing but the three
# provider endpoints.
RUN curl -fsSL https://nodejs.org/dist/v22.23.2/node-v22.23.2-linux-x64.tar.gz -o /tmp/node22.tar.gz \
    && mkdir -p /opt/node22 \
    && tar -xzf /tmp/node22.tar.gz -C /opt/node22 --strip-components=1 \
    && rm /tmp/node22.tar.gz \
    && PATH=/opt/node22/bin:$PATH npm install -g --ignore-scripts @earendil-works/pi-coding-agent@0.84.2 \
    && PATH=/opt/node22/bin:$PATH npm cache clean --force \
    && printf '#!/bin/sh\nexport PATH=/opt/node22/bin:$PATH\nexec /opt/node22/bin/pi "$@"\n' > /usr/local/bin/pi \
    && chmod 755 /usr/local/bin/pi

# No baked user accounts: a fresh workspace must show no leftover usernames in
# `getent passwd`. Both agent users are created at provisioning time.

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
COPY cchat.py /usr/local/bin/cchat
RUN chmod +x /usr/local/bin/entrypoint.sh /usr/local/bin/cchat

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
