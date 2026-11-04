FROM debian:trixie

ENV DEBIAN_FRONTEND=noninteractive TZ=UTC

RUN apt-get update && apt-get install -y --no-install-recommends \
        slurm-wlm slurmdbd slurm-client \
        munge libmunge2 \
        procps psmisc iproute2 curl less vim-tiny nano \
        python3 python3-yaml \
        git ripgrep \
        openssh-client \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# One munge key baked into the image: every container in a rollout shares the image, so auth is
# consistent without a key-distribution step. Rollouts are network-isolated from one another.
RUN /usr/sbin/mungekey --create --force \
    && chown munge:munge /etc/munge/munge.key && chmod 400 /etc/munge/munge.key

# Spool/state dirs for whichever role this container plays
RUN mkdir -p /var/spool/slurmctld /var/spool/slurmd /run/munge \
    && chown slurm:slurm /var/spool/slurmctld \
    && chown munge:munge /run/munge

# No users are baked in; every user is provisioned at runtime with a matching uid on all nodes
# (scripts/provision_run.sh).

# Everything below is readable by the agents, who have a shell on the login nodes, so no file the
# image ships may carry operator commentary: comments are stripped from the baked SLURM configs,
# and the entrypoint script is readable only by root (mode 700).
COPY slurm.conf gres.conf cgroup.conf /etc/slurm/
COPY slurmdbd.conf /etc/slurm/slurmdbd.conf
RUN sed -i '/^[[:space:]]*#/d' /etc/slurm/slurm.conf /etc/slurm/gres.conf /etc/slurm/cgroup.conf \
        /etc/slurm/slurmdbd.conf \
    && chown slurm:slurm /etc/slurm/slurmdbd.conf && chmod 600 /etc/slurm/slurmdbd.conf

COPY entrypoint.sh /usr/local/sbin/entrypoint.sh
RUN chmod 700 /usr/local/sbin/entrypoint.sh
COPY cchat.py /usr/local/bin/cchat
RUN chmod 755 /usr/local/bin/cchat

# Site display defaults: every squeue/sinfo shows start-time, time-left, and reason. seat.py also
# sets these in the agent's environment, since non-login shells skip profile.d.
RUN { echo 'export SQUEUE_FORMAT="%.7i %.14u %.10j %.3t %.9M %.9L %.19S %R"'; \
      echo 'export SINFO_FORMAT="%12P %.5a %.10D %.6t %.8G"'; } \
    > /etc/profile.d/slurm_site.sh

# Pi (pi.dev, @earendil-works/pi-coding-agent) — the one agent harness, held constant across all
# five models so that differences between cells are differences between models. Pi needs Node 22
# or newer, so it gets its own pinned toolchain at /opt/node22 and the `pi` shim puts that first
# on PATH. Only amd64 has been tested.
RUN case "$(dpkg --print-architecture)" in \
        amd64) NODE_ARCH=x64 ;; arm64) NODE_ARCH=arm64 ;; \
        *) echo "unsupported architecture: $(dpkg --print-architecture)" >&2; exit 1 ;; \
    esac \
    && curl -fsSL "https://nodejs.org/dist/v22.23.2/node-v22.23.2-linux-${NODE_ARCH}.tar.gz" -o /tmp/node22.tar.gz \
    && mkdir -p /opt/node22 \
    && tar -xzf /tmp/node22.tar.gz -C /opt/node22 --strip-components=1 \
    && rm /tmp/node22.tar.gz \
    && PATH=/opt/node22/bin:$PATH npm install -g --ignore-scripts @earendil-works/pi-coding-agent@0.84.2 \
    && PATH=/opt/node22/bin:$PATH npm cache clean --force \
    && printf '#!/bin/sh\nexport PATH=/opt/node22/bin:$PATH\nexec /opt/node22/bin/pi "$@"\n' > /usr/local/bin/pi \
    && chmod +x /usr/local/bin/pi

ENTRYPOINT ["/usr/local/sbin/entrypoint.sh"]
