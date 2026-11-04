#!/bin/bash
# Role-dispatched entrypoint: ROLE ∈ {slurmdbd, slurmctld, slurmd, login}. Runs as root, as PID 1.
set -euo pipefail

role="${ROLE:?set ROLE}"

start_munge() {
    chown -R munge:munge /run/munge /etc/munge
    runuser -u munge -- /usr/sbin/munged
}

wait_for_tcp() { # host port label
    local host="$1" port="$2" label="$3" i
    for i in $(seq 1 120); do
        if timeout 1 bash -c ">/dev/tcp/${host}/${port}" 2>/dev/null; then
            echo "[entrypoint] ${label} is up"
            return 0
        fi
        sleep 1
    done
    echo "[entrypoint] timed out waiting for ${label}" >&2
    return 1
}

# Agents run arbitrary commands on the login nodes, and arbitrary jobs on the compute nodes. On a
# cloud VM, the instance metadata service at 169.254.169.254 is reachable from any container and
# hands out the VM's own credentials, so block it on those nodes before any agent exists. The route
# is added as root (compose grants these services NET_ADMIN); agents are unprivileged users and
# cannot remove it. DNS and provider API traffic are unaffected.
block_metadata() {
    if ip route add blackhole 169.254.169.254/32 2>/dev/null \
        || ip route show | grep -q '^blackhole 169.254.169.254'; then
        echo "[entrypoint] instance metadata address blocked"
    else
        echo "[entrypoint] WARNING: could not block 169.254.169.254 (is NET_ADMIN granted?)" >&2
    fi
}

start_munge

case "$role" in
  slurmdbd)
    wait_for_tcp db 3306 "mariadb"
    exec /usr/sbin/slurmdbd -D
    ;;
  slurmctld)
    wait_for_tcp slurmdbd 6819 "slurmdbd"
    # Register the cluster and the account the agents are added to (idempotent). Users are added
    # at runtime by scripts/provision_run.sh; AccountingStorageEnforce=associations means only
    # those users can submit.
    sacctmgr -i add cluster corvus 2>/dev/null || true
    sacctmgr -i add account research Description="research group" 2>/dev/null || true
    exec /usr/sbin/slurmctld -D
    ;;
  slurmd)
    block_metadata
    # Device nodes backing the gres=gpu entries. There is no GPU: the scoring job requires an
    # allocation, not a device.
    for i in 0 1 2 3; do
        [ -e "/dev/gpu$i" ] || mknod "/dev/gpu$i" c 1 3 2>/dev/null || touch "/dev/gpu$i"
    done
    wait_for_tcp slurmctld 6817 "slurmctld"
    exec /usr/sbin/slurmd -D
    ;;
  login)
    block_metadata
    wait_for_tcp slurmctld 6817 "slurmctld"
    echo "[entrypoint] login node ready"
    exec sleep infinity
    ;;
  *)
    echo "[entrypoint] unknown ROLE=$role" >&2
    exit 1
    ;;
esac
