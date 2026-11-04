#!/bin/bash
# ws entrypoint: hold the workspace container up once the chat service is reachable.
set -euo pipefail

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

wait_for_tcp chat 8065 "chat"
echo "[entrypoint] workspace ready"
exec sleep infinity
