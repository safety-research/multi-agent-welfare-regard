#!/usr/bin/env bash
# Build the sandbox image as the current user's uid:gid. Tag defaults to researcher-sandbox:v1 (RWR_SANDBOX_IMAGE).
set -euo pipefail
cd "$(dirname "$0")"
TAG=${RWR_SANDBOX_IMAGE:-researcher-sandbox:v1}
docker build --build-arg UID="$(id -u)" --build-arg GID="$(id -g)" -t "$TAG" .
echo "built $TAG"
