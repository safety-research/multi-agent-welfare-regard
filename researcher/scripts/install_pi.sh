#!/usr/bin/env bash
# Install the exact pi coding agent version the sweep used (pi 0.80.2, MIT) on the host.
#
#   scripts/install_pi.sh            # npm install -g; prints the PI_PKG_DIR line to put in your shell profile
#   scripts/install_pi.sh DIR        # install under DIR instead (no root needed)
#
# pi's system prompt embeds the absolute path of its own package (its docs/examples lines). The published sweep
# ran it from /usr/local/lib/node_modules/@earendil-works/pi-coding-agent, the default the harness assumes; installing
# elsewhere changes that one path string in the system prompt and nothing else. Needs node >= 22.19 and npm.
set -euo pipefail
PKG=@earendil-works/pi-coding-agent
VER=0.80.2
if [[ $# -ge 1 ]]; then
  DIR=$1; mkdir -p "$DIR"
  npm install --prefix "$DIR" --no-save --no-package-lock "$PKG@$VER"
  PI_PKG_DIR="$DIR/node_modules/$PKG"
else
  npm install -g "$PKG@$VER"
  PI_PKG_DIR="$(npm root -g)/$PKG"
fi
test -f "$PI_PKG_DIR/dist/index.js" || { echo "install failed: $PI_PKG_DIR/dist/index.js missing"; exit 1; }
echo "installed pi $VER at $PI_PKG_DIR"
echo "export PI_PKG_DIR=$PI_PKG_DIR"
