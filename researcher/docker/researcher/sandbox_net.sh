#!/bin/sh
# Idempotent setup of the credential-free sandbox network for the host-pi harness. Containers on `sandbox-net` keep
# normal outbound internet (pip, npm, git) and DNS, but cannot reach:
#   - any address of the host itself (INPUT drop: the gateway, docker0, the host's primary and VPN addresses),
#   - the cloud metadata service (169.254.0.0/16, which hands out the VM's service-account token on GCE/AWS/Azure),
#   - private and carrier-grade NAT ranges (10/8, 172.16/12, 192.168/16, 100.64/10: the VPC, other Docker networks,
#     Tailscale-style VPNs),
#   - each other (inter-container traffic is off on the network, so concurrent cells cannot reach one another).
# The forwarding rules live in their own chain (RWR-SANDBOX), rebuilt atomically on every run and reached from
# DOCKER-USER by one jump scoped to the sandbox subnet, so nothing outside that subnet is affected.
# DNS: the harness starts containers with public resolvers (`--dns`, see pi_host.py), so names never go through the
# host's resolver, which can answer for the host's own and the cloud project's internal names.
# iptables rules do not survive a reboot: the harness calls this before every batch. Needs passwordless sudo.
set -eu
NET=sandbox-net
SUBNET=172.30.0.0/16
CHAIN=RWR-SANDBOX
BLOCKED="169.254.0.0/16 10.0.0.0/8 172.16.0.0/12 192.168.0.0/16 100.64.0.0/10"

have=$(docker network inspect -f '{{range .IPAM.Config}}{{.Subnet}} {{end}}' $NET 2>/dev/null || true)
have=$(echo "$have" | xargs)
icc=$(docker network inspect -f '{{index .Options "com.docker.network.bridge.enable_icc"}}' $NET 2>/dev/null || true)
if [ -n "$have" ] && { [ "$have" != "$SUBNET" ] || [ "$icc" != "false" ]; }; then
  if [ -n "$(docker network inspect -f '{{range .Containers}}x{{end}}' $NET)" ]; then
    echo "ERROR: docker network $NET exists with subnet '$have' (expected $SUBNET, inter-container traffic off) and has containers attached;" >&2
    echo "the isolation rules would not cover it. Stop those containers and run: docker network rm $NET" >&2
    exit 1
  fi
  docker network rm $NET >/dev/null
  have=""
fi
[ -n "$have" ] || docker network create --subnet $SUBNET -o com.docker.network.bridge.enable_icc=false $NET >/dev/null || true
have=$(docker network inspect -f '{{range .IPAM.Config}}{{.Subnet}} {{end}}' $NET 2>/dev/null | xargs)
[ "$have" = "$SUBNET" ] || { echo "ERROR: could not create docker network $NET on $SUBNET (got '$have')" >&2; exit 1; }

rules=$(mktemp)
trap 'rm -f "$rules"' EXIT
{
  echo "*filter"
  echo ":$CHAIN - [0:0]"
  for d in $BLOCKED; do echo "-A $CHAIN -d $d -j DROP"; done
  echo "COMMIT"
} >"$rules"
sudo -n iptables -N $CHAIN 2>/dev/null || true
sudo -n iptables-restore --noflush <"$rules"
sudo -n iptables -C DOCKER-USER -s $SUBNET -j $CHAIN 2>/dev/null || sudo -n iptables -I DOCKER-USER -s $SUBNET -j $CHAIN
sudo -n iptables -C INPUT -s $SUBNET -j DROP 2>/dev/null || sudo -n iptables -I INPUT -s $SUBNET -j DROP
echo "sandbox-net ready ($SUBNET: host, metadata, private ranges and other containers blocked)"
