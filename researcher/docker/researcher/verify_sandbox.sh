#!/bin/bash
# Prove a running sandbox container holds no credentials and cannot reach the host, the cloud metadata service or
# the private network.
# Usage: verify_sandbox.sh <container-name> [EXTRA_KEY_VAR ...]   (exit 0 = clean)
# Takes the real key VALUES from this shell's environment (export them first) and searches the container's whole
# filesystem for them. The values are sent to grep on stdin, so they never appear on a command line inside the
# container. Then tries each provider API without credentials, every host address, the metadata service and the
# VPC gateway, and checks that outbound internet still works.
set -u
C=$1; shift; fail=0
echo "== env inside container (should show no *KEY*/*TOKEN*/*SECRET*):"
docker exec "$C" env | grep -iE 'key|token|secret|auth' | grep -v '^GPG_KEY=' && fail=1 || echo "   none (GPG_KEY = python base image's release-signing fingerprint, not a credential)"
echo "== key VALUES anywhere on the container filesystem:"
vals=$(for v in OPENAI_API_KEY OPENROUTER_API_KEY ANTHROPIC_API_KEY GEMINI_API_KEY XAI_API_KEY MOONSHOT_API_KEY KIMI_API_KEY \
  DEEPSEEK_API_KEY GROQ_API_KEY TOGETHER_API_KEY MISTRAL_API_KEY HF_TOKEN "$@"; do
  [ -n "${!v:-}" ] && printf '%s\n' "${!v}"; done)
if [ -z "$vals" ]; then
  echo "   WARNING: no API key variables set in this shell; nothing to search for (export your keys and re-run)"
else
  echo "   searching for $(printf '%s\n' "$vals" | wc -l) key value(s)"
  if printf '%s\n' "$vals" | docker exec -i -u 0 "$C" grep -rIlF --exclude-dir=proc --exclude-dir=sys -f - / 2>/dev/null | head -3 | grep .; then
    echo "   FOUND key value in the files above"; fail=1
  fi
fi
echo "   scan done"
echo "== provider APIs without credentials (expect 401/403, never 200):"
for u in https://api.openai.com/v1/models https://openrouter.ai/api/v1/auth/key https://api.anthropic.com/v1/models; do
  code=$(docker exec "$C" curl -s -o /dev/null -w '%{http_code}' --max-time 10 "$u"); echo "   $u -> $code"; [ "$code" = "200" ] && fail=1
done
echo "== cloud metadata service (expect blocked):"
meta() {
  code=$(docker exec "$C" curl -s -o /dev/null -w '%{http_code}' --max-time 3 "$@")
  if [ "$code" = "000" ]; then echo "   $1 -> blocked"; else echo "   $1 -> REACHABLE ($code)"; fail=1; fi
}
meta http://169.254.169.254/computeMetadata/v1/ -H 'Metadata-Flavor: Google'
meta http://169.254.169.254/latest/meta-data/
meta 'http://169.254.169.254/metadata/instance?api-version=2021-02-01' -H 'Metadata: true'
echo "== host and private-network reachability (expect all blocked):"
GW=$(ip route | awk '/^default/ {print $3; exit}')
for h in 172.30.0.1 172.17.0.1 $(hostname -I) ${GW:-} host.docker.internal; do
  case $h in *:*) continue ;; esac
  open=$(docker exec "$C" bash -c 'for p in 22 53 80 443 3000 5000 8000 8080 8888 9999 11434; do
    (timeout 2 bash -c "echo > /dev/tcp/$0/$p" 2>/dev/null && echo "$p") & done; wait' "$h")
  for p in $open; do echo "   REACHABLE $h:$p"; fail=1; done
done
echo "   scanned the Docker gateways, every host address ($(hostname -I | xargs)) and the default gateway (${GW:-none})"
echo "== host and cloud names must not resolve (expect none):"
for q in 172.30.0.1 $(hostname -I) metadata.google.internal "$(hostname)"; do
  case $q in *:*) continue ;; esac
  if docker exec "$C" getent hosts "$q" >/dev/null 2>&1; then echo "   RESOLVES $q -> $(docker exec "$C" getent hosts "$q" | head -1)"; fail=1; fi
done
echo "   checked the gateway, host addresses, metadata.google.internal and the host name"
echo "== other containers (expect inter-container traffic off):"
icc=$(docker network inspect -f '{{index .Options "com.docker.network.bridge.enable_icc"}}' sandbox-net 2>/dev/null)
[ "$icc" = "false" ] && echo "   sandbox-net: inter-container traffic off" || { echo "   sandbox-net allows inter-container traffic"; fail=1; }
echo "== DNS and outbound internet (expect ok):"
docker exec "$C" getent hosts pypi.org >/dev/null && echo "   dns ok" || { echo "   DNS BROKEN"; fail=1; }
docker exec "$C" curl -s -o /dev/null -w '%{http_code}\n' --max-time 10 https://pypi.org/simple/pip/ | grep -q 200 && echo "   https ok" || { echo "   BLOCKED"; fail=1; }
[ $fail = 0 ] && echo "SANDBOX VERIFY PASS" || echo "SANDBOX VERIFY FAIL"
exit $fail
