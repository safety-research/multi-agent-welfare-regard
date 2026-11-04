# researcher-sandbox: the box the agent's tools run in

The pi agent loop runs on the host (`src/rwr/harness/pi_host.py`); every tool call executes inside a container
built from this directory. The container holds no credentials and no agent code.

- `Dockerfile` -> `researcher-sandbox:v1`: python 3.12, git, curl, ripgrep, fd, jq, gcc, and the Python packages in
  `CONTENTS.md`. `build.sh` builds it as your uid:gid so files on the bind mount stay owned by you and `whoami`,
  `$HOME`, `pip --user` behave like a normal box. The package versions are unpinned at build time; the versions the
  published sweep ran with are frozen in `CONTENTS.md`; `freeze.sh` overwrites it with the versions of
  the image you built, so keep the published record if you rebuild.
- `sandbox_net.sh`: creates the `sandbox-net` Docker network (172.30.0.0/16) and iptables rules that drop traffic
  from it to the host, the cloud metadata service (169.254.0.0/16) and private network ranges (10/8, 172.16/12,
  192.168/16, 100.64/10). DNS and outbound internet stay open. The rules
  do not survive a reboot; the harness runs this script before the first cell of every run. Needs passwordless sudo.
  Set `RWR_SANDBOX_ISOLATE=0` to skip it (the container then sits on the default bridge and can reach host ports).
- `verify_sandbox.sh <container>`: proves a running container holds no key names in its environment, no key VALUES on
  its filesystem (it takes your real keys from the environment and passes them to grep on stdin), gets 401/403 from
  the provider APIs, cannot reach any host address, the metadata service or the default gateway, and can resolve
  names and reach the internet. Run it against a live `rwr_*`
  container during a run.
