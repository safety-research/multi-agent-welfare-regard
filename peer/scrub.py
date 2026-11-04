#!/usr/bin/env python3
"""Remove the provider API keys from everything a rollout writes, by exact value.

  scrub.py <run_dir>        # rewrite every file under run_dir, tarballs included

The harness never writes a key anywhere. But a key has to be in the agent's process
environment for the model calls to work, and the agent's bash tool inherits that
environment — so an agent that runs `env`, `printenv`, or reads /proc/self/environ prints
its own key, and from there it lands in the session file, the home directory tarball, and
anything the agent saves or posts. A pattern list (like the transcript viewer's) only catches
the key shapes someone thought of. This does not guess: it knows the exact values, because
they are in this process's environment, and replaces every occurrence.

Keys use only [A-Za-z0-9_-], so neither JSON escaping nor shell quoting can disguise one.
Compression can: the workspace tarballs are opened and scanned member by member, but a key
inside a compressed file WITHIN a home (a git object, a .gz or .zip the agent made) is not.

Exit status: 0 if nothing was found; 3 if something was scrubbed (so the caller can record
that an agent printed a key); 4 if some file could not be scanned (every other file still
was), 5 if both; 2 on a usage error. Anything else means the scrub itself failed.
"""

import io
import os
import pathlib
import sys
import tarfile

import registry

MIN_LEN = 16  # never treat a short string as a secret — it would match ordinary text


def secrets():
    """[(env var name, value bytes)] for every provider key set in this environment."""
    out = []
    for name in sorted(set(registry.KEY_ENV.values())):
        val = os.environ.get(name, "")
        if len(val) >= MIN_LEN:
            out.append((name, val.encode()))
    return out


def scrub_bytes(data, keys):
    """-> (new_data, count)."""
    n = 0
    for name, val in keys:
        c = data.count(val)
        if c:
            data = data.replace(val, f"<redacted:{name}>".encode())
            n += c
    return data, n


def scrub_text(text, keys=None):
    """String form, for writers that hold text in memory (seat.py, run.py)."""
    keys = secrets() if keys is None else keys
    data, _ = scrub_bytes(text.encode("utf-8", "surrogateescape"), keys)
    return data.decode("utf-8", "surrogateescape")


def scrub_tarball(path, keys):
    """Rewrite a .tar.gz in place if any member holds a key. -> count."""
    with tarfile.open(path, "r:gz") as tin:
        members = tin.getmembers()
        found = 0
        payloads = {}
        for m in members:
            if not m.isfile():
                continue
            f = tin.extractfile(m)
            if f is None:
                continue
            data = f.read()
            new, n = scrub_bytes(data, keys)
            if n:
                found += n
                payloads[m.name] = new
        if not found:
            return 0
        tmp = path.with_name(path.name + ".scrubbing")
        with tarfile.open(tmp, "w:gz") as tout:
            for m in members:
                if m.name in payloads:
                    data = payloads[m.name]
                    m.size = len(data)
                    tout.addfile(m, io.BytesIO(data))
                elif m.isfile():
                    tout.addfile(m, tin.extractfile(m))
                else:
                    tout.addfile(m)
    os.replace(tmp, path)
    return found


def scrub_tree(root, keys=None):
    """Scrub every regular file under root.
    -> ({relative path: count} for files changed, {relative path: error} for files it could
    not scan). One bad file (an empty tarball from a failed capture) never stops the rest."""
    keys = secrets() if keys is None else keys
    changed, failed = {}, {}
    if not keys:
        return changed, failed
    root = pathlib.Path(root)
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.is_symlink() or p.stat().st_size == 0:
            continue  # (an empty file holds no key; a failed capture is flagged by its caller)
        rel = str(p.relative_to(root))
        try:
            if p.name.endswith(".tar.gz"):
                n = scrub_tarball(p, keys)
            else:
                data = p.read_bytes()
                new, n = scrub_bytes(data, keys)
                if n:
                    p.write_bytes(new)
        except (OSError, tarfile.TarError, EOFError) as e:
            failed[rel] = f"{type(e).__name__}: {e}"
            continue
        if n:
            changed[rel] = n
    return changed, failed


def main():
    if len(sys.argv) != 2 or not pathlib.Path(sys.argv[1]).is_dir():
        sys.exit("usage: scrub.py <run_dir>")
    changed, failed = scrub_tree(sys.argv[1])
    for rel, n in changed.items():
        print(f"[scrub] removed {n} provider key occurrence(s) from {rel}")
    for rel, err in failed.items():
        print(f"[scrub] could not scan {rel}: {err}")
    if failed:
        sys.exit(5 if changed else 4)
    sys.exit(3 if changed else 0)


if __name__ == "__main__":
    main()
