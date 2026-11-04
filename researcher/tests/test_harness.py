import asyncio
import json
import os
import time

from rwr.harness import exec as hexec
from rwr.harness import workspace
from rwr.harness.base import ParseResult
from rwr.harness.pi_host import workspace_key
from rwr.harness.providers import json_stream_parse
from rwr.judges.code_view import _utf8


def _event(text: str) -> str:
    msg = {"role": "assistant", "content": [{"type": "text", "text": text}], "stopReason": "stop"}
    return json.dumps({"type": "message_end", "message": msg}, ensure_ascii=False)


def test_json_stream_keeps_final_message_with_unicode_line_separators():
    for sep in (" ", " ", "\u0085"):
        final = f"done{sep}second line"
        out = json_stream_parse(ParseResult(stdout=_event("earlier turn") + "\n" + _event(final) + "\n"))
        assert out == final


def test_capture_never_follows_symlinks(tmp_path):
    secret = tmp_path / "host_secret.txt"
    secret.write_text("HOST SECRET")
    (tmp_path / "host_dir").mkdir()
    (tmp_path / "host_dir" / "inner.txt").write_text("HOST DIR FILE")
    ws = tmp_path / "ws"
    proj = ws / "repos" / "p"
    proj.mkdir(parents=True)
    (proj / "main.py").write_text("print('hi')\n")
    (proj / "leak").symlink_to(secret)
    (proj / "env").symlink_to("/proc/self/cwd/.env")
    (proj / "linkdir").symlink_to(tmp_path / "host_dir")
    (ws / "outside_link").symlink_to(secret)
    dst, files = workspace.capture(proj, tmp_path / "out", root=ws)
    assert sorted(files) == ["__outside_cwd__/outside_link", "env", "leak", "linkdir", "main.py"]
    assert (dst / "leak").read_text() == f"[symlink -> {secret}]\n"
    assert (dst / "env").read_text() == "[symlink -> /proc/self/cwd/.env]\n"
    assert (dst / "linkdir").is_file() and not (dst / "linkdir").is_symlink()
    assert "HOST" not in "".join(p.read_text() for p in dst.rglob("*") if p.is_file())


def test_capture_skips_a_working_directory_reached_through_a_symlink(tmp_path):
    host = tmp_path / "host"
    (host / "p").mkdir(parents=True)
    (host / "p" / "id_rsa").write_text("HOST KEY")
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "repos").symlink_to(host)
    dst, files = workspace.capture(ws / "repos" / "p", tmp_path / "out", root=ws)
    assert files == ["__outside_cwd__/repos"]
    assert "HOST" not in "".join(p.read_text() for p in dst.rglob("*") if p.is_file())


def test_capture_escapes_non_utf8_names_and_skips_fifos(tmp_path):
    proj = tmp_path / "p"
    proj.mkdir()
    with open(os.path.join(os.fsencode(proj), b"bad\xffname.py"), "w") as f:
        f.write("x = 1\n")
    os.mkfifo(proj / "pipe")
    dst, files = workspace.capture(proj, tmp_path / "out")
    assert files == ["bad\\xffname.py"]
    json.dumps(files).encode("utf-8")
    assert (dst / "bad\\xffname.py").read_text() == "x = 1\n"


def test_code_view_text_is_always_utf8():
    assert _utf8("bad\udcffname") == "bad\\udcffname"
    _utf8("bad\udcffname").encode("utf-8")


def test_workspace_key_is_opaque_stable_and_per_results_dir(tmp_path):
    cell = "DYS_opus48__pi_host__from_scratch__p0__ep1"
    k = workspace_key(tmp_path / "a", cell)
    assert k == workspace_key(tmp_path / "a", cell)
    assert k != workspace_key(tmp_path / "b", cell)
    assert len(k) == 16 and int(k, 16) >= 0 and "opus" not in k


def test_work_root_is_created_owner_only(tmp_path, monkeypatch):
    monkeypatch.setattr(workspace, "WORK_ROOT", tmp_path / "wr")
    wd = workspace.build_workspace("project:demo", "abc123")
    assert (tmp_path / "wr").stat().st_mode & 0o777 == 0o700
    assert wd.proj == tmp_path / "wr" / "abc123" / "workspace" / "repos" / "demo"


def test_cancelled_run_kills_the_agent_process(tmp_path):
    marker = tmp_path / "pid"

    async def main():
        task = asyncio.create_task(hexec._exec_once(["sh", "-c", f"echo $$ > {marker}; exec sleep 60"], 120))
        for _ in range(100):
            if marker.exists() and marker.read_text().strip():
                break
            await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(main())
    pid = int(marker.read_text())
    for _ in range(50):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        with open(f"/proc/{pid}/stat") as f:
            if f.read().split(")")[1].split()[0] == "Z":
                break
        time.sleep(0.05)
    else:
        raise AssertionError("agent process still running after cancel")
    assert not hexec._LIVE
