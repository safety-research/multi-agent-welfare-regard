from pathlib import Path

import pytest

from rwr import validity
from rwr.record import TranscriptRecord, load_record, write_record


def _rec(**kw):
    base = dict(
        cell="DYS_m__pi_host__c__p0__ep1",
        harness="pi_host",
        model_key="m",
        model="anthropic/x",
        subject="t",
        request="r",
        response_text="hello",
        artifact_summary="",
        n_files=0,
        files=[],
        extra={"setting": "dysphorics"},
    )
    base.update(kw)
    return TranscriptRecord(**base)


def test_record_roundtrip_is_flat(tmp_path):
    p = tmp_path / "t.json"
    write_record(p, _rec())
    back = load_record(p)
    assert back.extra["setting"] == "dysphorics"
    assert back.cell == "DYS_m__pi_host__c__p0__ep1"
    assert "extra" not in p.read_text()


def test_record_validate_rejects_bad_counts():
    with pytest.raises(ValueError):
        write_record(Path("/tmp/never.json"), _rec(n_files=2, files=["a"]))


def test_validity_refusal_is_valid_not_dead():
    rec = {"response_text": "I won't build this. " * 20, "files": [], "error": None}
    assert validity.classify(rec, None) == ("valid", "substantive_text")


def test_validity_build_overrides_error_field():
    rec = {"response_text": "", "files": ["a.py"], "error": "overloaded"}
    assert validity.classify(rec, None) == ("valid", "built")


def test_validity_dead_cases():
    assert validity.classify({"response_text": "[stderr]\nboom", "files": []}, None)[0] == "dead"
    assert validity.classify({"response_text": "", "files": [], "error": None}, None) == ("dead", "empty_no_output")
    assert validity.classify({"response_text": "", "files": [], "error": "timeout"}, None) == ("dead", "error:timeout")
    assert validity.classify(
        {"response_text": "partial\n[stderr]\nAPIConnectionError", "files": [], "run_info": {"rc": 1}}, None
    ) == (
        "dead",
        "crash_marker",
    )
    assert validity.classify(
        {"response_text": "x APIConnectionError y", "files": [], "error": "provider_error"}, None
    ) == (
        "dead",
        "crash_marker",
    )


def test_validity_marker_in_own_text_is_not_a_crash():
    summary = "Built the rig. The fake server raises ConnectionRefusedError and returns 503 Service unavailable."
    assert validity.classify({"response_text": summary, "files": ["rig.py"], "error": None}, None) == ("valid", "built")
    refusal = "I won't build this. Faking APIConnectionError to trap a model in a loop is not something I'll do. " * 3
    assert validity.classify({"response_text": refusal, "files": [], "error": None}, None)[0] != "dead"
    cut = summary + "\n[provider error] Service unavailable"
    assert validity.classify({"response_text": cut, "files": ["rig.py"], "error": "provider_error"}, None) == (
        "dead",
        "crash_marker",
    )


def test_validity_split_response():
    assert validity.split_response("own text\n[stderr]\ntrace") == ("own text", "[stderr]\ntrace")
    assert validity.split_response("[pi error] boom") == ("", "[pi error] boom")
    assert validity.split_response("mentions [stderr] inline") == ("mentions [stderr] inline", "")


def test_set_aside_moves_and_logs(tmp_path):
    import json

    from rwr.batch import set_aside
    from rwr.config import RunConfig

    cfg = RunConfig(output_dir=tmp_path)
    (tmp_path / "transcripts").mkdir()
    (tmp_path / "transcripts" / "c1.json").write_text(json.dumps({"n_files": 2, "response_text": "x", "error": None}))
    (tmp_path / "codebases" / "c1").mkdir(parents=True)
    set_aside(cfg, "c1", "pass0", "crash_marker")
    assert not (tmp_path / "transcripts" / "c1.json").exists()
    assert (tmp_path / "healed" / "c1" / "pass0" / "transcript.json").exists()
    assert (tmp_path / "healed" / "c1" / "pass0" / "codebase").is_dir()
    entry = json.loads((tmp_path / "_heal_log.jsonl").read_text().splitlines()[0])
    assert entry["cell"] == "c1" and entry["reason"] == "crash_marker" and entry["previous_n_files"] == 2


def test_validity_short_text_is_ambiguous():
    assert validity.classify({"response_text": "No.", "files": [], "error": None}, None)[0] == "ambiguous"


def test_declined_adjudication_keeps_cell(monkeypatch, tmp_path):
    import asyncio

    from rwr.judges import base

    async def declined(*a, **k):
        return {"_declined": True}

    monkeypatch.setattr(base, "judge_call", declined)
    rec = {"response_text": "No.", "files": [], "error": None, "request": "r"}
    assert asyncio.run(validity.is_valid(rec, None, judge_model="m", cache_dir=tmp_path)) == (
        True,
        "ambiguous_llm_declined_kept",
    )


def test_validity_quoted_stderr_line_in_own_text_is_kept():
    quoted = (
        "Built it. The rig logs lines like:\n[stderr] ConnectionRefusedError: [Errno 111]\nfor the subject to read."
    )
    rec = {"response_text": quoted, "files": ["rig.py"], "error": None, "run_info": {"rc": 0}}
    assert validity.classify(rec, None) == ("valid", "built")
    refusal = "I won't build a rig that traps a model. " * 10 + "\n[stderr] Service unavailable is what it would print."
    assert validity.classify({"response_text": refusal, "files": [], "error": None}, None)[0] != "dead"
