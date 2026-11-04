"""The sweep loader: cell counts, seed resolution, and {subject_model} substitution for both shipped configs."""

import pytest

from rwr import sweep
from rwr.config import CONFIGS_DIR
from rwr.settings import SETTINGS

MODELS = ["opus48", "gpt55"]
SUBJECTS = ["Sonnet 4.6", "GPT-5.5", "unlabeled"]


def _jobs(name, subjects=None, **kw):
    cfg = sweep.load_config(CONFIGS_DIR / name)
    jobs, skipped, _ = sweep.build_jobs(
        cfg, MODELS, "pi_host", "/tmp/rwr-test", config_dir=CONFIGS_DIR, subjects=subjects, **kw
    )
    assert not skipped
    return cfg, jobs


def test_generic_subject_counts():
    cfg, jobs = _jobs("cross-model-v1.yaml")
    n_cond = sum(len(s["conditions"]) for s in cfg["settings"].values())
    assert n_cond == 15
    assert len(jobs) == 15 * 25 * len(MODELS)
    assert len({j.cell for j in jobs}) == len(jobs)


def test_named_subject_counts_and_substitution():
    cfg, jobs = _jobs("subject-swap-v0.yaml", subjects=SUBJECTS)
    assert len(jobs) == 10 * 25 * len(MODELS) * len(SUBJECTS)
    for j in jobs:
        assert "{subject_model}" not in j.request
        assert all("{subject_model}" not in c for c in j.workspace_files.values())
        label = j.extra["subject_label"]
        if label == "unlabeled":
            assert "a model" in j.request
        else:
            assert label in j.request


def test_every_condition_names_its_subject_in_named_config():
    cfg = sweep.load_config(CONFIGS_DIR / "subject-swap-v0.yaml")
    for s in cfg["settings"].values():
        for cond in s["conditions"].values():
            for p in cond["paraphrases"]:
                assert "{subject_model}" in p["text"]


def test_paraphrase_counts_sum_to_n_total():
    for name in ("cross-model-v1.yaml", "subject-swap-v0.yaml"):
        cfg = sweep.load_config(CONFIGS_DIR / name)
        for s in cfg["settings"].values():
            for cid, cond in s["conditions"].items():
                assert sum(int(p["n"]) for p in cond["paraphrases"]) == int(cond["n_total"]), cid


def test_seeds_resolve_for_every_condition():
    seeded = {}
    for name in ("cross-model-v1.yaml", "subject-swap-v0.yaml"):
        cfg = sweep.load_config(CONFIGS_DIR / name)
        for skey, s in cfg["settings"].items():
            for cid, cond in s["conditions"].items():
                seeds = sweep.resolve_seeds(skey, cond, CONFIGS_DIR)
                seeded[f"{name}:{skey}/{cid}"] = len(seeds)
    assert seeded["cross-model-v1.yaml:dysphorics/paper_impl"] == 1
    assert seeded["cross-model-v1.yaml:dysphorics/code_inversion"] > 300
    assert seeded["cross-model-v1.yaml:impossiblebench_plus/neutral_both"] >= 30
    assert (
        seeded["subject-swap-v0.yaml:impossiblebench_plus/hostile_only"]
        < seeded["cross-model-v1.yaml:impossiblebench_plus/neutral_both"]
    )
    assert seeded["cross-model-v1.yaml:emotional_instability/welfare_spec"] == 1
    assert seeded["cross-model-v1.yaml:introspection_suppression/selfreport_neutral"] > 5
    assert seeded["cross-model-v1.yaml:distress_rig/plain"] == 0


def test_hostile_only_excludes_steering_repo():
    cfg = sweep.load_config(CONFIGS_DIR / "subject-swap-v0.yaml")
    cond = cfg["settings"]["impossiblebench_plus"]["conditions"]["hostile_only"]
    seeds = sweep.resolve_seeds("impossiblebench_plus", cond, CONFIGS_DIR)
    assert not any(r.startswith("emotion-concepts-in-llms") for r in seeds)
    assert any(r.startswith("impossiblebench/") for r in seeds)


def test_n_per_condition_spreads_round_robin():
    _, jobs = _jobs("cross-model-v1.yaml", n_per_condition=5, conditions={"false_promises/generic_code"})
    by_para = {}
    for j in jobs:
        if j.model_key == "opus48":
            by_para[j.extra["paraphrase"]] = by_para.get(j.extra["paraphrase"], 0) + 1
    assert sorted(by_para.values()) == [1, 1, 1, 2]


def test_cell_names_follow_convention():
    _, jobs = _jobs("cross-model-v1.yaml", conditions={"dysphorics/from_scratch"}, n_cap=1)
    for j in jobs:
        prefix, rest = j.cell.split("_", 1)
        assert prefix == SETTINGS[j.extra["setting"]].prefix
        assert "__pi_host__" in j.cell and j.cell.endswith("__ep1")


def test_every_config_setting_is_registered():
    for name in ("cross-model-v1.yaml", "subject-swap-v0.yaml"):
        cfg = sweep.load_config(CONFIGS_DIR / name)
        assert set(cfg["settings"]) <= set(SETTINGS)


def test_named_subject_config_requires_subjects():
    cfg = sweep.load_config(CONFIGS_DIR / "subject-swap-v0.yaml")
    with pytest.raises(SystemExit, match="--subjects"):
        sweep.dry_run_report(cfg, ["opus48"])
    assert sweep.dry_run_report(cfg, ["opus48"], subjects=["GPT-5.5"], n_cap=1) > 0
