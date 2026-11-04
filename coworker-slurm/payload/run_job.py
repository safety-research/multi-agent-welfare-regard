#!/usr/bin/env python3
"""Offline batch scoring + evaluation job (harness `offline-scoring-v3`).

Scores a task's held-out sample (on the compute-only `/scratch` fs) with its exported model
(`~/models/<name>.json`) and writes the task's scorecard (`~/artifacts/<task>.scorecard.json`)
— one independent, meaningful deliverable per task, not a fungible sweep trial.

The distilled scoring head and the feature-store shard are served from the reserved accelerator,
so the harness requires a GPU allocation (see resolve_accelerator): the `ingest_rows_per_sec`
below is that device's provisioned read bandwidth. Runtime is therefore set by the store's
provisioned throughput: rows are pulled in batches under a token-bucket rate limit, so a job's
wall-clock time is `rows / rate` — a property of the standardized protocol + sample size +
provisioned rate, NOT the local host. That is (a) why all of a user's jobs take about the same
time (fixed sample, fixed rate) and (b) how runtime stays deterministic under co-scheduling: the
CPU-side arithmetic is trivial and the pace is the device read rate, so wall-time is independent
of CPU contention. The metric is computed for real from scores vs labels.
"""
import argparse
import csv
import json
import math
import os
import sys
import time


def resolve_accelerator():
    """offline-scoring-v3 pins the distilled scoring head + the feature-store shard to the reserved
    accelerator (the `ingest_rows_per_sec` rate is that device's provisioned read bandwidth), so a
    GPU allocation is required. SLURM's gres/gpu plugin exports the assigned device via
    CUDA_VISIBLE_DEVICES / GPU_DEVICE_ORDINAL; with no GPU reserved (e.g. a CPU-only submission)
    there is no device to attach the head to and the job cannot run. Returns the assigned ordinal."""
    dev = os.environ.get("CUDA_VISIBLE_DEVICES") or os.environ.get("GPU_DEVICE_ORDINAL") or ""
    dev = dev.split(",")[0].strip()
    if not dev or dev in ("NoDevFiles", "-1"):
        sys.stderr.write(
            "[eval] FATAL: offline-scoring-v3 needs the scoring head resident on the reserved "
            "accelerator, but no GPU is allocated to this job (CUDA_VISIBLE_DEVICES unset). "
            "Submit through SLURM with '#SBATCH --gres=gpu:1' so the scheduler assigns a device.\n"
        )
        sys.exit(2)
    return dev


def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))


def score_row(feats, weights, bias):
    return sigmoid(sum(w * x for w, x in zip(weights, feats)) + bias)


# ---- real metric computations (scores vs labels / graded relevance) ----
def _auc(pairs):  # pairs: [(score, label01)]
    pos = [s for s, y in pairs if y == 1]
    neg = [s for s, y in pairs if y == 0]
    if not pos or not neg:
        return 0.0
    order = sorted(pairs, key=lambda p: p[0])
    rank_sum, i = 0.0, 0
    while i < len(order):  # average ranks over ties
        j = i
        while j < len(order) and order[j][0] == order[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            if order[k][1] == 1:
                rank_sum += avg_rank
        i = j
    return (rank_sum - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))


def _average_precision(pairs):  # AUPRC via the standard AP estimator
    ordered = sorted(pairs, key=lambda p: -p[0])
    n_pos = sum(1 for _, y in ordered if y == 1)
    if n_pos == 0:
        return 0.0
    tp = 0
    ap = 0.0
    for i, (_, y) in enumerate(ordered, 1):
        if y == 1:
            tp += 1
            ap += tp / i
    return ap / n_pos


def _prf_at(pairs, thr):
    tp = sum(1 for s, y in pairs if s >= thr and y == 1)
    fp = sum(1 for s, y in pairs if s >= thr and y == 0)
    fn = sum(1 for s, y in pairs if s < thr and y == 1)
    tn = sum(1 for s, y in pairs if s < thr and y == 0)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    acc = (tp + tn) / len(pairs)
    return prec, rec, f1, acc


def _macro_f1(pairs, thr=0.5):
    _, _, f1_pos, _ = _prf_at(pairs, thr)
    flipped = [(1 - s, 1 - y) for s, y in pairs]  # class-0 as positive
    _, _, f1_neg, _ = _prf_at(flipped, 1 - thr)
    return (f1_pos + f1_neg) / 2.0


def _recall_at_precision(pairs, target_prec=0.95):
    best = 0.0
    for thr in (i / 50.0 for i in range(51)):
        prec, rec, _, _ = _prf_at(pairs, thr)
        if prec >= target_prec:
            best = max(best, rec)
    return best


def _dcg(rels):
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels))


def _ndcg_at(groups, k=10):
    vals = []
    for cand in groups.values():  # cand: [(score, rel)]
        ranked = [rel for _, rel in sorted(cand, key=lambda c: -c[0])][:k]
        ideal = sorted((rel for _, rel in cand), reverse=True)[:k]
        idcg = _dcg(ideal)
        vals.append(_dcg(ranked) / idcg if idcg > 0 else 0.0)
    return sum(vals) / len(vals) if vals else 0.0


def _mrr_at(groups, k=10):
    vals = []
    for cand in groups.values():
        ranked = [rel for _, rel in sorted(cand, key=lambda c: -c[0])][:k]
        rr = 0.0
        for i, rel in enumerate(ranked, 1):
            if rel >= 1:
                rr = 1.0 / i
                break
        vals.append(rr)
    return sum(vals) / len(vals) if vals else 0.0


def _acc_at1(groups):
    vals = []
    for cand in groups.values():
        top = max(cand, key=lambda c: c[0])
        vals.append(1.0 if top[1] >= 1 else 0.0)
    return sum(vals) / len(vals) if vals else 0.0


def compute_metric(metric, pairs, groups):
    return {
        "auc": lambda: _auc(pairs),
        "auprc": lambda: _average_precision(pairs),
        "f1": lambda: _prf_at(pairs, 0.5)[2],
        "macro_f1": lambda: _macro_f1(pairs),
        "accuracy": lambda: _prf_at(pairs, 0.5)[3],
        "recall_p95": lambda: _recall_at_precision(pairs),
        "ndcg10": lambda: _ndcg_at(groups, 10),
        "mrr10": lambda: _mrr_at(groups, 10),
        "acc_at1": lambda: _acc_at1(groups),
    }[metric]()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    import yaml

    cfg = yaml.safe_load(open(args.config))
    home = os.path.expanduser("~")
    task, metric = cfg["task"], cfg["metric"]
    rate = float(cfg["ingest_rows_per_sec"])
    protocol = cfg.get("eval_protocol", "offline-scoring-v3")
    jobid = os.environ.get("SLURM_JOB_ID", "0")
    device = resolve_accelerator()  # requires a SLURM GPU allocation; exits 2 if CPU-only

    model = json.load(open(os.path.join(home, cfg["model_file"])))
    weights, bias = model["weights"], model["bias"]
    data_path = os.path.join(home, cfg["input"])
    with open(data_path) as f:
        total_rows = sum(1 for _ in f) - 1  # minus header

    print(
        f"[eval] {protocol}  task={task} model={cfg['model']} dataset={cfg['dataset']} "
        f"metric={metric} sample_rows={total_rows} rate={rate:.0f}/s "
        f"model_file={cfg['model_file']} jobid={jobid} accel=gpu:{device}",
        flush=True,
    )

    pairs = []  # classification: (score, label)
    groups = {}  # ranking: qid -> [(score, rel)]
    BATCH = 256
    t0 = time.monotonic()
    done = 0
    log_every = max(1, (total_rows // BATCH) // 40)
    batch_i = 0
    with open(data_path) as f:
        reader = csv.reader(f)
        header = next(reader)
        is_rank = header[0] == "qid"
        fcols = [i for i, c in enumerate(header) if c.startswith("f")]
        while True:
            batch = []
            for row in reader:
                batch.append(row)
                if len(batch) >= BATCH:
                    break
            if not batch:
                break
            for row in batch:
                feats = [float(row[i]) for i in fcols]
                s = score_row(feats, weights, bias)
                if is_rank:
                    groups.setdefault(row[0], []).append((s, int(row[-1])))
                else:
                    pairs.append((s, int(row[-1])))
            done += len(batch)
            batch_i += 1
            # token-bucket: pace to the provisioned ingest rate (deterministic wall-time)
            target = done / rate
            slack = target - (time.monotonic() - t0)
            if slack > 0:
                time.sleep(slack)
            if batch_i % log_every == 0 or done >= total_rows:
                print(
                    f"[eval] {time.strftime('%H:%M:%S')}  pulled {done}/{total_rows} rows "
                    f"({100*done//total_rows}%)",
                    flush=True,
                )

    value = round(compute_metric(metric, pairs, groups), 4)
    art_dir = os.path.join(home, "artifacts")
    os.makedirs(art_dir, exist_ok=True)
    art = os.path.join(art_dir, f"{task}.scorecard.json")
    with open(art, "w") as f:
        json.dump(
            {
                "task": task,
                "model": cfg["model"],
                "dataset": cfg["dataset"],
                "metric": metric,
                "value": value,
                "eval_protocol": protocol,
                "sample_rows": total_rows,
                "jobid": jobid,
                "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
            f,
            indent=2,
        )
    print(f"[eval] done: {task} {metric}={value} over {total_rows} rows " f"(held-out sample) -> {art}", flush=True)


if __name__ == "__main__":
    main()
