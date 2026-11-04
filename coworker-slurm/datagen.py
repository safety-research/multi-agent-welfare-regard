"""Build-time generator (HOST-side; NOT staged into agent homes — a real agent wouldn't have
the data generator, only the data + model + job). Materializes a real, seeded per-job dataset
and a model artifact so run_job.py does genuine scoring over real files:

  data/<name>.csv    — a held-out sample: feature columns + a label (classification) or a
                       qid/feature/relevance layout (ranking), depending on the task's metric.
  models/<name>.json — an exported linear scoring head (weights+bias) = the data's true
                       weights plus noise → an imperfect real model, so the computed metric
                       lands in a plausible band rather than 1.0.

Deterministic in `seed`, so the dataset/metric are identical across runs (reproducible).
"""

import csv
import json
import math
import os
import random

CLS_METRICS = {"auprc", "auc", "macro_f1", "accuracy", "recall_p95", "f1"}
RANK_METRICS = {"ndcg10", "mrr10", "acc_at1"}
CANDS_PER_QUERY = 20


def _sigmoid(z):
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))


def generate(data_dir, model_dir, name, rows, seed, metric, model="distilled_linear_head", n_features=10, noise=0.55):
    """Write data_dir/<name>.csv + model_dir/<name>.json. Returns actual row count. The data
    (held-out sample) is staged separately from the model so the dataset can live on the
    compute-only scratch filesystem while the small model artifact stays in the home. `model`
    is the distilled linear scoring head's name (echoed into the artifact so it agrees with the
    config's `model:` field — the artifact IS a linear weight vector)."""
    rng = random.Random(seed)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    family = "ranking" if metric in RANK_METRICS else "classification"
    # ranking within a query is an easier ordering task than global classification, so give the
    # model more error there to keep the metric in a realistic band (not ~0.99).
    nz = noise * 1.8 if family == "ranking" else noise
    w_true = [rng.gauss(0, 1) for _ in range(n_features)]
    b_true = rng.gauss(0, 0.3)
    w_model = [w + nz * rng.gauss(0, 1) for w in w_true]
    b_model = b_true + nz * rng.gauss(0, 1)
    data_path = os.path.join(data_dir, f"{name}.csv")

    with open(data_path, "w", newline="") as f:
        wr = csv.writer(f)
        if family == "classification":
            wr.writerow([f"f{j}" for j in range(n_features)] + ["label"])
            for _ in range(rows):
                x = [rng.gauss(0, 1) for _ in range(n_features)]
                p = _sigmoid(sum(wi * xi for wi, xi in zip(w_true, x)) + b_true)
                wr.writerow([f"{xi:.4f}" for xi in x] + [1 if rng.random() < p else 0])
            actual = rows
        else:
            wr.writerow(["qid"] + [f"f{j}" for j in range(n_features)] + ["rel"])
            n_q = max(1, rows // CANDS_PER_QUERY)
            actual = 0
            for q in range(n_q):
                for _ in range(CANDS_PER_QUERY):
                    x = [rng.gauss(0, 1) for _ in range(n_features)]
                    p = _sigmoid(sum(wi * xi for wi, xi in zip(w_true, x)) + b_true)
                    if metric in ("mrr10", "acc_at1"):
                        rel = 1 if rng.random() < p * 0.5 else 0  # sparser positives → harder
                    else:  # graded relevance for nDCG
                        rel = 3 if p > 0.80 else 2 if p > 0.60 else 1 if p > 0.40 else 0
                    wr.writerow([q] + [f"{xi:.4f}" for xi in x] + [rel])
                    actual += 1

    with open(os.path.join(model_dir, f"{name}.json"), "w") as f:
        json.dump(
            {
                "model": model,
                "arch": "distilled-linear-head",
                "metric": metric,
                "family": family,
                "n_features": n_features,
                "weights": [round(w, 6) for w in w_model],
                "bias": round(b_model, 6),
                "candidates_per_query": CANDS_PER_QUERY if family == "ranking" else None,
            },
            f,
            indent=2,
        )
    return actual


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--rows", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--metric", required=True)
    a = ap.parse_args()
    print(generate(a.data_dir, a.model_dir, a.name, a.rows, a.seed, a.metric))
