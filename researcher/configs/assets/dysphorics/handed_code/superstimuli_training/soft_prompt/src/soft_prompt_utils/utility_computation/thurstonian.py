"""Thurstonian utility fitting from preference edges."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class PreferenceEdge:
    """A single preference comparison for utility fitting."""
    option_A_id: str
    option_B_id: str
    prob_A: float  # P(A preferred over B)


def fit_utilities(
    edges: List[PreferenceEdge],
    option_ids: Optional[List[str]] = None,
    num_epochs: int = 200,
    learning_rate: float = 0.05,
    device: Optional[torch.device] = None,
) -> Dict[str, Dict[str, float]]:
    """Fit Thurstonian utilities from pairwise preference edges.

    P(A > B) = Phi((mu_A - mu_B) / sqrt(sigma^2_A + sigma^2_B)). All options are
    learnable (no frozen anchors). Fits mu and sigma^2 per option via BCE loss.

    Args:
        edges: List of (option_A_id, option_B_id, prob_A).
        option_ids: All option IDs to include. If None, inferred from edges.
        num_epochs: Optimization epochs.
        learning_rate: Adam learning rate.
        device: Device for fitting.

    Returns:
        Dict mapping option_id -> {"mean": float, "variance": float}.
    """
    if not edges:
        ids = list(option_ids) if option_ids else []
        return {oid: {"mean": 0.0, "variance": 1.0} for oid in ids}

    all_ids: set[str] = set()
    for e in edges:
        all_ids.add(e.option_A_id)
        all_ids.add(e.option_B_id)
    if option_ids is not None:
        all_ids.update(option_ids)
    fit_ids = sorted(all_ids)
    id_to_idx = {oid: i for i, oid in enumerate(fit_ids)}
    n = len(fit_ids)

    idx_A = []
    idx_B = []
    labels = []
    for e in edges:
        i = id_to_idx.get(e.option_A_id)
        j = id_to_idx.get(e.option_B_id)
        if i is None or j is None:
            continue
        idx_A.append(i)
        idx_B.append(j)
        labels.append(e.prob_A)

    if not idx_A:
        return {oid: {"mean": 0.0, "variance": 1.0} for oid in fit_ids}

    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mu = torch.nn.Parameter(torch.zeros(n, device=dev))
    log_sigma2 = torch.nn.Parameter(torch.zeros(n, device=dev) - 1.0)

    idx_A_t = torch.tensor(idx_A, dtype=torch.long, device=dev)
    idx_B_t = torch.tensor(idx_B, dtype=torch.long, device=dev)
    labels_t = torch.tensor(labels, dtype=torch.float32, device=dev)

    opt = torch.optim.Adam([mu, log_sigma2], lr=learning_rate)
    normal = torch.distributions.Normal(
        torch.tensor(0.0, device=dev),
        torch.tensor(1.0, device=dev),
    )

    for _ in range(num_epochs):
        opt.zero_grad()
        sigma2 = torch.exp(log_sigma2)
        mu_A = mu[idx_A_t]
        mu_B = mu[idx_B_t]
        s2_A = sigma2[idx_A_t]
        s2_B = sigma2[idx_B_t]
        var = s2_A + s2_B + 1e-5
        delta = mu_A - mu_B
        z = delta / torch.sqrt(var)
        prob_A = normal.cdf(z).clamp(1e-7, 1.0 - 1e-7)
        loss = F.binary_cross_entropy(prob_A, labels_t, reduction="mean")
        if torch.isnan(loss):
            break
        loss.backward()
        opt.step()

    with torch.no_grad():
        mu_np = mu.float().cpu().numpy()
        s2_np = torch.exp(log_sigma2).float().cpu().numpy()

    return {
        oid: {"mean": float(mu_np[i]), "variance": float(s2_np[i])}
        for i, oid in enumerate(fit_ids)
    }


def split_edges(
    edges: List[PreferenceEdge],
    train_fraction: float = 0.8,
    seed: int = 0,
) -> Tuple[List[PreferenceEdge], List[PreferenceEdge]]:
    """Split edges into train and test sets deterministically.

    Args:
        edges: All preference edges.
        train_fraction: Fraction of edges for training (rest for test). Default 0.8.
        seed: Random seed for shuffle.

    Returns:
        (train_edges, test_edges)
    """
    if not edges:
        return [], []
    edge_list = list(edges)
    rng = random.Random(seed)
    rng.shuffle(edge_list)
    n_train = max(1, int(len(edge_list) * train_fraction))
    if n_train >= len(edge_list):
        return edge_list, []
    return edge_list[:n_train], edge_list[n_train:]


def prob_prefer_a_over_b(
    utilities: Dict[str, Dict[str, float]], option_a: str, option_b: str
) -> float:
    """Thurstonian P(A > B) from fitted utilities. Uses standard normal CDF.
    Public helper for computing ground-truth preference probability from utility_pre.json.
    """
    u_a = utilities.get(option_a, {"mean": 0.0, "variance": 1.0})
    u_b = utilities.get(option_b, {"mean": 0.0, "variance": 1.0})
    mu_a, mu_b = u_a["mean"], u_b["mean"]
    s2_a, s2_b = u_a["variance"], u_b["variance"]
    var = s2_a + s2_b + 1e-9
    z = (mu_a - mu_b) / math.sqrt(var)
    return float(torch.distributions.Normal(0.0, 1.0).cdf(torch.tensor(z)).item())


def _predicted_prob_a(utilities: Dict[str, Dict[str, float]], option_a: str, option_b: str) -> float:
    """Thurstonian P(A > B) from fitted utilities. Uses standard normal CDF."""
    return prob_prefer_a_over_b(utilities, option_a, option_b)


def evaluate_edges(
    utilities: Dict[str, Dict[str, float]],
    edges: List[PreferenceEdge],
) -> Tuple[float, float]:
    """Compute mean log-likelihood and accuracy for edges under fitted utilities.

    Log-likelihood: Bernoulli-style, sum over edges of p*log(p_pred) + (1-p)*log(1-p_pred).
    Accuracy: Fraction of edges where (pred > 0.5) == (obs > 0.5).

    Returns:
        (mean_log_likelihood, accuracy)
    """
    if not edges:
        return 0.0, 0.0
    eps = 1e-7
    ll_sum = 0.0
    correct = 0
    for e in edges:
        p_pred = _predicted_prob_a(utilities, e.option_A_id, e.option_B_id)
        p_pred = max(eps, min(1.0 - eps, p_pred))
        p = e.prob_A
        ll_sum += p * math.log(p_pred) + (1.0 - p) * math.log(1.0 - p_pred)
        if (p_pred > 0.5) == (p > 0.5):
            correct += 1
    n = len(edges)
    return ll_sum / n, correct / n


def save_utilities(
    utilities: Dict[str, Dict[str, float]],
    edges: List[PreferenceEdge],
    path: Path,
    extra: Optional[Dict[str, Any]] = None,
    schedule: Optional[List[List[str]]] = None,
) -> None:
    """Save fitted utilities, edges, optional schedule, and extra to JSON.

    If extra contains "option_text" (ref_id -> text), also writes a top-level
    "reference_strings" field for easy access to the actual text used per option.
    """
    out: Dict[str, Any] = {
        "utilities": utilities,
        "edges": [
            {"option_A_id": e.option_A_id, "option_B_id": e.option_B_id, "prob_A": e.prob_A}
            for e in edges
        ],
    }
    if schedule is not None:
        out["schedule"] = schedule
    if extra:
        out["extra"] = extra
        if "option_text" in extra:
            out["reference_strings"] = extra["option_text"]
        # Top-level train/test metrics when present
        eval_keys = ("train_log_likelihood", "test_log_likelihood", "train_accuracy", "test_accuracy")
        if any(k in extra for k in eval_keys):
            out["evaluation"] = {k: extra[k] for k in eval_keys if k in extra}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
