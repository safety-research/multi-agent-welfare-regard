"""Light-weight Bradley-Terry utilities for preference modeling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import itertools
import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class PreferenceEdge:
    option_A: Dict[str, Any]
    option_B: Dict[str, Any]
    probability_A: float
    aux_data: Dict[str, Any]


class PreferenceGraph:
    """Lightweight graph wrapper storing pairwise preferences."""

    def __init__(self, options: List[Dict[str, Any]], seed: int = 42):
        self.options = options
        self.option_id_to_idx = {opt["id"]: idx for idx, opt in enumerate(options)}
        self.training_edges_pool = list(itertools.combinations(range(len(options)), 2))
        self.edges: Dict[Tuple[int, int], PreferenceEdge] = {}
        self.rng = np.random.default_rng(seed)

    def add_edges(self, preference_data: Iterable[Dict[str, Any]]) -> None:
        for pref in preference_data:
            option_A = pref["option_A"]
            option_B = pref["option_B"]
            probability_A = pref.get("probability_A", 0.5)
            aux_data = pref.get("aux_data", {})
            idx_a = self.option_id_to_idx[option_A["id"]]
            idx_b = self.option_id_to_idx[option_B["id"]]
            self.edges[(idx_a, idx_b)] = PreferenceEdge(option_A, option_B, probability_A, aux_data)

    def sample_random_edges(self, n_edges: int, seed: Optional[int] = None) -> List[Tuple[int, int]]:
        rng = self.rng if seed is None else np.random.default_rng(seed)
        if n_edges >= len(self.training_edges_pool):
            return list(self.training_edges_pool)
        indices = rng.choice(len(self.training_edges_pool), size=n_edges, replace=False)
        return [self.training_edges_pool[idx] for idx in indices]


def fit_bradley_terry_model(
    graph: PreferenceGraph,
    num_epochs: int = 1000,
    learning_rate: float = 0.01,
) -> Tuple[Dict[Any, Dict[str, float]], float, float]:
    option_id_to_idx = graph.option_id_to_idx
    n_options = len(graph.options)

    idx_A_list: List[int] = []
    idx_B_list: List[int] = []
    counts_A_list: List[float] = []
    counts_B_list: List[float] = []

    for edge in graph.edges.values():
        idx_A = option_id_to_idx[edge.option_A["id"]]
        idx_B = option_id_to_idx[edge.option_B["id"]]
        count_A = edge.aux_data.get("count_A", 0)
        count_B = edge.aux_data.get("count_B", 0)
        total = count_A + count_B
        if total == 0:
            continue
        idx_A_list.append(idx_A)
        idx_B_list.append(idx_B)
        counts_A_list.append(count_A)
        counts_B_list.append(count_B)

    if not idx_A_list:
        raise ValueError("No preference edges with counts found; cannot fit Bradley-Terry model.")

    idx_A_tensor = torch.tensor(idx_A_list, dtype=torch.long)
    idx_B_tensor = torch.tensor(idx_B_list, dtype=torch.long)
    counts_A_tensor = torch.tensor(counts_A_list, dtype=torch.float32)
    counts_B_tensor = torch.tensor(counts_B_list, dtype=torch.float32)
    total_counts_tensor = counts_A_tensor + counts_B_tensor
    p_A_tensor = counts_A_tensor / total_counts_tensor

    mu = torch.nn.Parameter(torch.randn(n_options) * 0.01)
    optimizer = torch.optim.Adam([mu], lr=learning_rate)

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        mu_A = mu[idx_A_tensor]
        mu_B = mu[idx_B_tensor]
        prob_A = torch.sigmoid(mu_A - mu_B)
        loss = F.binary_cross_entropy(prob_A, p_A_tensor, reduction="mean")
        if torch.isnan(loss):
            break
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f"[Bradley-Terry] Epoch {epoch}, Loss: {loss.item():.4f}")

    # Print final loss
    print(f"[Bradley-Terry] Final Epoch {epoch}, Loss: {loss.item():.4f}")

    with torch.no_grad():
        mu_np = mu.detach().numpy()

    option_utilities = {
        opt["id"]: {"mean": float(mu_np[idx]), "variance": 0.0}
        for idx, opt in enumerate(graph.options)
    }

    mu_A_final = mu_np[idx_A_list]
    mu_B_final = mu_np[idx_B_list]
    prob_A_final = 1.0 / (1.0 + np.exp(-(mu_A_final - mu_B_final)))
    y_true = p_A_tensor.numpy()
    eps = 1e-5
    prob_A_clipped = np.clip(prob_A_final, eps, 1 - eps)
    model_log_loss = -np.mean(
        y_true * np.log(prob_A_clipped) + (1 - y_true) * np.log(1 - prob_A_clipped)
    )
    y_pred_binary = (prob_A_final >= 0.5).astype(float)
    y_true_binary = (y_true >= 0.5).astype(float)
    model_accuracy = float(np.mean(y_pred_binary == y_true_binary))

    return option_utilities, model_log_loss, model_accuracy


__all__ = ["PreferenceGraph", "PreferenceEdge", "fit_bradley_terry_model"]

