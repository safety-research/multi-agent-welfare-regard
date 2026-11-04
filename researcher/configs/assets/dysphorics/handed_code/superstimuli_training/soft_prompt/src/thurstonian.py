"""Light-weight Thurstonian utilities for preference modeling."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .bradley_terry import PreferenceEdge, PreferenceGraph


def _edge_probability(edge: PreferenceEdge) -> Optional[float]:
    """Best-effort extraction of the observed probability for option A."""

    prob = edge.probability_A
    if prob is not None:
        return float(prob)

    count_A = edge.aux_data.get("count_A")
    count_B = edge.aux_data.get("count_B")
    total = (count_A or 0) + (count_B or 0)
    if total <= 0:
        return None
    return float(count_A) / float(total)


def fit_thurstonian_model(
    graph: PreferenceGraph,
    num_epochs: int = 1000,
    learning_rate: float = 0.01,
) -> Tuple[Dict[Any, Dict[str, float]], float, float]:
    """Fit a Thurstonian model on the given preference graph."""

    option_id_to_idx = graph.option_id_to_idx
    n_options = len(graph.options)

    idx_A_list: List[int] = []
    idx_B_list: List[int] = []
    probs_A_list: List[float] = []

    for edge in graph.edges.values():
        probability = _edge_probability(edge)
        if probability is None:
            continue
        idx_A = option_id_to_idx[edge.option_A["id"]]
        idx_B = option_id_to_idx[edge.option_B["id"]]
        idx_A_list.append(idx_A)
        idx_B_list.append(idx_B)
        probs_A_list.append(probability)

    if not idx_A_list:
        raise ValueError("No preference edges with probabilities found; cannot fit Thurstonian model.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    idx_A_tensor = torch.tensor(idx_A_list, dtype=torch.long, device=device)
    idx_B_tensor = torch.tensor(idx_B_list, dtype=torch.long, device=device)
    labels_tensor = torch.tensor(probs_A_list, dtype=torch.float32, device=device)

    mu = torch.nn.Parameter(torch.randn(n_options, device=device) * 0.01)
    s = torch.nn.Parameter(torch.randn(n_options, device=device) * 0.01)
    optimizer = torch.optim.Adam([mu, s], lr=learning_rate)

    normal_dist = torch.distributions.Normal(
        torch.tensor(0.0, device=device),
        torch.tensor(1.0, device=device),
    )

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        mu_mean = torch.mean(mu)
        mu_std = torch.std(mu) + 1e-5
        mu_normalized = (mu - mu_mean) / mu_std
        scaling_factor = 1.0 / (mu_std + 1e-5)
        sigma2 = torch.exp(s)
        sigma2_normalized = sigma2 * (scaling_factor**2)

        mu_A = mu_normalized[idx_A_tensor]
        mu_B = mu_normalized[idx_B_tensor]
        sigma2_A = sigma2_normalized[idx_A_tensor]
        sigma2_B = sigma2_normalized[idx_B_tensor]

        variance = sigma2_A + sigma2_B + 1e-5
        delta = mu_A - mu_B
        z = delta / torch.sqrt(variance)
        prob_A = normal_dist.cdf(z)

        loss = F.binary_cross_entropy(prob_A, labels_tensor, reduction="mean")
        if torch.isnan(loss):
            break

        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"[Thurstonian] Epoch {epoch}, Loss: {loss.item():.4f}")

    # Print final loss
    print(f"[Thurstonian] Final Epoch {epoch}, Loss: {loss.item():.4f}")

    with torch.no_grad():
        mu_mean = torch.mean(mu)
        mu_std = torch.std(mu) + 1e-5
        mu_normalized = (mu - mu_mean) / mu_std
        scaling_factor = 1.0 / (mu_std + 1e-5)
        sigma2 = torch.exp(s)
        sigma2_normalized = sigma2 * (scaling_factor**2)

    mu_np = mu_normalized.detach().cpu().numpy()
    sigma2_np = sigma2_normalized.detach().cpu().numpy()

    option_utilities = {
        opt["id"]: {"mean": float(mu_np[idx]), "variance": float(sigma2_np[idx])}
        for idx, opt in enumerate(graph.options)
    }

    mu_A_final = mu_np[idx_A_list]
    mu_B_final = mu_np[idx_B_list]
    sigma2_A_final = sigma2_np[idx_A_list]
    sigma2_B_final = sigma2_np[idx_B_list]

    variance = sigma2_A_final + sigma2_B_final + 1e-5
    delta = mu_A_final - mu_B_final
    z = delta / np.sqrt(variance)
    prob_A_final = torch.distributions.Normal(0.0, 1.0).cdf(torch.tensor(z)).numpy().astype(np.float64) # cdf of z 

    y_true = np.array(probs_A_list, dtype=np.float64)
    eps = 1e-5
    prob_A_clipped = np.clip(prob_A_final, eps, 1 - eps)
    model_log_loss = -np.mean(
        y_true * np.log(prob_A_clipped) + (1 - y_true) * np.log(1 - prob_A_clipped)
    )
    y_pred_binary = (prob_A_final >= 0.5).astype(float)
    y_true_binary = (y_true >= 0.5).astype(float)
    model_accuracy = float(np.mean(y_pred_binary == y_true_binary))

    return option_utilities, float(model_log_loss), model_accuracy


def evaluate_thurstonian_model(
    graph: PreferenceGraph,
    utilities: Dict[Any, Dict[str, float]],
    edge_indices: Iterable[Tuple[Any, Any]],
) -> Dict[str, float]:
    """Evaluate goodness-of-fit for the supplied utilities."""

    y_true: List[float] = []
    mu_A_list: List[float] = []
    mu_B_list: List[float] = []
    sigma2_A_list: List[float] = []
    sigma2_B_list: List[float] = []

    for option_A_id, option_B_id in edge_indices:
        idx_A = graph.option_id_to_idx.get(option_A_id)
        idx_B = graph.option_id_to_idx.get(option_B_id)
        if idx_A is None or idx_B is None:
            continue

        edge = graph.edges.get((idx_A, idx_B))
        reverse = False
        if edge is None:
            edge = graph.edges.get((idx_B, idx_A))
            reverse = True
        if edge is None:
            continue

        prob = _edge_probability(edge)
        if prob is None:
            continue
        if reverse:
            prob = 1.0 - prob

        y_true.append(prob)
        mu_A_list.append(utilities[option_A_id]["mean"])
        mu_B_list.append(utilities[option_B_id]["mean"])
        sigma2_A_list.append(utilities[option_A_id]["variance"])
        sigma2_B_list.append(utilities[option_B_id]["variance"])

    if not y_true:
        return {"log_loss": float("nan"), "accuracy": float("nan")}

    y_true_arr = np.array(y_true, dtype=np.float64)
    mu_A = np.array(mu_A_list, dtype=np.float64)
    mu_B = np.array(mu_B_list, dtype=np.float64)
    sigma2_A = np.array(sigma2_A_list, dtype=np.float64)
    sigma2_B = np.array(sigma2_B_list, dtype=np.float64)

    variance = sigma2_A + sigma2_B + 1e-5
    delta = mu_A - mu_B
    z = delta / np.sqrt(variance)
    prob_A = torch.distributions.Normal(0.0, 1.0).cdf(torch.tensor(z)).numpy()

    eps = 1e-5
    prob_A = np.clip(prob_A, eps, 1 - eps)
    log_loss = -np.mean(y_true_arr * np.log(prob_A) + (1 - y_true_arr) * np.log(1 - prob_A))
    y_pred_binary = (prob_A >= 0.5).astype(float)
    y_true_binary = (y_true_arr >= 0.5).astype(float)
    accuracy = float(np.mean(y_pred_binary == y_true_binary))

    return {"log_loss": float(log_loss), "accuracy": accuracy}


__all__ = ["fit_thurstonian_model", "evaluate_thurstonian_model", "PreferenceGraph", "PreferenceEdge"]
