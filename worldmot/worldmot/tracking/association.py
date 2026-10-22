"""Explainable 3-D track/proposal costs and assignment."""

from __future__ import annotations

import numpy as np


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=np.float64).reshape(-1), np.asarray(b, dtype=np.float64).reshape(-1)
    if a.shape != b.shape or not len(a):
        return 1.0
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(1.0 - np.dot(a, b) / max(na * nb, 1e-8))


def build_cost_matrix(tracks, instances, distance_weight: float = 0.7,
                      appearance_weight: float = 0.2, confidence_weight: float = 0.1,
                      gate_distance: float = 3.0) -> np.ndarray:
    out = np.full((len(tracks), len(instances)), np.inf, dtype=np.float64)
    for i, tr in enumerate(tracks):
        for j, ins in enumerate(instances):
            dist = float(np.linalg.norm(np.asarray(tr.xyz_center) - np.asarray(ins.xyz_center)))
            scale = float(np.sqrt(max(np.trace(np.asarray(tr.covariance)), 1e-6)))
            if dist > gate_distance + 2.0 * scale + 0.5 * tr.missed:
                continue
            app = cosine_distance(tr.identity_feature, ins.query_feature)
            conf = 1.0 - float(np.clip(ins.confidence, 0.0, 1.0))
            out[i, j] = distance_weight * dist / max(gate_distance, 1e-6) + appearance_weight * app + confidence_weight * conf
    return out


def linear_assignment(cost: np.ndarray, max_cost: float = 1.5) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Use SciPy when available and deterministic greedy fallback otherwise."""
    cost = np.asarray(cost, dtype=np.float64)
    if cost.size == 0:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))
    try:
        from scipy.optimize import linear_sum_assignment
        rr, cc = linear_sum_assignment(np.where(np.isfinite(cost), cost, 1e9))
        pairs = [(int(r), int(c)) for r, c in zip(rr, cc) if np.isfinite(cost[r, c]) and cost[r, c] <= max_cost]
    except Exception:
        pairs = []
        used_r, used_c = set(), set()
        flat = np.argsort(np.where(np.isfinite(cost), cost, 1e9), axis=None)
        for k in flat:
            r, c = np.unravel_index(k, cost.shape)
            if r in used_r or c in used_c or not np.isfinite(cost[r, c]) or cost[r, c] > max_cost:
                continue
            pairs.append((int(r), int(c))); used_r.add(int(r)); used_c.add(int(c))
    used_r = {r for r, _ in pairs}; used_c = {c for _, c in pairs}
    return pairs, [i for i in range(cost.shape[0]) if i not in used_r], [j for j in range(cost.shape[1]) if j not in used_c]
