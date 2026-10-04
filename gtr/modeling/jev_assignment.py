"""Assignment helpers shared by the GMT/JEV online adapter and tests."""

from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


def _to_numpy(scores) -> np.ndarray:
    """Convert a torch/numpy score matrix without changing its values."""
    if hasattr(scores, "detach"):
        scores = scores.detach().cpu().numpy()
    return np.asarray(scores, dtype=np.float64)


def constrained_hungarian(
    scores,
    banned_edges: Iterable[Tuple[int, int]] = (),
) -> List[Tuple[int, int]]:
    """Maximize a score matrix while excluding explicitly banned edges.

    The matrix remains global: all rows and columns are passed to one
    Hungarian solve.  A selected forbidden/non-finite edge is discarded from
    the returned proposal instead of being silently treated as valid.  This
    gives the caller an explicit unmatched result when a row has no feasible
    edge.
    """

    matrix = _to_numpy(scores)
    if matrix.ndim != 2:
        raise ValueError(f"scores must be a rank-2 matrix, got shape {matrix.shape}")
    row_count, col_count = matrix.shape
    if row_count == 0 or col_count == 0:
        return []

    banned = set()
    for row, col in banned_edges:
        row, col = int(row), int(col)
        if not (0 <= row < row_count and 0 <= col < col_count):
            raise IndexError(f"banned edge ({row}, {col}) is outside {matrix.shape}")
        banned.add((row, col))

    finite = np.isfinite(matrix)
    if not finite.any():
        return []

    # scipy rejects +inf/-inf in the cost matrix.  A floor well below every
    # valid score lets the global solve proceed even when a row is fully
    # masked; that row is then removed from the returned feasible proposal.
    finite_values = matrix[finite]
    span = max(1.0, float(np.max(finite_values) - np.min(finite_values)))
    forbidden_score = float(np.min(finite_values) - span - 1.0)
    masked = np.where(finite, matrix, forbidden_score)
    for row, col in banned:
        masked[row, col] = forbidden_score

    assigned_rows, assigned_cols = linear_sum_assignment(-masked)
    proposals = []
    for row, col in zip(assigned_rows.tolist(), assigned_cols.tolist()):
        if (row, col) in banned or not finite[row, col]:
            continue
        proposals.append((int(row), int(col)))
    return proposals

