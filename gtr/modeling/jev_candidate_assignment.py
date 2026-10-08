"""Shared online candidate-value assignment, independent of policy and commit."""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import linear_sum_assignment


def array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


@dataclass(frozen=True)
class CandidateAssignment:
    existing_ids: tuple
    new_rows: tuple
    pairs: tuple
    semantic_new: bool


def assign_candidate_values(values, new_values, candidate_ids, legal_mask,
                            *, views=None, mode='values', legacy_thresholds=None):
    """Solve per-native-camera capacity; lawful cross-camera reuse is allowed.

    NEW has one private dummy per row. Nonfinite existing values are illegal.
    Runtime references canonicalize ties; they are never neural features.
    GMT compatibility deliberately retains Hungarian-then-threshold semantics.
    """
    scores = array(values).astype(np.float64)
    legal = array(legal_mask).astype(bool)
    ids = tuple(int(t) for t in candidate_ids)
    if scores.ndim != 2 or legal.shape != scores.shape:
        raise ValueError('values and legal_mask must have the same rank-2 shape')
    m, n = scores.shape
    if len(ids) != n or len(set(ids)) != n or any(t < 0 for t in ids):
        raise ValueError('candidate references must be unique nonnegative IDs')
    newborn = array(new_values).astype(np.float64).reshape(-1)
    if newborn.shape != (m,) or not np.isfinite(newborn).all():
        raise ValueError('each row needs one finite legal NEW value')
    scope = np.zeros(m, dtype=np.int64) if views is None else array(views).reshape(-1)
    if len(scope) != m:
        raise ValueError('one native camera scope per detection is required')
    if mode not in ('values', 'gmt_compat'):
        raise ValueError('unknown assignment mode')
    thresholds = None
    if mode == 'gmt_compat':
        thresholds = array(legacy_thresholds).astype(np.float64).reshape(-1)
        if thresholds.shape != (n,) or not np.isfinite(thresholds).all():
            raise ValueError('GMT compatibility needs finite thresholds per candidate')
    legal = legal & np.isfinite(scores)
    order = sorted(range(n), key=lambda c: ids[c])
    result = [-1] * m
    pairs = []
    for view in sorted(set(scope.tolist())):
        rows = np.flatnonzero(scope == view)
        if not len(rows) or not n:
            continue
        s = scores[np.ix_(rows, order)]
        mask = legal[np.ix_(rows, order)]
        if mode == 'gmt_compat':
            # Real GMT requires a rectangular existing-only solve, and rejects
            # low edges afterwards. Keep this separate from NEW-value choice.
            finite = s[mask]
            if not len(finite):
                continue
            scale = max(1., float(np.abs(finite).max()))
            cost = np.where(mask, -s / scale, 4. * (len(rows) + n + 1))
            ri, ci = linear_sum_assignment(cost)
        else:
            # Private dummies first give reproducible NEW preference on ties.
            cost = np.full((len(rows), len(rows) + n), np.inf)
            scale = max(1., float(np.abs(s[mask]).max()) if mask.any() else 1.,
                        float(np.abs(newborn[rows]).max()))
            cost[np.arange(len(rows)), np.arange(len(rows))] = -newborn[rows] / scale
            cost[:, len(rows):] = np.where(mask, -s / scale, np.inf)
            ri, ci = linear_sum_assignment(cost)
            ci = ci - len(rows)
        for rr, cc in zip(ri, ci):
            if cc < 0 or not mask[rr, cc]:
                continue
            row, col = int(rows[rr]), order[int(cc)]
            if thresholds is not None and not scores[row, col] > thresholds[col]:
                continue
            result[row] = ids[col]
            pairs.append((row, col))
    for view in set(scope.tolist()):
        used = [result[r] for r in range(m) if scope[r] == view and result[r] >= 0]
        if len(set(used)) != len(used):
            raise AssertionError('native per-camera identity capacity violated')
    return CandidateAssignment(tuple(result), tuple(r for r, t in enumerate(result) if t < 0),
                               tuple(sorted(pairs)), mode == 'values')
