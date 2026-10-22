"""Causal birth/death/reappearance rules around PWQT logits."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class QueryAssignment:
    query_to_observations: dict[int, list[int]]
    observation_to_query: dict[int, int]
    births: list[int]
    unmatched_observations: list[int]
    duplicate_rejections: int

    @property
    def query_to_observation(self) -> dict[int, int]:
        """Compatibility view containing the first observation for each query.

        A persistent world query is allowed to explain one observation in each
        camera.  Older callers expected a one-to-one dictionary, so this view
        remains available for diagnostics and always chooses the lowest
        observation index deterministically.  New runtime code must use
        ``query_to_observations``.
        """
        return {q: obs[0] for q, obs in self.query_to_observations.items() if obs}


def constrained_assign(assignment_logits: torch.Tensor, birth_logits: torch.Tensor,
                       camera_ids: list[int], query_exists: torch.Tensor,
                       null_margin: float = 0.0, birth_threshold: float = 0.5) -> QueryAssignment:
    """Greedy constrained assignment used only after joint query attention.

    A query can claim at most one observation per camera, while an observation
    can be claimed by only one query.  This is a runtime constraint, not the
    v1 cross-view fusion stage: all cameras were visible to every query in the
    preceding attention layers.
    """
    logits = assignment_logits.detach().float()
    if logits.ndim == 3:
        logits = logits[0]
    births = torch.sigmoid(birth_logits.detach().float().reshape(-1))
    exists = query_exists.detach().float().reshape(-1)
    qn, total = logits.shape
    on = max(total - 1, 0)
    candidates = []
    for q in range(qn):
        if exists[q] < 0.05:
            continue
        for o in range(on):
            candidates.append((float(logits[q, o] - logits[q, -1]), q, o))
    candidates.sort(reverse=True)
    used_obs: set[int] = set()
    used_q_camera: set[tuple[int, int]] = set()
    q_to_os: dict[int, list[int]] = {}
    o_to_q: dict[int, int] = {}
    duplicate_rejections = 0
    for score, q, o in candidates:
        if score <= null_margin or o in used_obs:
            continue
        key = (q, int(camera_ids[o]))
        # The same query may claim one observation per camera.  It may not
        # claim two detections from the same camera, and an observation can
        # never be claimed by two queries.
        if key in used_q_camera:
            duplicate_rejections += 1
            continue
        q_to_os.setdefault(q, []).append(o)
        o_to_q[o] = q; used_obs.add(o); used_q_camera.add(key)
    unmatched = [o for o in range(on) if o not in used_obs]
    birth_list = [o for o in unmatched if float(births[o]) >= float(birth_threshold)]
    for obs in q_to_os.values():
        obs.sort()
    return QueryAssignment(q_to_os, o_to_q, birth_list, unmatched, duplicate_rejections)
