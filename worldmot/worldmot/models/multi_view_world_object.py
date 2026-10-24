"""Joint two-view world-object formation without forced nearest pairing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from .world_object_support import WorldObjectSupport, support_compatibility
from ..geometry.object_observation_consistency import merge_supports


@dataclass
class WorldObjectHypothesis:
    object_id: int
    observations: list[object]
    support: WorldObjectSupport
    score: float
    status: str
    diagnostics: dict = field(default_factory=dict)

    @property
    def cameras(self) -> tuple[int, ...]:
        return tuple(sorted({int(x.camera_id) for x in self.observations}))

    @property
    def center(self) -> np.ndarray:
        return np.asarray(self.support.center, dtype=np.float64)


class JointWorldObjectBuilder:
    """Build a set of hypotheses subject to one-observation/one-object rules."""

    def __init__(self, max_pair_distance: float = 3.5, min_pair_score: float = 0.08):
        self.max_pair_distance = float(max_pair_distance)
        self.min_pair_score = float(min_pair_score)

    def build(self, observations: Iterable[object], start_id: int = 1) -> tuple[list[WorldObjectHypothesis], dict]:
        obs = list(observations)
        by_camera: dict[int, list[object]] = {}
        for x in obs:
            by_camera.setdefault(int(x.camera_id), []).append(x)
        views = sorted(by_camera)
        used: set[int] = set()
        hypotheses: list[WorldObjectHypothesis] = []
        candidates = []
        if len(views) >= 2:
            for ai, a in enumerate(by_camera[views[0]]):
                for bi, b in enumerate(by_camera[views[1]]):
                    score, detail, reason = support_compatibility(
                        a.support, b.support, self.max_pair_distance)
                    candidates.append((score, ai, bi, detail, reason))
            # Greedy maximum-weight matching is deterministic for the two-view
            # case; hypotheses, rather than pairs, are the persistent state.
            for score, ai, bi, detail, reason in sorted(candidates, reverse=True, key=lambda x: x[0]):
                if not np.isfinite(score) or score < self.min_pair_score:
                    continue
                a, b = by_camera[views[0]][ai], by_camera[views[1]][bi]
                ia, ib = id(a), id(b)
                if ia in used or ib in used:
                    continue
                used.update((ia, ib))
                hypotheses.append(WorldObjectHypothesis(int(start_id + len(hypotheses)), [a, b],
                                                        merge_supports([a.support, b.support]),
                                                        float(score), "FUSED_TWO_VIEW",
                                                        {**detail, "pair_reason": reason}))
        for view in views:
            for x in by_camera[view]:
                if id(x) in used:
                    continue
                hypotheses.append(WorldObjectHypothesis(int(start_id + len(hypotheses)), [x], x.support,
                                                        float(x.support.quality), "SINGLE_VIEW",
                                                        {"reason": "no_safe_cross_view_hypothesis"}))
        return hypotheses, {
            "candidate_pairs": len(candidates),
            "accepted_pairs": sum(len(x.observations) == 2 for x in hypotheses),
            "single_view_objects": sum(len(x.observations) == 1 for x in hypotheses),
            "abstained_pairs": sum(1 for x in candidates if not np.isfinite(x[0]) or x[0] < self.min_pair_score),
        }
