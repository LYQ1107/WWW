"""Persistent world-object bank with explicit visibility/lifecycle rules."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .association import cosine_distance, linear_assignment
from .world_object_measurement_update import update_world_measurement


@dataclass
class PersistentWorldObject:
    global_id: int
    center: np.ndarray
    covariance: np.ndarray
    support_points: np.ndarray
    feature: np.ndarray
    visible_cameras: tuple[int, ...]
    last_timestamp: int
    age: int = 1
    hits: int = 1
    missed: int = 0
    status: str = "TENTATIVE"
    last_observation: object | None = None
    camera_history: set[int] = field(default_factory=set)

    @property
    def observed_this_step(self) -> bool:
        return self.missed == 0 and self.last_observation is not None

    def predict(self) -> None:
        self.age += 1
        self.missed += 1
        self.covariance = self.covariance + np.eye(3) * (0.02 * max(1, self.missed))
        if self.missed > 0 and self.status == "CONFIRMED":
            self.status = "LOST"


class PersistentWorldObjectBank:
    """Association is object-to-hypothesis, never camera-to-camera ID merge."""

    def __init__(self, max_missed: int = 12, gate_distance: float = 3.0, min_hits: int = 2):
        self.max_missed = int(max_missed); self.gate_distance = float(gate_distance); self.min_hits = int(min_hits)
        self.next_id = 1; self.objects: dict[int, PersistentWorldObject] = {}
        self.last_assignments: dict[int, int] = {}

    def reset(self) -> None:
        self.next_id = 1; self.objects.clear(); self.last_assignments.clear()

    @staticmethod
    def _feature(x):
        f = np.asarray(getattr(x, "feature", np.zeros(1)), dtype=np.float32).reshape(-1)
        n = np.linalg.norm(f); return f / max(n, 1e-8)

    def _cost(self, obj: PersistentWorldObject, hyp) -> float:
        dist = float(np.linalg.norm(obj.center - hyp.center))
        scale = float(np.sqrt(max(np.trace(obj.covariance), 1e-6)))
        if dist > self.gate_distance + 2.0 * scale + 0.5 * obj.missed:
            return np.inf
        app = cosine_distance(obj.feature, self._feature(hyp))
        # Unreliable/mixed geometry cannot aggressively rewrite a persistent
        # object; it is allowed as a visual fallback only within a narrow gate.
        reliability = float(np.clip(hyp.support.quality, 0.0, 1.0))
        cost = (0.75 * dist / max(self.gate_distance, 1e-6)
                + 0.15 * app + 0.10 * (1.0 - reliability))
        # W4 is optional and frozen at inference.  The learned gate is only a
        # small persistent-update prior; absent diagnostics reproduce W3
        # exactly.  It cannot create an identity or access GT.
        gate = hyp.diagnostics.get("learned_update_gate")
        if gate is not None and np.isfinite(float(gate)):
            cost += 0.10 * (1.0 - float(np.clip(gate, 0.0, 1.0)))
        return cost

    def step(self, hypotheses, timestamp: int) -> tuple[list[PersistentWorldObject], dict]:
        hypotheses = list(hypotheses); self.last_assignments = {}
        active = [x for x in self.objects.values() if x.status != "DELETED"]
        for x in active:
            x.predict(); x.last_observation = None
        cost = np.full((len(active), len(hypotheses)), np.inf)
        for i, obj in enumerate(active):
            for j, hyp in enumerate(hypotheses): cost[i, j] = self._cost(obj, hyp)
        pairs, unmatched_objects, unmatched_hyp = linear_assignment(cost, max_cost=1.5)
        for i, j in pairs:
            obj, hyp = active[i], hypotheses[j]
            center, cov = update_world_measurement(obj.center, obj.covariance, hyp.center,
                                                   hyp.support.covariance, hyp.support.quality)
            obj.center, obj.covariance = center, cov
            obj.support_points = np.asarray(hyp.support.foreground_points, dtype=np.float64)
            obj.feature = self._feature(hyp); obj.visible_cameras = hyp.cameras
            obj.camera_history.update(hyp.cameras); obj.last_timestamp = int(timestamp)
            obj.last_observation = hyp; obj.missed = 0; obj.hits += 1
            if obj.hits >= self.min_hits: obj.status = "CONFIRMED"
            self.last_assignments[int(j)] = obj.global_id
        for i in unmatched_objects:
            active[i].last_timestamp = int(timestamp)
            if active[i].missed > self.max_missed: active[i].status = "DELETED"
        for j in unmatched_hyp:
            hyp = hypotheses[j]
            obj = PersistentWorldObject(self.next_id, np.asarray(hyp.center, dtype=np.float64),
                                        np.asarray(hyp.support.covariance, dtype=np.float64),
                                        np.asarray(hyp.support.foreground_points, dtype=np.float64),
                                        self._feature(hyp), hyp.cameras, int(timestamp),
                                        last_observation=hyp, camera_history=set(hyp.cameras))
            if self.min_hits <= 1: obj.status = "CONFIRMED"
            self.objects[obj.global_id] = obj; self.last_assignments[int(j)] = obj.global_id; self.next_id += 1
        return [x for x in self.objects.values() if x.status != "DELETED"], {
            "objects": len([x for x in self.objects.values() if x.status != "DELETED"]),
            "matched": len(pairs), "births": len(unmatched_hyp),
            "lost": sum(x.status == "LOST" for x in active),
            "unmatched_hypotheses": len(unmatched_hyp),
        }
