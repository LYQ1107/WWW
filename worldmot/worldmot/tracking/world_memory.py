"""Causal persistent identity memory in the aligned world coordinate frame."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .association import build_cost_matrix, linear_assignment
from .state import TrackStatus, WorldTrackState


class WorldIdentityMemory:
    def __init__(self, max_missed: int = 12, min_hits: int = 2,
                 gate_distance: float = 3.0, next_id: int = 1):
        self.max_missed = int(max_missed)
        self.min_hits = int(min_hits)
        self.gate_distance = float(gate_distance)
        self.next_id = int(next_id)
        self.tracks: dict[int, WorldTrackState] = {}

    def reset(self) -> None:
        self.next_id = 1
        self.tracks.clear()

    def _new(self, instance, timestamp: int) -> WorldTrackState:
        gid = self.next_id; self.next_id += 1
        f = np.asarray(instance.query_feature, dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(f)
        if norm > 1e-8: f = f / norm
        tr = WorldTrackState(
            global_id=gid, xyz_center=np.asarray(instance.xyz_center, dtype=np.float64).copy(),
            xyz_velocity=np.zeros(3, dtype=np.float64), covariance=np.asarray(instance.covariance, dtype=np.float64).copy(),
            identity_feature=f.copy(), last_timestamp=int(timestamp), support_points=np.asarray(instance.xyz_support).copy(),
            masks_by_camera=dict(instance.masks_by_camera), source=instance.source,
        )
        if self.min_hits <= 1:
            tr.status = TrackStatus.CONFIRMED
        self.tracks[gid] = tr
        return tr

    def step(self, instances, timestamp: int, geometry_aligned: bool = True) -> list[WorldTrackState]:
        active = [x for x in self.tracks.values() if x.status != TrackStatus.DELETED]
        if not geometry_aligned:
            for tr in active:
                tr.predict()
                tr.mark_missed(timestamp, self.max_missed)
            return [x for x in active if x.status != TrackStatus.DELETED]
        for tr in active:
            tr.predict()
        cost = build_cost_matrix(active, instances, gate_distance=self.gate_distance)
        pairs, unmatched_tracks, unmatched_instances = linear_assignment(cost)
        for i, j in pairs:
            tr = active[i]; ins = instances[j]
            tr.update(ins.xyz_center, ins.covariance, ins.query_feature, ins.xyz_support,
                      ins.masks_by_camera, timestamp, ins.source)
        for i in unmatched_tracks:
            active[i].mark_missed(timestamp, self.max_missed)
        for j in unmatched_instances:
            self._new(instances[j], timestamp)
        # Tentative tracks are returned for diagnostics; deleted tracks are not.
        return [x for x in self.tracks.values() if x.status != TrackStatus.DELETED]

    def snapshot(self) -> list[dict]:
        return [x.to_dict() for x in sorted(self.tracks.values(), key=lambda t: t.global_id) if x.status != TrackStatus.DELETED]
