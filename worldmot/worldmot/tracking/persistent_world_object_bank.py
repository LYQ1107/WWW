"""Separate spatial state and long-term identity, updated from joint decisions."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import time
from .joint_world_object_assignment import JointWorldObjectAssignment, _sigmoid, AssignmentOption
from .world_object_config import WorldObjectConfig
from .world_object_measurement_update import update_world_measurement
from ..models.multi_view_world_object import assigned_measurement
from ..models.world_object_observation import normalized_feature, FeatureUnavailable


@dataclass
class PersistentWorldObject:
    global_id: int
    center: np.ndarray
    covariance: np.ndarray
    support_points: np.ndarray
    identity_embedding: np.ndarray | None
    identity_confidence: float
    geometry_valid: bool
    visible_cameras: tuple[int, ...]
    last_timestamp: int
    visibility_by_camera: dict = field(default_factory=dict)
    recent_view_features: dict = field(default_factory=dict)
    age: int = 1
    hits: int = 1
    missed: int = 0
    status: str = "TENTATIVE"
    last_observation: object | None = None
    camera_history: set = field(default_factory=set)
    decision_status: str = "BIRTH"
    spatial_state: object | None = None
    identity_state: object | None = None
    spatial_state_status: str = "LEGACY"

    @property
    def feature(self):
        if self.identity_embedding is None:
            raise FeatureUnavailable("FEATURE_UNAVAILABLE in persistent identity")
        return self.identity_embedding

    @property
    def observed_this_step(self):
        return self.missed == 0 and self.last_observation is not None


class PersistentWorldObjectBank:
    def __init__(self, config=None, *, feature_dim=None, max_missed=30, min_hits=2, gate_distance=6.0):
        self.config = config or WorldObjectConfig(feature_dim=feature_dim, max_missed=max_missed,
                                                 min_hits=min_hits, geometry_gate=gate_distance)
        self.assigner = JointWorldObjectAssignment(self.config)
        self.reset()

    def reset(self):
        self.next_id = 1
        self.objects = {}
        self.last_assignments = {}
        self.last_timestamp = None
        self.last_hypotheses = []
        self.last_timings = {}

    def active(self):
        return sorted((q for q in self.objects.values() if q.status != "DELETED"), key=lambda q: q.global_id)

    def predict(self,timestamp):
        """Idempotent prediction BEFORE both head inputs and assignment."""
        if not self.config.enable_motion_prediction:return
        for q in self.active():
            state=q.spatial_state
            if state is None:continue
            state.predict(timestamp)
            q.center=state.position.copy();q.covariance=state.covariance[:3,:3].copy()
            q.geometry_valid=bool(state.motion_valid)
            q.spatial_state_status=state.uncertainty_status

    def _identity_trusted(self,q,measurement,learned,qindex,indices,is_birth=False):
        cfg=self.config
        if is_birth:
            return all(o.feature_valid and o.score>=cfg.min_memory_score for o in measurement.observations)
        if q.identity_embedding is None:return False
        history=self.active()
        from .world_object_identity_memory import identity_similarity
        for j,o in zip(indices,measurement.observations):
            app=identity_similarity(q,o,cfg) if o.feature_valid else -1.
            others=[identity_similarity(other,o,cfg) for other in history
                    if other.global_id!=q.global_id and other.identity_embedding is not None and o.feature_valid]
            gap=app-max(others,default=-1.)
            current=[(other_index,other) for other_index,other in enumerate(self._identity_reference_observations)
                     if other.camera_id==o.camera_id and other.key!=o.key and other.feature_valid]
            current_gap=app-max((identity_similarity(q,other,cfg) for _,other in current),default=-1.)
            visual=(app>=cfg.recovery_similarity and gap>=cfg.identity_competition_margin
                    and current_gap>=cfg.identity_competition_margin)
            learned_strong=False
            if learned is not None:
                i=qindex[q.global_id];value=float(learned['assignment_logits'][i,j])
                margins=[float(learned['assignment_logits'][k,j]) for k in range(len(qindex)) if k!=i]
                advantage=value-max(margins,default=value-cfg.identity_learned_margin)
                current_advantage=value-max((float(learned['assignment_logits'][i,k]) for k,_ in current),
                    default=value-cfg.identity_learned_margin)
                learned_strong=(value-float(learned['null_logits'][i])>=cfg.identity_learned_margin
                    and advantage>=cfg.identity_learned_margin and current_advantage>=cfg.identity_learned_margin
                    and app>=cfg.min_appearance_similarity)
            if o.score>=cfg.min_memory_score and (visual or learned_strong):return True
        return False

    def _motion_update(self,q,measurement,timestamp,gate,spatial_conflict,identity_trusted):
        from .world_object_motion_state import (PersistentWorldMotionState,conservative_measurement,
                                              temporal_geometry_evidence,positive_covariance)
        cfg=self.config
        if not cfg.spatial_state_update or gate<=0:return False
        if cfg.enable_safe_identity_memory and not identity_trusted:
            q.spatial_state_status="IDENTITY_UNCERTAIN_SPATIAL_UPDATE_DEFERRED";return False
        conflict=spatial_conflict
        if cfg.enable_safe_identity_memory and q.spatial_state is not None:
            conflict |= any(temporal_geometry_evidence(q,o,cfg)[1] for o in measurement.observations)
        if conflict:
            if cfg.enable_safe_identity_memory and identity_trusted and q.spatial_state is not None:
                q.spatial_state.invalidate();q.geometry_valid=False;q.support_points=np.empty((0,3))
                q.spatial_state_status="SPATIAL_UNCERTAIN"
            return False
        if cfg.enable_multiview_coordination:
            value=conservative_measurement(measurement.observations)
            if value is not None:value=(value[0],value[1]/max(gate,.01),value[2])
        else:
            support=measurement.support
            value=None if support is None else (support.center,support.covariance/max(support.quality*gate,.01),support.quality)
        if value is None:
            q.spatial_state_status="NO_TRUSTED_CURRENT_GEOMETRY";return False
        center,cov,reliability=value
        try:
            cov=positive_covariance(cov,3)
            if not np.isfinite(center).all() or reliability<.6:
                q.spatial_state_status="LOW_RELIABILITY_MEASUREMENT_ABSTAIN";return False
            state=q.spatial_state
            if state is None or not state.motion_valid:
                state=PersistentWorldMotionState.initialize(center,cov,timestamp,confidence=reliability,
                    acceleration_ratio=cfg.motion_acceleration_ratio,ticks_per_second=cfg.timestamp_ticks_per_second)
                q.spatial_state=state;accepted=True
            else:accepted=state.update(center,cov,timestamp,confidence=reliability)
        except (ValueError,np.linalg.LinAlgError):
            q.spatial_state_status="INVALID_MEASUREMENT_ABSTAIN";return False
        if accepted:
            q.center=state.position.copy();q.covariance=state.covariance[:3,:3].copy();q.geometry_valid=True
            q.support_points=measurement.support.foreground_points.copy() if measurement.support is not None else np.empty((0,3))
            state.alignment_confidence=float(min(o.source.get('alignment_confidence',1.) for o in measurement.observations))
        q.spatial_state_status=state.uncertainty_status
        return bool(accepted)

    def _update(self, q, measurement, timestamp, gate=1.0, delta=None, spatial_conflict=False, identity_trusted=True,
                multiview_identity_confirmed=False):
        cfg = self.config
        support = measurement.support
        spatial_trusted=False
        if cfg.enable_motion_prediction:
            spatial_trusted=self._motion_update(q,measurement,timestamp,gate,spatial_conflict,identity_trusted)
        elif cfg.spatial_state_update and support is not None and not spatial_conflict and gate > 0:
            if q.geometry_valid:
                q.center, q.covariance = update_world_measurement(q.center, q.covariance,
                    support.center, support.covariance, support.quality * gate)
            else:
                q.center = support.center.copy()
                q.covariance = support.covariance.copy()
                q.geometry_valid = True
            q.support_points = support.foreground_points.copy()
            spatial_trusted=True
        actual = [o for o in measurement.observations if o.visible and o.feature_valid
                  and o.score >= cfg.min_memory_score]
        if cfg.enable_safe_identity_memory and cfg.use_appearance:
            from .world_object_identity_memory import WorldObjectIdentityMemory
            if q.identity_state is None:
                q.identity_state=WorldObjectIdentityMemory(None if q.identity_embedding is None else q.identity_embedding.copy(),
                    identity_confidence=q.identity_confidence)
            q.identity_state.update(actual,timestamp,trusted=identity_trusted and gate>0,spatial_trusted=spatial_trusted,
                dimension=cfg.feature_dim,momentum=cfg.memory_momentum,coordinate_views=cfg.enable_multiview_coordination,
                update_gate=gate,memory_delta=delta,multiview_identity_confirmed=multiview_identity_confirmed)
            q.identity_embedding=q.identity_state.long_term_embedding
            q.identity_confidence=q.identity_state.identity_confidence
        elif cfg.use_appearance and actual and gate > 0:
            from ..models.world_object_observation import fuse_identity_features
            candidate, confidence = fuse_identity_features(actual, cfg.feature_dim, use_geometry=cfg.use_geometry)
            old = q.identity_embedding
            compatible = old is None or np.dot(old, candidate) >= cfg.min_appearance_similarity
            if compatible and not spatial_conflict:
                if delta is not None:
                    delta = np.asarray(delta, dtype=np.float32).reshape(cfg.feature_dim)
                    if not np.isfinite(delta).all():
                        raise ValueError("NONFINITE_MEMORY_DELTA")
                    candidate = normalized_feature(candidate + 0.1 * np.tanh(delta), cfg.feature_dim)
                alpha = (1 - cfg.memory_momentum) * float(gate)
                q.identity_embedding = (candidate if old is None else
                                        normalized_feature((1-alpha)*old + alpha*candidate, cfg.feature_dim))
                q.identity_confidence = float(confidence)
                for o in actual:
                    entries = q.recent_view_features.setdefault(o.camera_id, [])
                    entries.append((timestamp, o.roi_feature.copy()))
                    del entries[:-4]
        q.visible_cameras = measurement.cameras
        q.camera_history.update(measurement.cameras)
        q.visibility_by_camera = {v: v in measurement.cameras for v in sorted(q.camera_history)}
        q.last_timestamp = int(timestamp)
        q.last_observation = measurement
        q.missed = 0
        q.hits += 1
        q.status = "CONFIRMED" if q.hits >= cfg.min_hits else "TENTATIVE"

    def step(self, observations, timestamp, learned=None, *, training_decision=None):
        observations = sorted(list(observations), key=lambda o: o.key)
        if self.last_timestamp is not None and int(timestamp) <= self.last_timestamp:
            raise ValueError("NONCAUSAL_OR_DUPLICATE_TIMESTAMP")
        keys = [o.key for o in observations]
        if len(keys) != len(set(keys)):
            raise ValueError("DUPLICATE_CAMERA_DETECTION_KEY")
        if len({o.camera_id for o in observations}) > 2:
            raise ValueError("V3_TWO_CAMERA_CONTRACT")
        for o in observations:
            if o.timestamp != int(timestamp):
                raise ValueError("FUTURE_OR_STALE_OBSERVATION")
            if self.config.use_appearance:
                if not o.feature_valid:
                    raise FeatureUnavailable("FEATURE_UNAVAILABLE: choose GEOMETRY_ONLY_ABLATION_V3 explicitly")
                normalized_feature(o.roi_feature, self.config.feature_dim)
        history = self.active()
        self.predict(timestamp)
        if learned is not None:
            m,n=len(history),len(observations)
            shapes={"assignment_logits":(m,n),"null_logits":(m,),"existence_logits":(m,),
                    "birth_logits":(n,),"update_gate":(m,n),"memory_delta":(m,n,self.config.feature_dim)}
            for key,shape in shapes.items():
                value=learned[key]
                if hasattr(value,"detach"):
                    import torch
                    finite=bool(torch.isfinite(value).all().item())
                else:finite=np.isfinite(value).all()
                if tuple(value.shape)!=shape or not finite:
                    raise ValueError("INVALID_LEARNED_EVIDENCE_"+key)
            if "birth_pair_logits" in learned and (learned["birth_pair_logits"].shape!=(n,n)
                    or not np.isfinite(learned["birth_pair_logits"]).all()):
                raise ValueError("INVALID_LEARNED_BIRTH_PAIR_EVIDENCE")
        assignment_started = time.perf_counter()
        decision = training_decision or self.assigner.assign(history, observations, learned)
        self.last_timings = {"joint_assignment_s":time.perf_counter()-assignment_started}
        update_started = time.perf_counter()
        assigned=[j for option in decision.selected for j in option.indices]
        ids=[option.global_id for option in decision.selected if option.global_id is not None]
        if (len(assigned)!=len(set(assigned)) or len(ids)!=len(set(ids))
                or set(ids)-{q.global_id for q in history}
                or set(assigned).intersection(decision.unmatched)
                or set(assigned).union(decision.unmatched)!=set(range(len(observations)))
                or any(len({observations[j].camera_id for j in option.indices})!=len(option.indices)
                       for option in decision.selected)):
            raise ValueError("INVALID_JOINT_DECISION_CONSTRAINTS")
        qindex = {q.global_id: i for i, q in enumerate(history)}
        if self.config.enable_safe_identity_memory:self._identity_reference_observations=observations
        trusted_selected={option.global_id:self._identity_trusted(self.objects[option.global_id],
            assigned_measurement([observations[j] for j in option.indices],0,self.config.feature_dim,self.config.use_geometry,
                                 self.config.enable_multiview_coordination),
            learned,qindex,option.indices) for option in decision.selected if option.global_id is not None
            } if self.config.enable_safe_identity_memory else {}
        selected_delta={}
        if learned is not None:
            keys=[(qindex[option.global_id],j) for option in decision.selected if option.global_id is not None for j in option.indices]
            if keys:
                rows,cols=zip(*keys);delta=learned["memory_delta"][list(rows),list(cols)]
                delta=delta.detach().cpu().numpy() if hasattr(delta,"detach") else np.asarray(delta)
                selected_delta=dict(zip(keys,delta))
        self.last_assignments = {}
        self.last_hypotheses = []
        for q in history:
            q.age += 1
            q.missed += 1
            q.last_observation = None
            q.visible_cameras = ()
            q.visibility_by_camera = {v: False for v in q.camera_history}
        selections = list(decision.selected)
        selections.extend(AssignmentOption(None, (j,), 0, "SINGLETON_BIRTH_OR_UNCONFIRMED") for j in decision.unmatched)
        selections.sort(key=lambda option: (option.global_id is None, option.global_id or 0,
                                            tuple(observations[j].key for j in option.indices)))
        birth_count = 0
        for option in selections:
            obs = [observations[j] for j in option.indices]
            measurement = assigned_measurement(obs, len(self.last_hypotheses), self.config.feature_dim,
                                               self.config.use_geometry,self.config.enable_multiview_coordination)
            self.last_hypotheses.append(measurement)
            if option.global_id is not None:
                q = self.objects[option.global_id]
                gate, delta = 1.0, None
                if learned is not None:
                    i = qindex[q.global_id]
                    gate = float(np.min(learned["update_gate"][i, list(option.indices)]))
                    delta = np.mean([selected_delta[i,j] for j in option.indices], axis=0)
                self._update(q, measurement, timestamp, gate, delta,
                             option.reason == "VISUAL_RECOVERY_SPATIAL_CONFLICT",
                             trusted_selected.get(q.global_id,True),len(option.indices)>1)
            else:
                birth_count += 1
                prob = (min(_sigmoid(learned["birth_logits"][j]) for j in option.indices)
                        if learned is not None else 1.0)
                support = measurement.support
                if not self.config.spatial_state_update:support=None
                q = PersistentWorldObject(self.next_id,
                    support.center.copy() if support is not None else np.full(3, np.nan),
                    support.covariance.copy() if support is not None else np.eye(3)*1e6,
                    support.foreground_points.copy() if support is not None else np.empty((0,3)),
                    None, 0.0, support is not None, (), int(timestamp), hits=0,
                    camera_history=set(measurement.cameras))
                self.objects[q.global_id] = q
                self.next_id += 1
                self._update(q, measurement, timestamp,multiview_identity_confirmed=len(option.indices)>1)
                if prob < self.config.birth_threshold or decision.diagnostics.get("fallback"):
                    q.status = "UNCONFIRMED"
            q.decision_status = option.reason
            for j in option.indices:
                self.last_assignments[observations[j].key] = q.global_id
        for q in history:
            if q.missed:
                q.status = "LOST"
                q.decision_status = "NULL_NO_CURRENT_OBSERVATION"
                low_existence = (learned is not None and
                                 _sigmoid(learned["existence_logits"][qindex[q.global_id]]) < self.config.death_threshold)
                if q.missed > self.config.max_missed or (low_existence and q.missed >= 2):
                    q.status = "DELETED"
                if q.geometry_valid and self.config.spatial_state_update and not self.config.enable_motion_prediction:
                    q.covariance = q.covariance + np.eye(3)*0.02
        self.last_timestamp = int(timestamp)
        self.last_timings["identity_update_s"] = time.perf_counter()-update_started
        return self.active(), {**decision.diagnostics, "births": birth_count,
            "matched": sum(o.global_id is not None for o in decision.selected),
            "lost": sum(q.status == "LOST" for q in history),
            "geometry_failures": sum(not o.geometry_valid for o in observations),
            "objects": len(self.active()), "all_detections_assigned": len(self.last_assignments) == len(observations)}
