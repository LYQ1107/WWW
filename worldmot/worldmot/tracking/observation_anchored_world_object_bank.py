"""Independent membership, birth, motion, memory and observable lifecycle."""
from dataclasses import dataclass,field
from collections import Counter
import time
import numpy as np
from .persistent_world_object_bank import PersistentWorldObject,PersistentWorldObjectBank
from .world_object_birth_manager import WorldObjectBirthManager
from .world_object_historical_assignment import HistoricalWorldObjectAssignment
from .world_object_motion_state import PersistentWorldMotionState,conservative_measurement
from ..models.world_object_observation import normalized_feature
from ..models.multi_view_world_object import assigned_measurement


@dataclass
class ObservationAnchoredWorldObject(PersistentWorldObject):
    view_identity_features: dict = field(default_factory=dict)
    support_center: np.ndarray | None = None
    ever_confirmed: bool = False
    last_identity_write: int | None = None
    identity_update_status: str = "NOT_WRITTEN"
    pending_spatial_measurement: object | None = None


class ObservationAnchoredWorldObjectBank(PersistentWorldObjectBank):
    def __init__(self,config):
        self.config=config;self.assigner=HistoricalWorldObjectAssignment(config)
        self.birth_manager=WorldObjectBirthManager(config);self.reset()

    def reset(self):
        super().reset();self.last_detection_decisions={};self.last_birth_audit={}
        self.last_membership_confidence={};self.alias_events=[]

    def active(self):
        return sorted((q for q in self.objects.values() if q.status!="REMOVED"),key=lambda q:q.global_id)

    def predict(self,timestamp):
        # Keep the verified C1 state propagation and its zero-dt idempotence.
        if not self.config.use_geometry:return
        for q in self.active():
            state=q.spatial_state
            if state is None:continue
            state.predict(timestamp)
            q.center=state.position.copy();q.covariance=state.covariance[:3,:3].copy()
            q.geometry_valid=bool(state.motion_valid);q.spatial_state_status=state.uncertainty_status

    def _write_identity(self,q,observations,timestamp,trusted,birth):
        cfg=self.config
        usable=[o for o,ok in zip(observations,trusted) if ok and o.visible and o.feature_valid]
        if not usable:
            q.identity_update_status="NO_TRUSTED_CURRENT_IDENTITY_WRITE";return False
        weights=np.asarray([o.score for o in usable]);candidate=normalized_feature(np.average([o.roi_feature for o in usable],axis=0,weights=weights),cfg.feature_dim)
        q.identity_embedding=candidate if q.identity_embedding is None else normalized_feature(cfg.memory_momentum*q.identity_embedding+(1-cfg.memory_momentum)*candidate,cfg.feature_dim)
        q.identity_confidence=float(np.average([o.score for o in usable],weights=weights))
        for o in usable:
            old=q.view_identity_features.get(o.camera_id)
            q.view_identity_features[o.camera_id]=o.roi_feature.copy() if old is None else normalized_feature(cfg.memory_momentum*old+(1-cfg.memory_momentum)*o.roi_feature,cfg.feature_dim)
            entries=q.recent_view_features.setdefault(o.camera_id,[]);entries.append((int(timestamp),o.roi_feature.copy()));del entries[:-4]
        q.last_identity_write=int(timestamp);q.identity_update_status="ACTUAL_OBSERVATION_BIRTH_WRITE" if birth else "TRUSTED_MEMBERSHIP_WRITE"
        return True

    def _update_spatial(self,q,observations,timestamp,identity_trusted,birth):
        cfg=self.config
        if not cfg.use_geometry:
            q.spatial_state_status="REGISTERED_NO_OMEGA_B4";return False
        usable=[o for o in observations if o.geometry_valid and o.world_state_update_eligible
                and o.geometry_reliability>=cfg.reliable_geometry_min_quality
                and o.source.get("alignment_confidence",0)>=cfg.alignment_min_inlier_ratio]
        if not usable or not (birth or identity_trusted):
            q.spatial_state_status="NO_RELIABLE_CURRENT_SPATIAL_MEASUREMENT";return False
        value=conservative_measurement(usable)
        if value is None or value[2]<cfg.reliable_geometry_min_quality:
            q.spatial_state_status="CORRELATED_VIEWS_OR_SUPPORT_DISAGREEMENT_ABSTAIN";return False
        center,covariance,reliability=value;state=q.spatial_state;reinitialized=False
        if state is not None and state.motion_valid:
            _,_,d2=state.innovation(center,covariance)
            if d2>cfg.geometry_pair_max_mahalanobis**2:
                old=q.pending_spatial_measurement
                coherent=False
                if old is not None and timestamp-old[0]<=2:
                    delta=center-old[1];c=covariance+old[2];floor=max(float(np.trace(c))/3*1e-4,1e-10)
                    coherent=bool(delta@np.linalg.solve(c+np.eye(3)*floor,delta)<=9)
                q.pending_spatial_measurement=(int(timestamp),center.copy(),covariance.copy())
                if not coherent:
                    q.spatial_state_status="IDENTITY_RETAINED_SPATIAL_CONFLICT_AWAIT_CURRENT_CONFIRMATION";return False
                # Two independent timestamps of reliable current evidence permit
                # spatial reinitialization without changing the object's ID.
                state=None;reinitialized=True
        if state is None or not state.motion_valid:
            state=PersistentWorldMotionState.initialize(center,covariance,timestamp,confidence=reliability,
                                                        acceleration_ratio=.01,ticks_per_second=None)
            q.spatial_state=state;accepted=True
        else:accepted=state.update(center,covariance,timestamp,confidence=reliability)
        if accepted:
            q.center=state.position.copy();q.covariance=state.covariance[:3,:3].copy();q.geometry_valid=True
            points=np.concatenate([o.spatial_support.foreground_points for o in usable])
            if len(points)>64:points=points[np.linspace(0,len(points)-1,64).astype(int)]
            q.support_points=points.copy();q.support_center=center.copy();q.pending_spatial_measurement=None
            state.alignment_confidence=float(min(o.source.get("alignment_confidence",0) for o in usable))
        q.spatial_state_status="CURRENT_EVIDENCE_SPATIAL_REINITIALIZATION_SAME_ID" if reinitialized else state.uncertainty_status
        return bool(accepted)

    def _update_object(self,q,observations,timestamp,trust,birth):
        started=time.perf_counter();wrote=self._write_identity(q,observations,timestamp,trust,birth)
        self.last_timings["identity_memory_write_s"]+=time.perf_counter()-started
        started=time.perf_counter();spatial=self._update_spatial(q,observations,timestamp,any(trust),birth)
        self.last_timings["C1_spatial_measurement_update_s"]+=time.perf_counter()-started
        q.last_observation=assigned_measurement(observations,0,self.config.feature_dim,self.config.use_geometry)
        q.visible_cameras=tuple(sorted(o.camera_id for o in observations));q.camera_history.update(q.visible_cameras)
        q.visibility_by_camera={v:v in q.visible_cameras for v in sorted(q.camera_history)}
        q.last_timestamp=int(timestamp);q.missed=0;q.hits+=1
        q.ever_confirmed|=q.hits>=self.config.min_confirmation_hits or (birth and min(o.score for o in observations)>=self.config.birth_high_score)
        q.status="CONFIRMED" if q.ever_confirmed else "TENTATIVE"
        return wrote,spatial

    def step(self,observations,timestamp,margins,context,geometry=None):
        if self.last_timestamp is not None and int(timestamp)<=self.last_timestamp:raise ValueError("NONCAUSAL_V5_TIMESTAMP")
        observations=list(observations);keys=[o.key for o in observations]
        if keys!=sorted(keys) or len(keys)!=len(set(keys)):raise ValueError("ORDERED_UNIQUE_CURRENT_KEYS_REQUIRED")
        if context.timestamp!=int(timestamp) or any(o.timestamp!=int(timestamp) for o in observations):raise ValueError("STALE_OBSERVATIONS")
        if set(o.camera_id for o in observations)-{1,2}:raise ValueError("REGISTERED_DUAL_CAMERA_ORDER")
        history=self.active();self.predict(timestamp)
        self.last_timings=Counter();started=time.perf_counter()
        decision=self.assigner.assign(history,observations,margins,context,geometry)
        self.last_timings["historical_assignment_s"]=time.perf_counter()-started
        assigned={j:o.global_id for o in decision.selected for j in o.indices};hindex={q.global_id:i for i,q in enumerate(history)}
        hints={}
        if margins is not None and len(history):
            for j in range(len(observations)):
                indices=np.argsort(-margins[:,j],kind="stable");i=int(indices[0]);other=float(margins[indices[1],j]) if len(indices)>1 else -np.inf
                if margins[i,j]>=self.config.identity_write_min_margin and margins[i,j]-other>=self.config.identity_write_competition_margin:hints[j]=history[i].global_id
        started=time.perf_counter();birth=self.birth_manager.form(observations,decision.unmatched,assigned,context,history_hint=hints)
        self.last_timings["birth_manager_s"]=time.perf_counter()-started
        self.last_assignments={};self.last_detection_decisions={};self.last_membership_confidence={};counts=Counter()
        for q in history:
            q.age+=1;q.missed=int(timestamp)-q.last_timestamp;q.visible_cameras=();q.last_observation=None
            q.visibility_by_camera={v:False for v in q.camera_history}
        for option in decision.selected:
            q=self.objects[option.global_id];i=hindex[q.global_id];trust=[]
            recovered=q.missed>1
            for j in option.indices:
                if margins is None:ok=True;score=None
                else:
                    score=float(margins[i,j]);rivals=[float(margins[k,j]) for k in range(len(history)) if k!=i]
                    rivals+= [float(margins[i,k]) for k,o in enumerate(observations) if k!=j and o.camera_id==observations[j].camera_id]
                    ok=score>=self.config.identity_write_min_margin and score-max(rivals,default=score-1)>=self.config.identity_write_competition_margin
                trust.append(bool(ok));self.last_membership_confidence[observations[j].key]=score
                self.last_assignments[observations[j].key]=q.global_id
                self.last_detection_decisions[observations[j].key]={"decision":option.reason,"stage":"HISTORICAL_MEMBERSHIP","identity_write_eligible":bool(ok),"membership_margin":score,"recovered_from_LOST":recovered}
            wrote,spatial=self._update_object(q,[observations[j] for j in option.indices],timestamp,trust,False)
            q.decision_status=option.reason;counts.update(identity_writes=int(wrote),spatial_updates=int(spatial),historical_objects_matched=1,recoveries=int(recovered))
        for option in birth.selected:
            obs=[observations[j] for j in option.indices]
            q=ObservationAnchoredWorldObject(self.next_id,np.full(3,np.nan),np.eye(3)*1e6,np.empty((0,3)),None,0.,False,(),int(timestamp),hits=0)
            self.next_id+=1;self.objects[q.global_id]=q
            wrote,spatial=self._update_object(q,obs,timestamp,[True]*len(obs),True)
            q.decision_status=option.reason
            for j in option.indices:
                self.last_assignments[observations[j].key]=q.global_id
                self.last_detection_decisions[observations[j].key]={"decision":option.reason,"stage":"INDEPENDENT_BIRTH","identity_write_eligible":True,"membership_margin":None,"recovered_from_LOST":False}
            counts.update(identity_writes=int(wrote),spatial_updates=int(spatial))
        for j,anchor in birth.duplicates.items():
            key=observations[j].key;gid=self.last_assignments[observations[anchor].key]
            self.last_assignments[key]=gid
            self.last_detection_decisions[key]={"decision":birth.reasons[j],"stage":"TRACK_AWARE_DUPLICATE","output_eligible":False,"duplicate_of":list(observations[anchor].key),"identity_write_eligible":False}
        for q in history:
            if q.missed:
                q.status="LOST";q.decision_status="NO_CURRENT_OBSERVATION";q.identity_update_status="LOST_NO_APPEARANCE_WRITE"
                retention=self.config.confirmed_lost_retention_ticks if q.ever_confirmed else self.config.tentative_lost_retention_ticks
                if q.missed>retention:q.status="REMOVED";q.decision_status="OBSERVABLE_DURATION_EXCEEDED"
        self.last_timestamp=int(timestamp);self.last_birth_audit=birth.diagnostics
        if set(self.last_assignments)!=set(keys):raise AssertionError("V5_DETECTION_DECISION_COVERAGE")
        for q in self.active():
            s=q.spatial_state
            if s is not None and (not np.isfinite(s.state).all() or not np.isfinite(s.covariance).all() or np.linalg.eigvalsh(s.covariance).min()<-1e-7):raise ValueError("INVALID_C1_6D_STATE")
        return self.active(),{**decision.diagnostics,**dict(counts),**{k:v for k,v in birth.diagnostics.items() if k!="birth_pair_audit"},
            "objects":len(self.active()),"lost":sum(q.status=="LOST" for q in self.active()),"removed_total":sum(q.status=="REMOVED" for q in self.objects.values()),
            "candidate_count":len(observations),"output_count":len(observations)-len(birth.duplicates),"all_detections_have_decision":True}
