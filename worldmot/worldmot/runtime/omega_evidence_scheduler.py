"""Causal evidence requests: takes visual observations and past state ONLY."""
from dataclasses import dataclass
import numpy as np

REQUEST_OMEGA="REQUEST_OMEGA"
SKIP_OMEGA="SKIP_OMEGA"


@dataclass(frozen=True)
class EvidenceRequest:
    action: str
    reasons: tuple
    timestamp: int


class OmegaEvidenceScheduler:
    def __init__(self,config):
        self.config=config;self.reset()

    def reset(self):
        self.last_timestamp=None;self.last_trusted_geometry_timestamp=None
        self.requests=0;self.skips=0

    def decide(self,visual_observations,history,timestamp):
        timestamp=int(timestamp)
        if self.last_timestamp is not None and timestamp<=self.last_timestamp:
            raise ValueError("NONCAUSAL_SCHEDULER_TIMESTAMP")
        observations=list(visual_observations);history=list(history);reasons=[]
        if self.last_timestamp is None:reasons.append("SCENE_INITIALIZATION")
        if (self.last_trusted_geometry_timestamp is None or
                timestamp-self.last_trusted_geometry_timestamp>=self.config.geometry_stale_ticks):
            reasons.append("NO_RECENT_TRUSTED_GEOMETRY")
        proposals={};by_camera={}
        for index,o in enumerate(observations):
            # Deliberately do not access spatial_support/world_center/quality.
            if not o.feature_valid:
                reasons.append("NO_VISUAL_FEATURE");continue
            scores=sorted(((float(np.dot(q.identity_embedding,o.roi_feature)),q.global_id,q.missed)
                for q in history if q.identity_embedding is not None),reverse=True)
            if not scores or scores[0][0]<self.config.min_appearance_similarity:
                reasons.append("NEW_OR_UNEXPLAINED_IDENTITY");continue
            best=scores[0];runner=scores[1][0] if len(scores)>1 else -1.
            if best[0]-runner<self.config.identity_competition_margin:
                reasons.append("HISTORICAL_IDENTITY_COMPETITION")
            if best[2]>0 and best[0]<self.config.recovery_similarity:
                reasons.append("AMBIGUOUS_REAPPEARANCE")
            proposals.setdefault((best[1],o.camera_id),[]).append(index)
            by_camera.setdefault(o.camera_id,[]).append((index,best[1],o.roi_feature))
        if any(len(indices)>1 for indices in proposals.values()):
            reasons.append("MULTIPLE_CURRENT_OBSERVATIONS_COMPETE")
        cameras=sorted(by_camera)
        if len(cameras)==2:
            for _,left_id,left in by_camera[cameras[0]]:
                for _,right_id,right in by_camera[cameras[1]]:
                    if float(np.dot(left,right))>=self.config.recovery_similarity and left_id!=right_id:
                        reasons.append("CROSS_VIEW_IDENTITY_CONTRADICTION")
        for q in history:
            state=getattr(q,'spatial_state',None)
            if q.missed or not any(gid==q.global_id for gid,_ in proposals):continue
            if state is None or not state.motion_valid:
                reasons.append("SPATIAL_STATE_UNTRUSTED");continue
            # The ratio is invariant to an arbitrary common Sim(3) scale.
            baseline=max(float(np.trace(state.acceleration_density))/max(self.config.motion_acceleration_ratio,1e-12),1e-30)
            if float(np.trace(state.covariance[:3,:3]))/baseline>100:
                reasons.append("SPATIAL_UNCERTAINTY_EXPANDED")
        reasons=tuple(sorted(set(reasons)))
        self.last_timestamp=timestamp
        action=REQUEST_OMEGA if reasons else SKIP_OMEGA
        if reasons:self.requests+=1
        else:self.skips+=1
        return EvidenceRequest(action,reasons,timestamp)

    def observe_alignment(self,timestamp,trusted):
        if self.last_timestamp != int(timestamp):raise ValueError("ALIGNMENT_WITHOUT_CURRENT_REQUEST")
        if trusted:self.last_trusted_geometry_timestamp=int(timestamp)
