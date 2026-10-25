"""Confidence-gated long/short visual memory inspired by MeMOTR (MIT).

No MeMOTR detector/Transformer, lifecycle or association is imported. GT-free
1024-D GMT ROI inputs keep the frozen V3 head's input shape unchanged.
"""
from __future__ import annotations
from dataclasses import dataclass,field
import numpy as np
from ..models.world_object_observation import normalized_feature


@dataclass
class WorldObjectIdentityMemory:
    long_term_embedding: np.ndarray | None=None
    short_term_embedding_by_camera: dict=field(default_factory=dict)
    identity_confidence: float=0.
    last_identity_confirmed_timestamp: int | None=None
    identity_update_status: str="UNCONFIRMED"

    def update(self,observations,timestamp,*,trusted,spatial_trusted,dimension,momentum=.85,
               coordinate_views=False,update_gate=1.,memory_delta=None):
        usable=[o for o in observations if o.visible and o.feature_valid and o.score>=.5]
        if not usable:
            self.identity_update_status="NO_CURRENT_VISUAL_EVIDENCE";return
        old=self.long_term_embedding
        if old is None and coordinate_views:
            # New objects have no trusted reference. Bootstrap from the best
            # actual view; retain agreement before mixing unlike appearances.
            anchor=sorted(usable,key=lambda o:(-o.score,o.key))[0].roi_feature
            usable=[o for o in usable if np.dot(anchor,o.roi_feature)>=.55]
        compatible=[o for o in usable if old is None or np.dot(old,o.roi_feature)>=.55]
        for o in compatible:
            prev=self.short_term_embedding_by_camera.get(o.camera_id)
            # Uncertain identity is quarantined: it must not update even the
            # short-term descriptor of an already assigned historical person.
            if trusted:
                self.short_term_embedding_by_camera[o.camera_id]=normalized_feature(
                    o.roi_feature if prev is None else .5*prev+.5*o.roi_feature,dimension)
        if not trusted:
            self.identity_update_status="IDENTITY_UNCERTAIN_NO_LONG_TERM_UPDATE";return
        self.last_identity_confirmed_timestamp=int(timestamp)
        if old is None and compatible:
            self.long_term_embedding=normalized_feature(np.average([o.roi_feature for o in compatible],axis=0,
                weights=[o.score for o in compatible]),dimension)
            self.identity_confidence=float(max(o.score for o in compatible))
            self.identity_update_status="CURRENT_VISUAL_INITIALIZATION"
            return
        if not spatial_trusted:
            self.identity_update_status="SHORT_TERM_ONLY_SPATIAL_UNCERTAIN";return
        if not compatible:
            self.identity_update_status="INCOMPATIBLE_VIEWS_QUARANTINED";return
        weights=np.asarray([o.score for o in compatible],float)
        if coordinate_views and old is not None:
            # A contrary view is retained separately, not averaged into the ID.
            weights*=np.asarray([max(0.,float(np.dot(old,o.roi_feature)))**2 for o in compatible])
        if weights.sum()<=0:return
        candidate=normalized_feature(np.average([o.roi_feature for o in compatible],axis=0,weights=weights),dimension)
        if memory_delta is not None:
            delta=np.asarray(memory_delta,float).reshape(dimension)
            if not np.isfinite(delta).all():raise ValueError("NONFINITE_MEMORY_DELTA")
            candidate=normalized_feature(candidate+.1*np.tanh(delta),dimension)
        confidence=float(np.average([o.score for o in compatible],weights=weights))
        alpha=(1-momentum)*confidence*float(np.clip(update_gate,0,1))
        self.long_term_embedding=candidate if old is None else normalized_feature((1-alpha)*old+alpha*candidate,dimension)
        self.identity_confidence=confidence
        self.identity_update_status="TRUSTED_LONG_SHORT_UPDATE"
