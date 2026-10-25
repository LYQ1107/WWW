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
    short_term_confidence_by_camera: dict=field(default_factory=dict)
    short_term_timestamp_by_camera: dict=field(default_factory=dict)

    def trusted_view(self,camera,timestamp,min_confidence=.5):
        camera=int(camera);feature=self.short_term_embedding_by_camera.get(camera)
        confidence=self.short_term_confidence_by_camera.get(camera,0.)
        observed=self.short_term_timestamp_by_camera.get(camera)
        if (feature is None or observed is None or observed>timestamp or confidence<min_confidence
                or not np.isfinite(confidence) or not np.isfinite(feature).all()):return None
        return feature

    def context_embedding(self,cameras,timestamp,dimension,min_confidence=.5):
        """View-conditioned query, same D as the frozen head; no GT/current ID."""
        views=[(camera,self.trusted_view(camera,timestamp,min_confidence)) for camera in sorted(set(cameras))]
        views=[(camera,feature) for camera,feature in views if feature is not None]
        if not views:return self.long_term_embedding
        candidate=np.average([f for _,f in views],axis=0,
                             weights=[self.short_term_confidence_by_camera[c] for c,_ in views])
        if np.linalg.norm(candidate)<1e-8:return self.long_term_embedding
        return normalized_feature(candidate,dimension)

    def update(self,observations,timestamp,*,trusted,spatial_trusted,dimension,momentum=.85,
               coordinate_views=False,update_gate=1.,memory_delta=None,multiview_identity_confirmed=False):
        usable=[o for o in observations if o.visible and o.feature_valid and o.score>=.5]
        if not usable:
            self.identity_update_status="NO_CURRENT_VISUAL_EVIDENCE";return
        old=self.long_term_embedding
        reference=old
        if reference is None and coordinate_views:
            reference=sorted(usable,key=lambda o:(-o.score,o.key))[0].roi_feature
        compatible=[o for o in usable if old is None or np.dot(old,o.roi_feature)>=.55]
        paired=(coordinate_views and multiview_identity_confirmed and spatial_trusted and len(usable)>1
                and all(o.world_state_update_eligible for o in usable))
        short_usable=[]
        for o in usable:
            previous=self.trusted_view(o.camera_id,timestamp)
            if (reference is None or np.dot(reference,o.roi_feature)>=.55 or paired
                    or (coordinate_views and previous is not None and np.dot(previous,o.roi_feature)>=.55)):
                short_usable.append(o)
        for o in short_usable:
            prev=self.short_term_embedding_by_camera.get(o.camera_id)
            # Uncertain identity is quarantined: it must not update even the
            # short-term descriptor of an already assigned historical person.
            if trusted:
                candidate=o.roi_feature if prev is None else .5*prev+.5*o.roi_feature
                if np.linalg.norm(candidate)<1e-8:continue
                self.short_term_embedding_by_camera[o.camera_id]=normalized_feature(candidate,dimension)
                self.short_term_confidence_by_camera[o.camera_id]=float(o.score)
                self.short_term_timestamp_by_camera[o.camera_id]=int(timestamp)
        if not trusted:
            self.identity_update_status="IDENTITY_UNCERTAIN_NO_LONG_TERM_UPDATE";return
        self.last_identity_confirmed_timestamp=int(timestamp)
        if old is None and coordinate_views:
            # The stable long-term reference need not average unlike views;
            # independently confirmed camera evidence remains available above.
            anchor=sorted(usable,key=lambda o:(-o.score,o.key))[0].roi_feature
            compatible=[o for o in usable if np.dot(anchor,o.roi_feature)>=.55]
        if old is None and compatible:
            self.long_term_embedding=normalized_feature(np.average([o.roi_feature for o in compatible],axis=0,
                weights=[o.score for o in compatible]),dimension)
            self.identity_confidence=float(max(o.score for o in compatible))
            self.identity_update_status="CURRENT_VISUAL_INITIALIZATION"
            return
        if not spatial_trusted:
            self.identity_update_status="SHORT_TERM_ONLY_SPATIAL_UNCERTAIN";return
        if not compatible:
            self.identity_update_status=("VIEW_SPECIFIC_UPDATE_LONG_TERM_REFERENCE_PRESERVED" if short_usable
                                         else "INCOMPATIBLE_VIEWS_QUARANTINED");return
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


def identity_similarity(query,observation,config,base=None):
    """A confirmed camera descriptor is identity evidence, not new supervision."""
    if query.identity_embedding is None or not observation.feature_valid:return None
    value=float(np.clip(query.identity_embedding@observation.roi_feature,-1,1)) if base is None else float(base)
    memory=getattr(query,'identity_state',None)
    if config.enable_multiview_coordination and memory is not None:
        short=memory.trusted_view(observation.camera_id,observation.timestamp,config.min_memory_score)
        if short is not None:value=max(value,float(np.clip(short@observation.roi_feature,-1,1)))
    return value
