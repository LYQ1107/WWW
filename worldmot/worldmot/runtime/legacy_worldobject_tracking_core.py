"""Explicit legacy control adapter, isolated from the formal V3 identity core."""
from __future__ import annotations
from dataclasses import replace
import numpy as np
from ..models.legacy_worldobject_builder_v1 import LegacyPairThenMergeBuilder
from ..tracking.legacy_worldobject_bank_v1 import LegacyPersistentWorldObjectBank
from ..models.world_object_observation import fuse_identity_features
from ..models.world_object_support import WorldObjectSupport

LEGACY_ROI_METHOD="LEGACY_PAIR_THEN_MERGE_ROI_RESTORED_CONTROL"


class LegacyWorldObjectTrackingCore:
    def __init__(self,config):
        self.config=config;self.method_id=LEGACY_ROI_METHOD
        self.bank=LegacyPersistentWorldObjectBank(min_hits=1);self.builder=LegacyPairThenMergeBuilder()
        self.scene_id=None;self.last_timestamp=None

    def reset(self,scene_id):
        self.scene_id=int(scene_id);self.bank.reset();self.last_timestamp=None

    def step(self,observations,timestamp):
        if self.last_timestamp is not None and timestamp<=self.last_timestamp:raise ValueError("NONCAUSAL_LEGACY_CONTROL")
        observations=sorted(observations,key=lambda o:o.key);prepared=[]
        for o in observations:
            if o.support is None:
                support=WorldObjectSupport(np.empty((0,3)),np.empty((0,3)),np.full(3,np.nan),np.eye(3)*1e6,0.,0.,"WORLD_ALIGNMENT_FAILED")
                o=replace(o,spatial_support=support)
            prepared.append(o)
        hypotheses,diagnostics=self.builder.build(prepared)
        for h in hypotheses:
            h.feature,_=fuse_identity_features(h.observations,self.config.feature_dim)
            if h.feature is None:raise ValueError("FEATURE_UNAVAILABLE_IN_LEGACY_ROI_CONTROL")
        objects,bank=self.bank.step(hypotheses,timestamp);outputs={}
        for index,h in enumerate(hypotheses):
            for o in h.observations:
                outputs.setdefault(str(o.camera_id),[]).append({"timestamp":int(timestamp),"camera_id":o.camera_id,
                    "detection_id":o.detection_id,"global_id":self.bank.last_assignments[index],"bbox_xywh":o.bbox_xywh.tolist(),
                    "score":o.score,"visible":True,"evaluation_eligible":True})
        self.last_timestamp=int(timestamp)
        return {"timestamp":int(timestamp),"scene_id":self.scene_id,"method_id":self.method_id,
            "tracks_2d_by_camera":outputs,"formation":diagnostics,"bank":bank,"causal":True,
            "GT_in_inference":False,"object_count":len(objects),"legacy_control":True}
