"""Shared causal identity core. No images, checkpoint loading, GT or future access."""
from __future__ import annotations
import numpy as np
import time
from ..tracking.persistent_world_object_bank import PersistentWorldObjectBank
from ..models.world_object_model_inputs import model_inputs


class WorldObjectTrackingCore:
    def __init__(self, config, model=None, device="cpu", *, training=False):
        self.config,self.model,self.device = config,model,str(device)
        self.bank = PersistentWorldObjectBank(config)
        self.scene_id = None
        self.training_context = bool(training)
        self.evidence_status = "RULE_BASED"
        self.last_timings = {}

    def reset(self, scene_id):
        self.scene_id = int(scene_id)
        self.bank.reset()

    def forward_evidence(self, observations, timestamp, training=False):
        self.validate_current(observations,timestamp)
        self.bank.predict(timestamp)
        if self.model is None:
            return None
        if len(self.bank.active())*len(observations)>self.config.max_model_pairs:
            if training:
                raise ValueError("TRAIN_MODEL_PAIR_LIMIT: reduce registered clip budget; never silently drop samples")
            self.evidence_status="LEARNED_PAIR_LIMIT_RULE_EVIDENCE_FALLBACK"
            return None
        self.evidence_status="V3_LEARNED_IDENTITY_ASSIGNMENT"
        import torch
        inputs = model_inputs(self.bank.active(), observations, self.config, timestamp, self.device)
        with torch.set_grad_enabled(training):
            return self.model(*inputs, use_geometry=self.config.head_geometry)

    def validate_current(self,observations,timestamp):
        if self.bank.last_timestamp is not None and int(timestamp)<=self.bank.last_timestamp:
            raise ValueError("NONCAUSAL_OR_DUPLICATE_TIMESTAMP")
        keys=[o.key for o in observations]
        if len(keys)!=len(set(keys)):raise ValueError("DUPLICATE_CAMERA_DETECTION_KEY")
        if len({o.camera_id for o in observations})>2:raise ValueError("V3_TWO_CAMERA_CONTRACT")
        if any(o.timestamp!=int(timestamp) for o in observations):raise ValueError("FUTURE_OR_STALE_OBSERVATION")

    def step(self, observations, timestamp, *, scene_id=None, evidence=None, training_decision=None):
        if scene_id is not None and int(scene_id) != self.scene_id:
            self.reset(scene_id)
        if self.scene_id is None:
            raise ValueError("SCENE_RESET_REQUIRED")
        observations = sorted(list(observations), key=lambda o:o.key)
        self.validate_current(observations,timestamp)
        if training_decision is not None and not self.training_context:
            raise ValueError("TEACHER_DECISION_FORBIDDEN_IN_INFERENCE")
        evidence_started = time.perf_counter()
        if evidence is None:
            evidence = self.forward_evidence(observations, timestamp)
        consumed={"assignment_logits","null_logits","existence_logits","birth_logits","update_gate",
                  "memory_delta","birth_pair_logits","birth_pair_mask"}
        learned = None if evidence is None else {
            k: (v.detach() if k=="memory_delta" and hasattr(v,"detach") else
                v.detach().cpu().numpy() if hasattr(v,"detach") else np.asarray(v))
            for k,v in evidence.items() if k in consumed}
        self.last_timings = {"head_and_evidence_transfer_s":time.perf_counter()-evidence_started}
        objects,diag = self.bank.step(observations,timestamp,learned,training_decision=training_decision)
        self.last_timings.update(self.bank.last_timings)
        diag["evidence_status"]=self.evidence_status
        tracks = {str(v):[] for v in sorted({o.camera_id for o in observations})}
        decisions = []
        for o in observations:
            gid = self.bank.last_assignments[o.key]
            q = self.bank.objects[gid]
            if o.visible:
                tracks[str(o.camera_id)].append({"timestamp":int(timestamp),"camera_id":o.camera_id,
                    "detection_id":o.detection_id,"bbox_xywh":o.bbox_xywh.tolist(),"global_id":gid,
                    "score":o.score,"visible":True,"evaluation_eligible":True,"status":q.status,
                    "output_source":"CURRENT_GMT_DETECTION", "projected_from_world":False})
            decisions.append({"camera_id":o.camera_id,"detection_id":o.detection_id,"global_id":gid,
                "geometry_status":o.geometry_status,"feature_status":"AVAILABLE" if o.feature_valid else "FEATURE_UNAVAILABLE",
                "point_support_available":o.point_support_available,"geometry_valid":o.geometry_valid,
                "geometry_reliability":o.geometry_reliability,"cross_view_fusion_eligible":o.cross_view_fusion_eligible,
                "world_state_update_eligible":o.world_state_update_eligible,
                "decision":q.decision_status,"source":o.source})
        result={"schema":"worldobject_prediction_v3","timestamp":int(timestamp),"scene_id":self.scene_id,
            "method_id":self.config.method_id,"tracks_2d_by_camera":tracks,"decisions":decisions,
            "tracks_3d":[{"global_id":q.global_id,"center":q.center.tolist() if q.geometry_valid else None,
                          "geometry_valid":q.geometry_valid,"status":q.status,"missed":q.missed,
                          "visible_cameras":list(q.visible_cameras)} for q in objects],
            "bank":diag,"causal":True,"GT_in_inference":False,
            "output_policy":"CURRENT_DETECTION_ONLY"}
        if self.config.enable_motion_prediction:
            for row,q in zip(result['tracks_3d'],objects):
                state=q.spatial_state
                row['spatial_state_status']=q.spatial_state_status
                if state is not None:
                    row.update(position=state.position.tolist(),velocity=state.velocity.tolist(),
                        covariance_6d=state.covariance.tolist(),motion_valid=state.motion_valid,
                        geometry_frame_id=state.geometry_frame_id,
                        last_prediction_timestamp=state.last_prediction_timestamp,
                        last_trusted_geometry_timestamp=state.last_trusted_geometry_timestamp,
                        current_trusted_3D_measurement=state.last_trusted_geometry_timestamp==int(timestamp),
                        geometry_confidence=state.geometry_confidence,
                        alignment_confidence=state.alignment_confidence,
                        time_unit=state.time_unit,spatial_unit='relative_world_unit')
                if q.identity_state is not None:
                    row['identity_state']={'identity_confidence':q.identity_state.identity_confidence,
                        'last_identity_confirmed_timestamp':q.identity_state.last_identity_confirmed_timestamp,
                        'identity_update_status':q.identity_state.identity_update_status,
                        'short_term_cameras':sorted(q.identity_state.short_term_embedding_by_camera)}
        return result
