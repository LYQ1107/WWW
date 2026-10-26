"""Observation-anchored causal V5 core; no GT or file/checkpoint access."""
import time
import numpy as np
from ..tracking.observation_anchored_world_object_bank import ObservationAnchoredWorldObjectBank
from ..models.world_object_membership_head import membership_inputs


class ObservationAnchoredWorldObjectCore:
    def __init__(self,config,model,device="cpu"):
        self.config=config;self.model=model;self.device=str(device);self.scene_id=None
        self.bank=ObservationAnchoredWorldObjectBank(config);self.last_timings={}
        self.scene_reset_count=0

    def reset(self,scene):
        self.scene_id=int(scene);self.bank.reset();self.scene_reset_count=1

    def forward_evidence(self,observations,timestamp,context,training=False):
        if self.scene_id is None:raise ValueError("V5_SCENE_RESET_REQUIRED")
        if self.bank.last_timestamp is not None and timestamp<=self.bank.last_timestamp:raise ValueError("V5_NONCAUSAL_TIMESTAMP")
        if any(o.timestamp!=timestamp for o in observations) or context.timestamp!=timestamp:raise ValueError("V5_CURRENT_INPUTS_ONLY")
        predicted=time.perf_counter();self.bank.predict(timestamp);predict_s=time.perf_counter()-predicted
        history=self.bank.active();import torch
        started=time.perf_counter()
        if self.config.frozen_legacy_membership:
            from ..models.world_object_model_inputs import model_inputs
            from ..tracking.world_object_v4_variants import state_configs
            inputs=model_inputs(history,observations,state_configs(1024)["C1"],timestamp,self.device)
            if str(self.device).startswith("cuda"):torch.cuda.synchronize()
            measured=time.perf_counter()
            with torch.no_grad():old=self.model(*inputs)
            if str(self.device).startswith("cuda"):torch.cuda.synchronize()
            model_s=time.perf_counter()-measured
            # This registered frozen control consumes no old birth, existence,
            # update gate, memory delta or learned pair-birth predictions.
            n=len(observations)
            slots=torch.as_tensor([o.camera_id-1 for o in observations],device=self.device)
            result={"membership_logits":old["assignment_logits"],"no_observation_logits":old["null_logits"][:,None].expand(-1,2)}
            geometry=None
        else:
            inputs=membership_inputs(history,observations,context,self.config,self.device)
            if not training and str(self.device).startswith("cuda"):torch.cuda.synchronize()
            measured=time.perf_counter()
            with torch.set_grad_enabled(training):result=self.model(*inputs)
            if not training and str(self.device).startswith("cuda"):torch.cuda.synchronize()
            model_s=time.perf_counter()-measured
            geometry=inputs[2].detach().cpu().numpy()
        self.last_timings={"membership_forward_and_input_s":time.perf_counter()-started,"C1_motion_predict_s":predict_s}
        if not training:self.last_timings["membership_model_only_s"]=model_s
        return result,geometry

    def step(self,observations,timestamp,context,*,evidence=None,geometry=None):
        observations=sorted(observations,key=lambda o:o.key)
        if evidence is None:evidence,geometry=self.forward_evidence(observations,timestamp,context)
        score=evidence["membership_logits"].detach().cpu().numpy()
        null=evidence["no_observation_logits"].detach().cpu().numpy()
        if score.shape!=(len(self.bank.active()),len(observations)) or null.shape!=(len(self.bank.active()),2):raise ValueError("V5_MEMBERSHIP_EVIDENCE_SHAPE")
        margins=score-null[:,np.asarray([o.camera_id-1 for o in observations],int)]
        objects,diagnostics=self.bank.step(observations,timestamp,margins,context,geometry)
        self.last_timings.update(self.bank.last_timings)
        tracks={str(v):[] for v in sorted({o.camera_id for o in observations})};decisions=[]
        for o in observations:
            gid=self.bank.last_assignments[o.key];q=self.bank.objects[gid];record=self.bank.last_detection_decisions[o.key]
            if record.get("output_eligible",True) and o.visible:
                tracks[str(o.camera_id)].append({"timestamp":int(timestamp),"camera_id":o.camera_id,"detection_id":o.detection_id,
                    "bbox_xywh":o.bbox_xywh.tolist(),"score":o.score,"global_id":gid,"visible":True,"evaluation_eligible":True,
                    "status":q.status,"output_source":"CURRENT_GMT_DETECTION","projected_from_world":False})
            decisions.append({"camera_id":o.camera_id,"detection_id":o.detection_id,"global_id":gid,**record,
                "geometry_status":o.geometry_status,"geometry_valid":o.geometry_valid,"geometry_reliability":o.geometry_reliability,
                "point_support_available":o.point_support_available,"cross_view_fusion_eligible":o.cross_view_fusion_eligible,
                "world_state_update_eligible":o.world_state_update_eligible})
        for camera,items in tracks.items():
            if len({o["global_id"] for o in items})!=len(items):raise AssertionError("ONE_CURRENT_OBSERVATION_PER_CAMERA_PER_ID")
        rows=[]
        for q in objects:
            row={"global_id":q.global_id,"center":q.center.tolist() if q.geometry_valid else None,"geometry_valid":q.geometry_valid,
                 "status":q.status,"missed":q.missed,"visible_cameras":list(q.visible_cameras),"identity_update_status":q.identity_update_status,
                 "last_identity_write":q.last_identity_write,"spatial_state_status":q.spatial_state_status}
            s=q.spatial_state
            if s is not None:row.update(position=s.position.tolist(),velocity=s.velocity.tolist(),covariance_6d=s.covariance.tolist(),
                motion_valid=s.motion_valid,last_prediction_timestamp=s.last_prediction_timestamp,last_trusted_geometry_timestamp=s.last_trusted_geometry_timestamp,
                current_trusted_3D_measurement=s.last_trusted_geometry_timestamp==timestamp,geometry_confidence=s.geometry_confidence,time_unit=s.time_unit)
            rows.append(row)
        return {"schema":"worldobject_prediction_v5","variant":self.config.variant,"scene_id":self.scene_id,"timestamp":int(timestamp),
                "method_id":"V5_"+self.config.variant,"tracks_2d_by_camera":tracks,"decisions":decisions,"tracks_3d":rows,"bank":diagnostics,
                "geometry_context":{"reprojection_status":context.reprojection_status,"alignment_success":context.alignment_success,"alignment_confidence":context.alignment_confidence},
                "alias_events":[],"scene_reset_count":self.scene_reset_count,"causal":True,"GT_in_inference":False,"output_policy":"CURRENT_DETECTION_ONLY"}
