"""Real RGB front end with current Sim(3), shared with the cached V5 core."""
import time
import numpy as np
from .detection_first import DetectionFirstWorldMOT
from ..geometry.dual_view_static_alignment import DualCameraStaticAligner
from ..geometry.world_frame_contract import WorldFrameTransform
from ..geometry.world_alignment import Sim3
from ..models.geometry_interface import GeometryFrame
from ..models.world_object_v5_geometry import WorldGeometryContext
from ..data.worldobject_observations import live_observations


class OnlineObservationAnchoredTracker:
    def __init__(self,detector,geometry,core):
        self.detector=detector;self.geometry=geometry;self.core=core;self.config=core.config
        self.aligner=DualCameraStaticAligner(min_inliers=6);self.scene_id=None;self.previous=None;self.previous_world={};self.last_timings={}

    def reset(self,scene):
        self.scene_id=int(scene);self.core.reset(scene);self.aligner.reset();self.previous=None;self.previous_world={};self.last_timings={}

    def step(self,images,timestamp):
        if self.scene_id is None:raise ValueError("V5_LIVE_SCENE_RESET_REQUIRED")
        started=time.perf_counter();t=time.perf_counter();detections=self.detector.infer(images)
        self.last_timings={"detector_s":time.perf_counter()-t};transform=Sim3();aligned=False;reason="REGISTERED_NO_OMEGA_B4";alignment=None
        if self.config.use_geometry:
            t=time.perf_counter();frame=self.geometry.reconstruct(images,int(timestamp));self.last_timings["omega_wall_s"]=time.perf_counter()-t
            self.last_timings["omega_model_only_s"]=float(frame.diagnostics.get("model_only_s",0.))
            self.last_timings["omega_preprocess_s"]=float(frame.diagnostics.get("preprocess_s",0.))
            DetectionFirstWorldMOT._set_dynamic_masks(frame,detections);t=time.perf_counter()
            if frame.status=="OK":
                if self.previous is None:alignment=self.aligner.initialize()
                else:
                    self.previous.world_pointmaps=self.previous_world
                    alignment=self.aligner.update(self.previous,frame,sorted(frame.pointmaps))
                aligned=bool(alignment.result.success);transform=alignment.result.transform;reason=alignment.result.reason
            else:reason="INVALID_OMEGA_GEOMETRY"
            self.last_timings["world_alignment_s"]=time.perf_counter()-t
        else:
            frame=GeometryFrame(int(timestamp),images,{},{},{},np.empty((0,0)),{})
            self.last_timings.update(omega_wall_s=0.,omega_model_only_s=0.,omega_preprocess_s=0.,world_alignment_s=0.)
        t=time.perf_counter()
        observations=live_observations(detections,frame,WorldFrameTransform(transform,int(timestamp),aligned),use_geometry=self.config.use_geometry)
        evidence={"success":aligned,"reason":reason,"world_from_omega":transform.matrix().tolist()}
        if alignment is not None:evidence.update(inlier_ratio=alignment.result.inlier_ratio,residual_p95=alignment.result.residual_p95)
        context=WorldGeometryContext.from_record(timestamp,{"alignment":evidence,"camera_models":{str(v):c.to_dict() for v,c in frame.cameras.items()}})
        for o in observations:
            o.source={**o.source,"alignment_confidence":context.alignment_confidence,"reprojection_status":context.reprojection_status}
            if context.residual_p95 is not None:o.source["alignment_residual_p95"]=context.residual_p95
        self.last_timings["support_extraction_s"]=time.perf_counter()-t;t=time.perf_counter()
        result=self.core.step(observations,timestamp,context)
        self.last_timings["membership_and_state_s"]=time.perf_counter()-t;self.last_timings.update(self.core.last_timings)
        if aligned:
            self.previous=frame;self.previous_world={v:transform.apply(x.reshape(-1,3)).reshape(x.shape) for v,x in frame.pointmaps.items()}
        self.last_timings["tracking_total_s"]=time.perf_counter()-started
        result["alignment"]={"success":aligned,"reason":reason,"historical_transform_reused":False}
        result["runtime"]={"stages":dict(self.last_timings),"definition":"LIVE_RGB_TO_ID"}
        return result
