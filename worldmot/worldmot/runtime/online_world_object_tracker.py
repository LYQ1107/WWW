"""RGB front-end; identity decisions are delegated to WorldObjectTrackingCore."""
from __future__ import annotations
import time
import numpy as np
from .world_object_tracking_core import WorldObjectTrackingCore
from .detection_first import DetectionFirstWorldMOT
from ..geometry.dual_view_static_alignment import DualCameraStaticAligner
from ..geometry.world_frame_contract import WorldFrameTransform
from ..geometry.world_alignment import Sim3
from ..models.geometry_interface import GeometryFrame
from ..data.worldobject_observations import live_observations


class OnlineWorldObjectTracker:
    def __init__(self, detector, geometry, config, *, core=None, device="cpu", min_confidence=1.5):
        self.detector,self.geometry,self.config=detector,geometry,config
        self.core=core or WorldObjectTrackingCore(config,device=device)
        self.bank=self.core.bank
        self.min_confidence=float(min_confidence)
        self.aligner=DualCameraStaticAligner(min_inliers=6)
        self.previous=None;self.previous_world={};self.scene_id=None
        self.last_timings={}

    def reset(self,scene_id):
        self.scene_id=int(scene_id)
        self.core.reset(scene_id);self.aligner.reset()
        self.previous=None;self.previous_world={};self.last_timings={}

    def step_observations(self, observations, timestamp):
        """Synthetic/live adapter seam; exactly the same core as cache replay."""
        return self.core.step(observations,timestamp)

    def step(self,images_by_camera,timestamp):
        if self.scene_id is None:
            raise ValueError("SCENE_RESET_REQUIRED")
        started=time.perf_counter();t=time.perf_counter()
        detections=self.detector.infer(images_by_camera)
        self.last_timings={"detector":time.perf_counter()-t}
        aligned=False;reason="VISUAL_ONLY_FALLBACK";transform=Sim3()
        if self.config.use_geometry:
            t=time.perf_counter()
            frame=self.geometry.reconstruct(images_by_camera,int(timestamp))
            self.last_timings["omega"]=time.perf_counter()-t
            self.last_timings["omega_model_only_s"]=float(frame.diagnostics.get("model_only_s",0.))
            self.last_timings["omega_preprocess_s"]=float(frame.diagnostics.get("preprocess_s",0.))
            DetectionFirstWorldMOT._set_dynamic_masks(frame,detections)
            t=time.perf_counter()
            if frame.status=="OK":
                if self.previous is None:
                    alignment=self.aligner.initialize()
                else:
                    self.previous.world_pointmaps=self.previous_world
                    alignment=self.aligner.update(self.previous,frame,sorted(frame.pointmaps))
                aligned=bool(alignment.result.success)
                transform=alignment.result.transform
                reason=alignment.result.reason
            else:
                reason="INVALID_OMEGA_GEOMETRY"
            self.last_timings["alignment"]=time.perf_counter()-t
        else:
            frame=GeometryFrame(int(timestamp),images_by_camera,{},{},{},np.empty((0,0)),{})
            self.last_timings.update(omega=0.,alignment=0.)
        t=time.perf_counter()
        observations=live_observations(detections,frame,WorldFrameTransform(transform,int(timestamp),aligned),
                                      min_confidence=self.min_confidence,use_geometry=self.config.use_geometry)
        self.last_timings["support"]=time.perf_counter()-t
        t=time.perf_counter();result=self.step_observations(observations,timestamp)
        self.last_timings["assignment_and_state_update"]=time.perf_counter()-t
        self.last_timings.update(getattr(self.core,"last_timings",{}))
        if aligned:
            self.previous=frame
            self.previous_world={v:transform.apply(x.reshape(-1,3)).reshape(x.shape) for v,x in frame.pointmaps.items()}
        self.last_timings["total"]=time.perf_counter()-started
        result["alignment"]={"success":aligned,"reason":reason,"historical_transform_reused":False}
        result["runtime"]={"stages":dict(self.last_timings),"fps_definition":"LIVE_RGB_TO_ID"}
        return result
