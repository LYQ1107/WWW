"""Only synthetic observations: no dataset, pretrained detector or geometry model."""
import numpy as np
from worldmot.models.world_object_observation import WorldObjectObservation
from worldmot.models.world_object_support import WorldObjectSupport
from worldmot.tracking.world_object_config import WorldObjectConfig
from worldmot.runtime.world_object_tracking_core import WorldObjectTrackingCore


def obs(camera=1,det=0,center=(0,0,3),feature=(1.,0.,0.),timestamp=0,geometry=True,score=.9):
    center=np.asarray(center,float)
    points=center+np.array([[i*.001,j*.001,k*.001] for i in range(3) for j in range(2) for k in range(2)])
    support=WorldObjectSupport(points,points,center,np.eye(3)*.01,1.,1.,"DEPTH_CLUSTERED",
                               coordinate_frame="PERSISTENT_WORLD",depth_source="SYNTHETIC_CAMERA_DEPTH")
    return WorldObjectObservation(camera,det,np.array([10+det*20,10,10,20]),score,
        None if feature is None else np.array(feature,np.float32),feature is not None,
        center,np.eye(3)*.01,support,geometry,1. if geometry else 0.,timestamp,
        geometry_status="GEOMETRY_VALID" if geometry else "WORLD_ALIGNMENT_FAILED")


def core(method="FULL_WORLD_OBJECT_V3",**kwargs):
    c=WorldObjectTrackingCore(WorldObjectConfig(3,method,min_hits=1,**kwargs))
    c.reset(1)
    return c


def gid(result,camera,det=0):
    return next(row["global_id"] for row in result["tracks_2d_by_camera"][str(camera)] if row["detection_id"]==det)
