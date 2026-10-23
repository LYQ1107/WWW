"""Deterministic geometric oracle used only in unit tests, never real metrics."""
from types import SimpleNamespace
import numpy as np
from worldmot.models.person_detector import PersonDetection


class PointOracle:
    def __init__(self,confidence=1.,cross_scale=1.):self.calls=[];self.confidence=confidence;self.cross_scale=cross_scale
    def track(self,images,query_points,*,image_keys,reference_key,current_timestamp):
        assert tuple(image_keys[0])==tuple(reference_key)
        assert all(k[1]<=current_timestamp for k in image_keys)
        self.calls.append((tuple(image_keys),tuple(reference_key),current_timestamp))
        source,stamp=reference_key
        xy=np.stack([np.asarray(query_points)+[70*(camera-source)+2*(time-stamp),0] for camera,time in image_keys]).astype(np.float32)
        return SimpleNamespace(xy=xy,visibility=np.full(xy.shape[:2],self.confidence),
            confidence=np.stack([np.full(len(query_points),self.confidence*(self.cross_scale if camera!=source else 1.)) for camera,_ in image_keys]),valid=np.ones(xy.shape[:2],bool),
            timings={'e2e_s':0.},diagnostics={'status':'UNIT_TEST_ORACLE'})


def scene_detections(timestamp=1,views=(1,2)):
    return {v:[PersonDetection(v,0,(10+70*(v-1)+2*(timestamp-1),10,20,40),.9,
                              source='gmt_pre_gta_detector')] for v in views}


def images(views=(1,2)):
    return {v:np.zeros((80,180,3),np.uint8) for v in views}
