import numpy as np
from worldmot.models.person_detector import PersonDetection
from worldmot.models.omega_object_features import FrozenObjectFeatureExtractor


def test_gmt_roi_branch_preserves_real_frozen_features_and_source():
    det=PersonDetection(1,0,(4,4,24,40),.8,source='gmt_pre_gta_detector')
    det.detector_roi_feature=np.linspace(-1,1,1024,dtype=np.float32)
    image=np.full((64,48,3),100,np.uint8)
    omega=np.ones((64,8,6),np.float32)
    item=FrozenObjectFeatureExtractor(require_intermediate=True).extract(
        {1:[det]},{1:image},omega_patch_features={1:omega})[1,0]
    assert item.valid and item.detector_feature.shape==(128,)
    assert np.linalg.norm(item.detector_feature)>.99
    assert item.source=='rgb+gmt_roi+omega_patch'
    assert item.diagnostics['detector_roi_available']
    assert not item.diagnostics['rtmdet_roi_available']
    assert item.appearance_feature.shape==(512,)


def test_original_rtmdet_feature_source_remains_compatible():
    det=PersonDetection(1,0,(4,4,24,40),.8)
    item=FrozenObjectFeatureExtractor(require_intermediate=True).extract(
        {1:[det]},{1:np.ones((64,48,3),np.uint8)},
        detector_feature_maps={1:np.ones((128,8,6),np.float32)},
        omega_patch_features={1:np.ones((256,8,6),np.float32)})[1,0]
    assert item.valid and item.source=='rgb+rtmdet_roi+omega_patch'
    assert item.diagnostics['rtmdet_roi_available']
