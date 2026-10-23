from types import SimpleNamespace
import numpy as np
from worldmot.models.person_detector import PersonDetection
from worldmot.runtime.detection_first import DetectionFirstWorldMOT
from worldmot.models.detection_to_world import lift_detection,_mask_for_detection,_mask_coverage


def test_bbox_only_current_detection_does_not_claim_an_instance_mask():
    detection=PersonDetection(1,0,(10,20,30,40),.9,source='gmt_pre_gta_detector')
    observation=SimpleNamespace(detections=[detection])
    instances=[SimpleNamespace(proposal_id=0)]
    support=DetectionFirstWorldMOT._current_detection_support([observation],instances,{0:17})
    current=DetectionFirstWorldMOT._visibility_evidence(17,1,support)
    assert current['evaluation_eligible'] and current['native_view_detection']
    assert not current['native_view_mask']
    assert current['visibility_evidence']=='current_target_detection_bbox'
    # An old mask in another camera cannot create a current detection entry.
    other=DetectionFirstWorldMOT._visibility_evidence(17,2,support)
    assert not other['evaluation_eligible'] and not other['native_view_mask']
    assert other['visibility_evidence']=='projection_only'


def test_real_mask_and_actual_memory_proposal_mapping_are_preserved():
    mask=np.ones((80,80),bool)
    observations=[SimpleNamespace(detections=[]),SimpleNamespace(detections=[
        PersonDetection(2,0,(10,20,30,40),.9,mask=mask)])]
    support=DetectionFirstWorldMOT._current_detection_support(observations,[SimpleNamespace(proposal_id=1)],{0:23})
    actual=DetectionFirstWorldMOT._visibility_evidence(23,2,support)
    assert actual['native_view_mask'] and actual['evaluation_eligible']
    assert actual['visibility_state']=='CURRENT_MASK_AND_PROJECTION'


def test_empty_segmentation_cannot_claim_mask_visibility():
    observation=SimpleNamespace(detections=[PersonDetection(1,0,(0,0,10,10),.9,mask=np.zeros((10,10),bool))])
    support=DetectionFirstWorldMOT._current_detection_support([observation],[SimpleNamespace(proposal_id=0)],{0:1})
    assert not DetectionFirstWorldMOT._visibility_evidence(1,1,support)['native_view_mask']


def test_lifting_keeps_missing_real_mask_and_preserves_exact_bbox_sampling_geometry():
    yy,xx=np.indices((16,16))
    points=np.stack([xx,yy,np.ones_like(xx)],axis=-1).astype(float)
    frame=SimpleNamespace(images={1:np.zeros((16,16,3),np.uint8)},pointmaps={1:points},
        confidences={1:np.full((16,16),2.)},diagnostics={})
    det=PersonDetection(1,0,(3.2,4.3,6.7,7.1),.9,source='gmt_pre_gta_detector')
    rectangle=_mask_for_detection(det,(16,16))
    legacy=PersonDetection(1,0,det.bbox_xywh,.9,mask=rectangle)
    current=lift_detection(det,frame,min_points=3)
    reference=lift_detection(legacy,frame,min_points=3)
    assert det.mask is None and det.geometry_support_source=='bbox_rectangle'
    assert np.array_equal(current.support_points,reference.support_points)
    assert np.array_equal(current.center,reference.center)
    assert np.array_equal(current.covariance,reference.covariance)
    assert current.to_world_instance(0).masks_by_camera=={}
    assert current.to_world_instance(0).source=='gmt-pre-gta+omega512'
    camera=SimpleNamespace(original_size=(16,16),project=lambda pts,clip=False:(pts[:,:2],np.ones(len(pts),bool)))
    probes=np.array([[3.0,5.0,1.],[10.,11.,1.],[2.,3.,1.]])
    assert _mask_coverage(probes,det,camera)==_mask_coverage(probes,legacy,camera)
