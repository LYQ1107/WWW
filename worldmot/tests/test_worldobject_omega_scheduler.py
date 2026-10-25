from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
from worldobject_synthetic import obs,core,gid
from worldmot.tracking.world_object_config import WorldObjectConfig
from worldmot.runtime.omega_evidence_scheduler import OmegaEvidenceScheduler,REQUEST_OMEGA,SKIP_OMEGA


def test_scheduler_only_reads_visual_current_and_past():
    c=core(enable_motion_prediction=True);c.step([obs()],0)
    s=OmegaEvidenceScheduler(c.config);assert s.decide([obs()],[],0).action==REQUEST_OMEGA
    s.observe_alignment(0,True)
    class VisualOnly:
        feature_valid=True;roi_feature=np.array([1.,0,0]);camera_id=1
        def __getattr__(self,key):raise AssertionError('CURRENT_GEOMETRY_FORBIDDEN: '+key)
    decision=s.decide([VisualOnly()],c.bank.active(),1)
    assert decision.action==SKIP_OMEGA


def test_scheduler_requests_ambiguous_identity_and_old_geometry():
    c=core(enable_motion_prediction=True);c.step([obs(1,0),obs(1,1,feature=(1,.01,0),center=(10,0,3))],0)
    s=OmegaEvidenceScheduler(c.config);s.decide([obs()],[],0);s.observe_alignment(0,True)
    assert 'HISTORICAL_IDENTITY_COMPETITION' in s.decide([obs(timestamp=1)],c.bank.active(),1).reasons
    assert 'NO_RECENT_TRUSTED_GEOMETRY' in s.decide([],c.bank.active(),31).reasons


def test_scheduler_does_not_accept_future_alignment_or_duplicate_time():
    s=OmegaEvidenceScheduler(WorldObjectConfig(3));s.decide([],[],1)
    with pytest.raises(ValueError):s.observe_alignment(2,True)
    with pytest.raises(ValueError):s.decide([],[],1)


def test_skipped_omega_keeps_predictions_and_all_visual_detections():
    c=core(enable_motion_prediction=True,enable_safe_identity_memory=True,geometry_policy='UNCERTAINTY_SOFT')
    c.step([obs()],0)
    for t in range(1,5):
        r=c.step([obs(geometry=False,timestamp=t)],t)
        assert gid(r,1)==1 and c.bank.objects[1].spatial_state.last_trusted_geometry_timestamp==0


def test_live_scheduler_reconstructs_only_after_request_and_skips_without_old_pointmap():
    from test_worldobject_causal_runtime import synthetic_frontend
    from worldmot.runtime.online_world_object_tracker import OnlineWorldObjectTracker
    frame,detections=synthetic_frontend(1)
    class Detector:
        def infer(self,images):return detections
    class Geometry:
        calls=0
        def reconstruct(self,images,timestamp):self.calls+=1;raise AssertionError('should skip Ω')
    c=core(enable_motion_prediction=True,enable_safe_identity_memory=True,enable_omega_scheduler=True)
    c.step([obs(1),obs(2)],0)
    geometry=Geometry();tracker=OnlineWorldObjectTracker(Detector(),geometry,c.config,core=c)
    tracker.scene_id=1;tracker.scheduler.decide([],[],0);tracker.scheduler.observe_alignment(0,True)
    r=tracker.step(frame.images,1)
    assert geometry.calls==0 and r['omega_evidence_request']['action']==SKIP_OMEGA
    assert not any(x['geometry_valid'] for x in r['decisions'])
    assert all(x['geometry_status']=='OMEGA_SKIPPED_NO_CURRENT_GEOMETRY' for x in r['decisions'])
    assert all(x['source']['geometry_abstention_reason']=='SCHEDULER_SKIPPED_OMEGA' for x in r['decisions'])
    assert sum(map(len,r['tracks_2d_by_camera'].values()))==2


def test_keyframe_preserves_its_own_dynamic_masks_and_failure_abstains(monkeypatch):
    from test_worldobject_causal_runtime import synthetic_frontend
    from worldmot.geometry.anchored_world_alignment import AnchoredWorldAligner
    from worldmot.geometry.dual_view_static_alignment import DualAlignmentResult
    from worldmot.geometry.world_alignment import AlignmentResult,Sim3
    a,_=synthetic_frontend(0);a.dynamic_masks={1:np.ones((8,8),bool)}
    aligner=AnchoredWorldAligner();aligner.initialize(a,0)
    b,_=synthetic_frontend(10);b.dynamic_masks={1:np.zeros((8,8),bool)}
    def reject(old,current):
        assert old.dynamic_masks[1].all() and not current.dynamic_masks[1].any()
        return DualAlignmentResult(AlignmentResult(False,Sim3(),0,0,0,float('inf'),float('inf'),'failed'))
    monkeypatch.setattr(aligner.dual,'update',reject)
    result=aligner.update(b,10)
    assert not result.result.success and len(aligner.anchors)==1


@pytest.mark.parametrize('local_scale',[1.,2.])
def test_live_skip_then_fresh_reconstruction_reanchors_new_local_gauge(monkeypatch,local_scale):
    from worldmot.geometry.camera_model import CameraModel
    from worldmot.models.geometry_interface import GeometryFrame
    from worldmot.models.person_detector import PersonDetection
    from worldmot.runtime.online_world_object_tracker import OnlineWorldObjectTracker
    from worldmot.geometry.dual_view_static_alignment import DualAlignmentResult
    from worldmot.geometry.world_alignment import estimate_sim3
    def frame(timestamp):
        yy,xx=np.mgrid[:32,:32]
        depth=np.where((xx>=8)&(xx<24)&(yy>=8)&(yy<24),3.,6.)
        points=np.stack([(xx-16)*depth/32,(yy-16)*depth/32,depth],axis=-1)
        shift=10. if timestamp>=3 else 0.
        if timestamp>=3:points*=local_scale
        points[:,:,0]+=shift
        pose=np.eye(4);pose[0,3]=shift
        camera=CameraModel(np.array([[32.,0,16],[0,32.,16],[0,0,1.]]),pose,(32,32),(32,32),np.eye(3))
        image=np.zeros((32,32,3),np.uint8)
        return GeometryFrame(timestamp,{1:image,2:image},{1:points,2:points},
            {1:np.ones((32,32))*4,2:np.ones((32,32))*4},{},np.empty((0,0)),{1:camera,2:camera},
            image_to_model={1:np.eye(3),2:np.eye(3)})
    class Detector:
        def infer(self,images):
            result={}
            for camera in (1,2):
                d=PersonDetection(camera,0,(4,4,24,24),.9)
                d.detector_roi_feature=np.array([1.,0,0]);result[camera]=[d]
            return result
    class Geometry:
        def __init__(self):self.calls=[]
        def reconstruct(self,images,timestamp):self.calls.append(timestamp);return frame(timestamp)
    c=core(enable_motion_prediction=True,enable_safe_identity_memory=True,enable_multiview_coordination=True,
           enable_omega_scheduler=True,geometry_policy='UNCERTAINTY_SOFT',geometry_stale_ticks=3)
    geometry=Geometry();tracker=OnlineWorldObjectTracker(Detector(),geometry,c.config,core=c);tracker.reset(1)
    def align(old,current):
        # Exact synthetic background correspondences, not GT identities. The
        # new reconstruction's +10 local gauge must actually be estimated out.
        indices=([0,0,31,31],[0,31,0,31])
        result=estimate_sim3(current.pointmaps[1][indices],old.world_pointmaps[1][indices],
                             threshold=1e-6,min_inliers=3)
        np.testing.assert_allclose(result.transform.scale,1/local_scale,atol=1e-8)
        np.testing.assert_allclose(result.transform.translation,[-10/local_scale,0,0],atol=1e-8)
        return DualAlignmentResult(result)
    monkeypatch.setattr(tracker.anchor_aligner.dual,'update',align)
    initial=tracker.step(frame(0).images,0);center=c.bank.objects[1].center.copy()
    assert gid(initial,1)==gid(initial,2)==1
    for timestamp in (1,2):
        row=tracker.step(frame(timestamp).images,timestamp)
        assert row['omega_evidence_request']['action']==SKIP_OMEGA
        assert c.bank.objects[1].spatial_state.last_trusted_geometry_timestamp==0
    row=tracker.step(frame(3).images,3)
    assert geometry.calls==[0,3] and row['alignment']['success']
    assert not row['alignment']['historical_transform_reused']
    assert gid(row,1)==gid(row,2)==1
    assert c.bank.objects[1].spatial_state.last_trusted_geometry_timestamp==3
    np.testing.assert_allclose(c.bank.objects[1].center,center,atol=1e-8)
