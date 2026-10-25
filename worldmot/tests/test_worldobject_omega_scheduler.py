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
