"""Verify observer invariance and attribution against actual frozen decisions."""
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tests'))
from worldobject_synthetic import core,obs
from test_worldobject_memory import evidence
from worldobject_v3_spatial_temporal_attribution import FrozenDecisionObserver,Calibration
from worldmot.training.world_object_identity_targets import update_history_labels


def replay(method, observed):
    tracker=core(method);observer=FrozenDecisionObserver(tracker) if observed else None
    labels_by_gid={};outputs=[]
    frames=[[obs(1),obs(2)],[],
        [obs(1,center=(20,0,3),timestamp=2),obs(2,center=(20,0,3),timestamp=2)],
        [obs(1,center=(20,0,3),timestamp=3),obs(2,center=(20,0,3),timestamp=3)]]
    for timestamp,items in enumerate(frames):
        n=len(items);e=evidence(len(tracker.bank.active()),n)
        e['assignment_logits'][:]=8;e['null_logits'][:]=-2;e['birth_logits'][:]=-10
        e['birth_pair_logits']=np.full((n,n),10.)
        e['birth_pair_mask']=np.triu(np.ones((n,n),bool),1)
        labels=['PERSON_A']*n
        if observer:observer.start(items,labels,timestamp,labels_by_gid)
        result=tracker.step(items,timestamp,evidence=e)
        if observer:observer.finish(result)
        update_history_labels(labels_by_gid,tracker.bank.last_assignments,items,labels)
        outputs.append(result)
    return outputs,observer


def test_observer_keeps_actual_frozen_ID_decisions_and_state_identical():
    for method in ('FULL_WORLD_OBJECT_V3','HEAD_ONLY_NO_GEOMETRY_V3','FULL_NO_GEOMETRY_V3'):
        plain,_=replay(method,False);observed,_=replay(method,True)
        assert plain==observed


def test_observer_counts_correct_recovery_followed_by_stale_geometry_new_birth():
    _,observer=replay('FULL_WORLD_OBJECT_V3',True)
    c=observer.counts
    assert c['correct_selected_spatial_conflict_visual_recoveries']==1
    assert c['correct_spatial_recovery_next_frame_clean_observation_opportunities']==1
    assert c['correct_spatial_recovery_next_frame_new_ID']==1
    assert c['recovery_next_new_ID_with_geometry_rejection_and_stale_center']==1
    assert len(observer.examples['RECOVERY_NEXT_FRAME_ID_BREAK'])==1
    json.dumps(observer.report(),allow_nan=False)


def test_no_geometry_control_does_not_claim_spatial_recovery_or_chatter():
    _,observer=replay('FULL_NO_GEOMETRY_V3',True)
    assert observer.counts['correct_selected_spatial_conflict_visual_recoveries']==0
    assert observer.counts['correct_spatial_recovery_next_frame_new_ID']==0


def test_calibration_sufficient_statistics_merge_without_changing_ECE_or_Brier():
    first=Calibration();second=Calibration();whole=Calibration()
    first.add([.1,.9],[0,1]);second.add([.8],[0]);whole.add([.1,.9,.8],[0,1,0])
    first.rows+=second.rows
    assert first.report()==whole.report()
    assert abs(whole.report()['Brier']-.22)<1e-12
