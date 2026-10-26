"""Safety boundaries for a one-shot complete TEST run, with numerical checks."""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
import numpy as np
import pytest

from worldmot.runtime.world_object_v4_test_contract import (
    c1_config,validate_c1_config,common_fields_equal,compare_candidates,
    validate_legacy_header,MotionObserver,validate_motion_bank,completed_scene,
    R2_SOURCE,OMEGA_SHA,FULL_SHA)
from worldmot.tracking.world_object_v4_variants import state_configs
from worldmot.tracking.world_object_motion_state import PersistentWorldMotionState
from worldmot.data.worldobject_cache import sha256


def test_exactly_one_c1_config_change():
    config=c1_config();assert config.enable_motion_prediction
    assert config.geometry_policy=='LEGACY_HARD' and not config.enable_safe_identity_memory
    for name in ['C0','C2','C3','C4','C5']:
        with pytest.raises(ValueError):validate_c1_config(state_configs(1024)[name])
    with pytest.raises(ValueError):validate_c1_config(replace(config,geometry_gate=7.))


def test_r2_source_and_complete_scope_cannot_be_silently_rebound():
    lock={'schema':'worldobject_joint_birth_frozen_scope_v3_1','source_sha256':R2_SOURCE,
          'method_id':'FULL_WORLD_OBJECT_V3','omega_sha256':OMEGA_SHA,
          'splits':{'test':{'scene_count':22,'timestamps':29019,'scenes':{str(i):{} for i in range(1,23)}}}}
    validate_legacy_header(lock)
    with pytest.raises(ValueError):validate_legacy_header({**lock,'source_sha256':'new-source'})
    with pytest.raises(ValueError):validate_legacy_header({**lock,'omega_sha256':'416-weight'})
    lock['splits']['test']['scenes'].pop('22')
    with pytest.raises(ValueError):validate_legacy_header(lock)


def row(ids=(1,2)):
    return {'tracks_2d_by_camera':{'1':[{'detection_id':i,'bbox_xywh':[10.,20.,30.,40.],
        'score':.8,'visible':True,'evaluation_eligible':True,'projected_from_world':False,
        'global_id':gid} for i,gid in enumerate(ids)]}}


def test_detection_audit_allows_only_identity_change():
    a=row();b=row((80,90));assert compare_candidates(a,b)['output_candidates']==2
    b['tracks_2d_by_camera']['1'].pop()
    with pytest.raises(ValueError):compare_candidates(a,b)
    b=row();b['tracks_2d_by_camera']['1'][0]['bbox_xywh'][0]+=1e-8
    with pytest.raises(ValueError):compare_candidates(a,b)
    b=row();b['tracks_2d_by_camera']['1'][0]['score']+=1e-8
    with pytest.raises(ValueError):compare_candidates(a,b)


def test_common_prediction_fields_ignore_added_metadata_only():
    a={'timestamp':1,'values':[{'ID':3,'position':[1.,2.,3.]}]}
    b={'timestamp':1,'values':[{'ID':3,'position':[1.,2.,3.],'new':True}],'variant':'C0'}
    common_fields_equal(a,b)
    b['values'][0]['ID']=4
    with pytest.raises(AssertionError):common_fields_equal(a,b)
    with pytest.raises(AssertionError):common_fields_equal(a,{'timestamp':1})


def test_motion_observer_proves_single_effective_propagation_and_joseph_psd():
    s=PersistentWorldMotionState.initialize([0.,0.,0.],np.eye(3),1)
    s.state[3:]=[1.,2.,3.]
    observer=MotionObserver();observer.begin()
    with observer.installed():
        s.predict(2);x=s.state.copy();p=s.covariance.copy()
        s.predict(2)
        np.testing.assert_array_equal(s.state,x);np.testing.assert_array_equal(s.covariance,p)
        np.testing.assert_allclose(s.position,[1.,2.,3.])
        assert s.update([1.1,2.1,3.1],np.eye(3),2)
        assert np.linalg.eigvalsh(s.covariance).min()>0
    counts=observer.finish()
    assert counts['effective_predict_calls']==1 and counts['idempotent_predict_calls']==2
    assert counts['kalman_update_calls']==1 and counts['kalman_updates_accepted']==1


def test_motion_finite_state_and_no_extra_identity_module():
    s=PersistentWorldMotionState.initialize([1.,2.,3.],np.eye(3),1)
    q=SimpleNamespace(identity_state=None,spatial_state=s)
    bank=SimpleNamespace(objects={1:q})
    assert validate_motion_bank(bank,1)['finite_6d_states']==1
    s.state[3]=np.nan
    with pytest.raises(AssertionError):validate_motion_bank(bank,1)
    s.state[3]=0;s.covariance[0,0]=-1
    with pytest.raises(AssertionError):validate_motion_bank(bank,1)


def test_completed_scene_is_readonly_and_checksum_verified(tmp_path):
    (tmp_path/'tracks.jsonl').write_text('frozen\n')
    scope={'source_sha256':'test-source','config':c1_config().to_dict(),
           'scenes':{'1':{'timestamps':2,'candidates':3}}}
    m={'status':'COMPLETE','variant':'C1','scene':1,'timestamps':2,'candidate_count':3,
       'source_sha256':'test-source','checkpoint_sha256':FULL_SHA,'tracking_config':scope['config'],
       'GT_in_inference':False,'causal':True,'output_sha256':{'tracks.jsonl':sha256(tmp_path/'tracks.jsonl')}}
    (tmp_path/'manifest.json').write_text(json.dumps(m))
    assert completed_scene(tmp_path,scope,1)==m
    (tmp_path/'tracks.jsonl').write_text('changed\n')
    with pytest.raises(ValueError):completed_scene(tmp_path,scope,1)


def test_gt_read_guard_denies_original_annotation_access_in_child(tmp_path):
    path=tmp_path/'VisionTrack/annotations/test.json';path.parent.mkdir(parents=True);path.write_text('{}')
    code="""from worldmot.runtime.world_object_v4_test_contract import install_no_gt_guard
import sys
guard=install_no_gt_guard()
try:
 open(sys.argv[1]).read()
except PermissionError:
 assert guard['blocked_GT_reads']==1
else:
 raise AssertionError('GT was readable')
"""
    subprocess.run([sys.executable,'-c',code,str(path)],check=True)
