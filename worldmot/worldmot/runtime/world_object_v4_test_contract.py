"""Explicit one-shot C1 TEST contract, separate from immutable R2 locks.

This module validates provenance and observes an unchanged C1 algorithm. It
does not import datasets, labels, evaluation code, or a training optimizer.
"""
from __future__ import annotations
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

from ..data.worldobject_cache import sha256
from ..models.world_object_v4_checkpoint import FULL_SHA, GATE_SHA, R2_SOURCE
from ..tracking.world_object_v4_variants import state_configs
from .experiment_authorization import source_fingerprint, source_commit, require_experiment_authorization

BASE_COMMIT = '44b59bc1198ab069f64023e7c8d1339e01077029'
LEGACY_FULL_SCOPE_SHA = '0fbb52166edf1c53e422e80707436787926893290842819aa378043c385f7dc6'
LEGACY_TRAIN_SCOPE_SHA = 'f1160f7122dcc4e33570dc5c415ec7ddc46072982b287d4eade01858c7a13a74'
OMEGA_SHA = 'c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934'
SCOPE_SCHEMA = 'worldmot_v4_c1_single_full_TEST_scope_v1'
LEGAL_TEST = (1,2,3,4,6,7,8,9,10,11,15,16,18,19)
TEST_TIMES = 29019
TEST_CANDIDATES = 548583


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.partial.'+str(os.getpid()))
    with temp.open('x') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    temp.replace(path)


def c1_config():
    config=state_configs(1024)['C1'];validate_c1_config(config);return config


def validate_c1_config(config):
    base=asdict(state_configs(1024)['C0']);actual=asdict(config)
    difference={k for k in base if actual[k]!=base[k]}
    if (difference!={'enable_motion_prediction'} or not config.enable_motion_prediction
            or config.enable_safe_identity_memory or config.enable_multiview_coordination
            or config.enable_omega_scheduler or config.geometry_policy!='LEGACY_HARD'
            or not config.use_geometry or not config.use_appearance or not config.head_geometry
            or not config.sync_geometry or not config.temporal_geometry or not config.spatial_state_update):
        raise ValueError('ONLY_REGISTERED_C1_IS_AUTHORIZED')
    return actual


def validate_legacy_header(lock):
    if (lock.get('schema')!='worldobject_joint_birth_frozen_scope_v3_1'
            or lock.get('source_sha256')!=R2_SOURCE
            or lock.get('method_id')!='FULL_WORLD_OBJECT_V3'
            or lock.get('omega_sha256')!=OMEGA_SHA
            or lock.get('splits',{}).get('test',{}).get('scene_count')!=22
            or lock['splits']['test'].get('timestamps')!=TEST_TIMES
            or set(lock['splits']['test'].get('scenes',{}))!=set(map(str,range(1,23)))):
        raise ValueError('ORIGINAL_R2_FULL_SCOPE_REQUIRED')


def require_scope(scope_path,authorization_path,operation,scene=None,variant='C1'):
    approval=require_experiment_authorization(authorization_path,operation,'test','FULL_WORLD_OBJECT_V3')
    auth=json.loads(Path(authorization_path).read_text());scope=json.loads(Path(scope_path).read_text())
    if (scope.get('schema')!=SCOPE_SCHEMA or scope.get('source_sha256')!=source_fingerprint()
            or scope.get('base_commit')!=BASE_COMMIT or scope.get('checkpoint_sha256')!=FULL_SHA
            or scope.get('gate_sha256')!=GATE_SHA or scope.get('legacy_full_scope_sha256')!=LEGACY_FULL_SCOPE_SHA
            or scope.get('legacy_train_scope_sha256')!=LEGACY_TRAIN_SCOPE_SHA
            or scope.get('config')!=validate_c1_config(c1_config())
            or scope.get('timestamps')!=TEST_TIMES or scope.get('candidates')!=TEST_CANDIDATES
            or set(scope.get('scenes',{}))!=set(map(str,range(1,23)))
            or auth.get('scope_lock_sha256')!=sha256(scope_path)
            or auth.get('one_exploratory_C1_full_TEST_only') is not True
            or auth.get('TEST_parameter_selection') is not False
            or auth.get('training_authorized') is not False):
        raise PermissionError('V4_C1_TEST_SCOPE_OR_AUTHORIZATION_MISMATCH')
    if variant!='C1' and not (operation=='preflight' and variant=='C0' and scene==1):
        raise PermissionError('NON_C1_VARIANT_FORBIDDEN')
    if scene is not None and str(scene) not in scope['scenes']:
        raise PermissionError('UNREGISTERED_TEST_SCENE')
    if operation=='preflight' and scene!=1:raise PermissionError('PREFLIGHT_SCENE1_ONLY')
    return scope,approval


def verify_scene_inputs(scope,scene):
    evidence=scope['scenes'][str(scene)]
    for item in evidence['files']:
        path=Path(item['path'])
        if sha256(path)!=item['sha256']:raise ValueError('FROZEN_TEST_INPUT_CHANGED: '+str(path))
    return evidence


def output_candidates(row):
    result={}
    for camera,items in row['tracks_2d_by_camera'].items():
        for item in items:
            key=int(camera),int(item['detection_id'])
            if key in result:raise ValueError('DUPLICATE_OUTPUT_DETECTION')
            if not item.get('visible') or not item.get('evaluation_eligible') or item.get('projected_from_world'):
                raise ValueError('CURRENT_OBSERVATION_OUTPUT_REQUIRED')
            result[key]=item
    return result


def compare_candidates(reference,result):
    left,right=output_candidates(reference),output_candidates(result)
    counts={'reference_candidates':len(left),'output_candidates':len(right),
            'missing_keys':len(left.keys()-right.keys()),'extra_keys':len(right.keys()-left.keys()),
            'bbox_differences':0,'score_differences':0}
    for key in left.keys()&right.keys():
        counts['bbox_differences']+=left[key]['bbox_xywh']!=right[key]['bbox_xywh']
        counts['score_differences']+=left[key]['score']!=right[key]['score']
    if any(counts[k] for k in ['missing_keys','extra_keys','bbox_differences','score_differences']):
        raise ValueError('FROZEN_DETECTION_OUTPUT_DIFFERENCE: '+json.dumps(counts))
    return counts


def observation_signature(observations):
    h=hashlib.sha256()
    for o in observations:
        h.update(np.asarray([o.camera_id,o.detection_id,o.timestamp],np.int64).tobytes())
        h.update(np.asarray(o.bbox_xywh,np.float64).tobytes());h.update(np.asarray(o.score,np.float64).tobytes())
        h.update(np.ascontiguousarray(o.roi_feature).tobytes())
    return h.hexdigest()


def common_fields_equal(reference,result,path='row'):
    """Only ignore added dictionary fields, never changed shared semantics."""
    if isinstance(reference,dict) and isinstance(result,dict):
        for key in reference.keys()&result.keys():common_fields_equal(reference[key],result[key],path+'.'+key)
        missing=reference.keys()-result.keys()
        if missing:raise AssertionError('MISSING_R2_FIELD '+path+': '+str(missing))
    elif isinstance(reference,list) and isinstance(result,list):
        if len(reference)!=len(result):raise AssertionError('R2_LIST_LENGTH '+path)
        for i,(a,b) in enumerate(zip(reference,result)):common_fields_equal(a,b,path+'['+str(i)+']')
    elif reference!=result:
        raise AssertionError('R2_COMMON_FIELD_DIFFERENCE '+path+': '+repr(reference)[:120]+' != '+repr(result)[:120])


def validate_motion_bank(bank,timestamp):
    count=0;without=0
    for q in bank.objects.values():
        if q.identity_state is not None:raise AssertionError('C3_IDENTITY_MEMORY_MUST_REMAIN_DISABLED')
        state=q.spatial_state
        if state is None:without+=1;continue
        x,p=state.state,state.covariance
        if x.shape!=(6,) or p.shape!=(6,6) or not np.isfinite(x).all() or not np.isfinite(p).all():
            raise AssertionError('NONFINITE_OR_INVALID_C1_STATE')
        if not np.allclose(p,p.T,rtol=1e-7,atol=1e-10) or np.linalg.eigvalsh(p).min()<=0:
            raise AssertionError('NON_POSITIVE_C1_COVARIANCE')
        if state.geometry_frame_id!='PERSISTENT_WORLD' or state.time_unit!='frame_id_tick':
            raise AssertionError('C1_COORDINATE_OR_TIME_UNIT_CHANGED')
        if state.last_prediction_timestamp>timestamp or state.last_trusted_geometry_timestamp>timestamp:
            raise AssertionError('FUTURE_MOTION_STATE')
        count+=1
    return {'finite_6d_states':count,'objects_without_trusted_3d_state':without}


class MotionObserver:
    """Transparent timers and numerical checks; original algorithms stay intact."""
    def __init__(self):self.total=Counter();self.current=Counter();self.effective={}
    def begin(self):self.current=Counter();self.effective={}
    @contextmanager
    def installed(self):
        from ..tracking.world_object_motion_state import PersistentWorldMotionState as State
        predict,update=State.predict,State.update
        observer=self
        def observed_predict(state,timestamp):
            raw=float(timestamp)-state.last_prediction_timestamp
            before=(state.state.copy(),state.covariance.copy()) if raw==0 else None
            start=time.perf_counter();result=predict(state,timestamp);elapsed=time.perf_counter()-start
            observer.current['predict_calls']+=1
            if raw>0:
                key=id(state),float(timestamp);observer.effective[key]=observer.effective.get(key,0)+1
                if observer.effective[key]!=1:raise AssertionError('DOUBLE_EFFECTIVE_PREDICTION')
                observer.current['effective_predict_calls']+=1;observer.current['kalman_predict_effective_s']+=elapsed
            elif raw==0:
                if not np.array_equal(before[0],state.state) or not np.array_equal(before[1],state.covariance):
                    raise AssertionError('DUPLICATE_PREDICTION_CHANGED_POSITION_OR_Q')
                observer.current['idempotent_predict_calls']+=1
            return result
        def observed_update(state,*args,**kwargs):
            start=time.perf_counter();result=update(state,*args,**kwargs)
            observer.current['kalman_update_s']+=time.perf_counter()-start
            observer.current['kalman_update_calls']+=1;observer.current['kalman_updates_accepted']+=int(result)
            return result
        State.predict,State.update=observed_predict,observed_update
        try:yield self
        finally:State.predict,State.update=predict,update
    def finish(self):self.total.update(self.current);return dict(self.current)


def install_no_gt_guard():
    """Forbid original annotations/raw GT in this inference process."""
    checked={'blocked_GT_reads':0}
    def audit(event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
        name=os.fsdecode(args[0]).replace('\\','/')
        if (('/VisionTrack/annotations/' in name and name.endswith('.json'))
                or ('/VisionTrack/' in name and ('/gt/' in name or name.endswith('/gt.txt')))):
            checked['blocked_GT_reads']+=1;raise PermissionError('TEST_GT_FORBIDDEN_DURING_INFERENCE')
    sys.addaudithook(audit);return checked


def completed_scene(path,scope,scene):
    path=Path(path);m=json.loads((path/'manifest.json').read_text());e=scope['scenes'][str(scene)]
    if (m.get('status')!='COMPLETE' or m.get('variant')!='C1' or m.get('scene')!=scene
            or m.get('timestamps')!=e['timestamps'] or m.get('candidate_count')!=e['candidates']
            or m.get('source_sha256')!=scope['source_sha256'] or m.get('checkpoint_sha256')!=FULL_SHA
            or m.get('tracking_config')!=scope['config'] or m.get('GT_in_inference') is not False
            or m.get('causal') is not True):raise ValueError('COMPLETED_SCENE_SCOPE_MISMATCH')
    for file,digest in m['output_sha256'].items():
        if sha256(path/file)!=digest:raise ValueError('COMPLETED_SCENE_OUTPUT_CHANGED')
    return m
