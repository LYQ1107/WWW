#!/usr/bin/env python3
"""One explicitly authorized, frozen C1 TEST run. Never loads TEST labels.

Scene1 C0 is an engineering reference only. C1 Scene1 is retained as the first
official scene, not discarded/reselected. No scene-internal reset or clip loop.
"""
from __future__ import annotations
import argparse
from collections import Counter
from contextlib import ExitStack
from datetime import datetime,timezone
import fcntl
import hashlib
import importlib
import importlib.util
import io
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import tarfile
import time
import traceback
import numpy as np

from worldmot.runtime.world_object_v4_test_contract import (
    BASE_COMMIT,LEGACY_FULL_SCOPE_SHA,LEGACY_TRAIN_SCOPE_SHA,OMEGA_SHA,SCOPE_SCHEMA,
    TEST_TIMES,TEST_CANDIDATES,FULL_SHA,GATE_SHA,R2_SOURCE,canonical_sha,write_json,
    c1_config,validate_legacy_header,require_scope,verify_scene_inputs,compare_candidates,
    observation_signature,common_fields_equal,validate_motion_bank,MotionObserver,
    install_no_gt_guard,completed_scene)
from worldmot.runtime.experiment_authorization import source_commit,source_fingerprint,APPROVAL_SCHEMA
from worldmot.data.worldobject_cache import sha256,WorldObjectCache
from worldmot.data.worldobject_observations import replay_observations
from worldmot.models.gmt_detection_cache import ImmutableGMTDetectionCache
from worldmot.models.world_object_v4_checkpoint import load_frozen_r2_head
from worldmot.runtime.world_object_tracking_core import WorldObjectTrackingCore
from worldmot.tracking.world_object_v4_variants import state_configs


def now():return datetime.now(timezone.utc).isoformat()


def register(a):
    root=a.output;scope_path=root/'scope_v4_c1_TEST.json';auth_path=root/'authorization_v4_c1_TEST.json'
    if scope_path.exists() or auth_path.exists():
        require_scope(scope_path,auth_path,'replay');return
    r2=a.r2_root;inputs=r2.parent;raw_root=inputs/'gmt_detector_comparison_20261023/raw_test_v1'
    full=root/'input_provenance/FULL_WORLD_OBJECT_V3.full.json'
    train=root/'input_provenance/FULL_WORLD_OBJECT_V3.train.json'
    if sha256(full)!=LEGACY_FULL_SCOPE_SHA or sha256(train)!=LEGACY_TRAIN_SCOPE_SHA:
        raise ValueError('ORIGINAL_UNMODIFIED_R2_SCOPE_CHECKSUM_REQUIRED')
    legacy=json.loads(full.read_text());validate_legacy_header(legacy)
    gate_path=r2/'train/FULL_WORLD_OBJECT_V3/train_gate.json';gate=json.loads(gate_path.read_text())
    if sha256(gate_path)!=GATE_SHA or gate['scope_lock_sha256']!=sha256(train):
        raise ValueError('R2_TRAIN_GATE_SCOPE_BINDING_CHANGED')
    if sha256(gate_path.parent/gate['checkpoint'])!=FULL_SHA:raise ValueError('R2_WEIGHTS_CHANGED')
    mirror=inputs/'mirror_receipt.json';receipt=json.loads(mirror.read_text());plan_path=inputs/'mirror_plan.json'
    if (receipt.get('status')!='VERIFIED' or not receipt.get('all_original_file_checksums_verified')
            or sha256(plan_path)!=receipt['plan_sha256']):raise ValueError('VERIFIED_ORIGINAL_INPUT_MIRROR_REQUIRED')
    plan={x['relative']:x for x in json.loads(plan_path.read_text())['entries']}
    def proof(path,expected=None):
        path=Path(path);relative=str(path.relative_to(inputs));digest=sha256(path)
        if relative not in plan or plan[relative]['sha256']!=digest or (expected and digest!=expected):
            raise ValueError('INPUT_DIFFERS_FROM_ORIGINAL_VERIFIED_MIRROR: '+relative)
        return {'path':str(path.resolve()),'sha256':digest,'bytes':path.stat().st_size}
    raw=ImmutableGMTDetectionCache(raw_root)
    if raw.feature_dim!=1024 or raw.lock.get('split')!='test':raise ValueError('FROZEN_TEST_ROI1024_REQUIRED')
    scenes={};total_times=0;total_candidates=0
    for sid in range(1,23):
        support=r2/f'cache_test_v2/scene_{sid:02d}';candidate=raw_root/f'scene_{sid:02d}'
        cm=json.loads((support/'manifest.json').read_text());rm=json.loads((candidate/'manifest.json').read_text())
        original=r2/f'predictions/FULL_WORLD_OBJECT_V3/scene_{sid:02d}'
        pm=json.loads((original/'manifest.json').read_text())
        expected=legacy['splits']['test']['scenes'][str(sid)]
        original_config=state_configs(1024)['C0'].to_dict()
        if any(original_config.get(k)!=v for k,v in pm['tracking_config'].items()):
            raise ValueError('R2_SHARED_TRACKING_CONFIGURATION_CHANGED')
        files=[proof(support/'manifest.json',expected['manifest_sha256']),proof(candidate/'manifest.json'),
               proof(original/'manifest.json'),proof(original/'tracks.jsonl',pm['tracks_sha256'])]
        times=[]
        for shard in cm['shards']:
            p=support/shard['file'];files.extend([proof(p,shard['sha256']),proof(p.with_suffix('.json'),shard['sidecar_sha256'])])
        for shard in rm['shards']:
            p=candidate/shard['file'];files.extend([proof(p,shard['sha256']),proof(p.with_suffix('.json'))]);times.extend(shard['timestamps'])
        if (cm['status']!='COMPLETE' or rm['status']!='COMPLETE' or pm['status']!='COMPLETE'
                or cm['split']!='test' or rm['split']!='test' or cm.get('gt_in_cache') is not False
                or rm.get('GT_in_cache') is not False or rm.get('track_ids_in_cache') is not False
                or rm['signature']!=raw.lock or pm['cache_manifest_sha256']!=sha256(support/'manifest.json')
                or cm['timestamps']!=expected['timestamps'] or pm['timestamps']!=expected['timestamps']
                or times!=list(range(1,expected['timestamps']+1)) or rm['frames']!=expected['timestamps']):
            raise ValueError('INCOMPLETE_OR_CHANGED_FROZEN_TEST_SCENE '+str(sid))
        scenes[str(sid)]={'scene':sid,'timestamps':len(times),'first_timestamp':times[0],
            'last_timestamp':times[-1],'candidates':rm['observations'],
            'cache_manifest_sha256':sha256(support/'manifest.json'),
            'raw_manifest_sha256':sha256(candidate/'manifest.json'),
            'original_R2_tracks_sha256':pm['tracks_sha256'],'original_R2_prediction':str(original/'tracks.jsonl'),
            'files':files}
        total_times+=len(times);total_candidates+=rm['observations']
    if (total_times,total_candidates)!=(TEST_TIMES,TEST_CANDIDATES):raise ValueError('EXACT_FULL_TEST_SCOPE_REQUIRED')
    scope={'schema':SCOPE_SCHEMA,'base_commit':BASE_COMMIT,'source_commit':source_commit(),
        'source_sha256':source_fingerprint(),'checkpoint_sha256':FULL_SHA,'gate_sha256':GATE_SHA,
        'legacy_full_scope_sha256':sha256(full),'legacy_train_scope_sha256':sha256(train),
        'legacy_source_sha256':R2_SOURCE,'legacy_strict_loader_not_disabled':True,
        'config':c1_config().to_dict(),'config_sha256':canonical_sha(c1_config().to_dict()),
        'variant':'C1','raw_cache':str(raw_root),'r2_root':str(r2),'omega_sha256':OMEGA_SHA,
        'detector_lock':proof(raw_root/'DETECTOR_LOCK.json'),'mirror_receipt_sha256':sha256(mirror),
        'scenes':scenes,'timestamps':total_times,'candidates':total_candidates,'RGB_images':total_times*2,
        'GT_in_inference':False,'TEST_parameter_selection':False,'created_UTC':now(),
        'interpretation':'one authorized exploratory TEST; V4 design previously informed by V3 TEST error analysis'}
    write_json(scope_path,scope)
    authorization={'schema':APPROVAL_SCHEMA,'approved':True,
        'user_authorization_reference':'2026-10-26 human explicitly authorized one V4 C1 full22 TEST; attachment SHA '+sha256(a.instruction),
        'source_commit':source_commit(),'source_sha256':source_fingerprint(),
        'instruction_sha256':sha256(a.instruction),'scope_lock_sha256':sha256(scope_path),
        'operations':['preflight','replay','evaluate','diagnostics'],'splits':['test'],
        'method_ids':['FULL_WORLD_OBJECT_V3'],'one_exploratory_C1_full_TEST_only':True,
        'TEST_parameter_selection':False,'training_authorized':False,'new_model_search':False,
        'allowed_inference_variants':['C1'],'preflight_C0_scene1_only':True}
    write_json(auth_path,authorization)
    print(json.dumps({'registered':True,'scenes':22,'timestamps':total_times,'candidates':total_candidates,
        'scope_sha256':sha256(scope_path),'source_sha256':scope['source_sha256']}),flush=True)


def frozen_legacy(repo,directory):
    """Load the original package under a distinct module name, never edit it."""
    import torch
    directory.mkdir(parents=True,exist_ok=False)
    payload=subprocess.check_output(['git','archive','b448af4a7dfdf738aeab0c9dd5bc01e9a875beac','worldmot/worldmot'],cwd=repo)
    package=directory/'frozen_r2_package';package.mkdir()
    with tarfile.open(fileobj=io.BytesIO(payload)) as archive:
        for entry in archive.getmembers():
            if entry.isfile() and entry.name.endswith('.py'):
                p=package/Path(entry.name).relative_to('worldmot/worldmot');p.parent.mkdir(parents=True,exist_ok=True)
                p.write_bytes(archive.extractfile(entry).read())
    spec=importlib.util.spec_from_file_location('frozen_r2_package',package/'__init__.py',submodule_search_locations=[str(package)])
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    Core=importlib.import_module('frozen_r2_package.runtime.world_object_tracking_core').WorldObjectTrackingCore
    Config=importlib.import_module('frozen_r2_package.tracking.world_object_config').WorldObjectConfig
    inputs=importlib.import_module('frozen_r2_package.models.world_object_model_inputs').model_inputs
    return Core(Config(1024)),inputs,hashlib.sha256(payload).hexdigest()


def create_core(scope,variant,device):
    raw=ImmutableGMTDetectionCache(scope['raw_cache'])
    head,proof=load_frozen_r2_head(Path(scope['r2_root'])/'train/FULL_WORLD_OBJECT_V3/train_gate.json',raw.lock,device)
    if head.training or any(p.requires_grad for p in head.parameters()):raise AssertionError('FROZEN_EVAL_MODEL_REQUIRED')
    return raw,WorldObjectTrackingCore(state_configs(1024)[variant],head,device),proof


def run_scene(a,scope,scene,variant='C1',legacy=None,old_inputs=None,destination=None,limit=None):
    import torch
    job_started=time.perf_counter();started_UTC=now();source=source_fingerprint();execution_commit=source_commit()
    evidence=verify_scene_inputs(scope,scene)
    raw,core,checkpoint=create_core(scope,variant,a.device)
    cache=WorldObjectCache(Path(scope['r2_root'])/'cache_test_v2',scene,raw_cache=raw,require_features=True,require_geometry=True)
    destination=destination or a.output/f'predictions/V4_C1/scene_{scene:02d}'
    if destination.exists():
        if variant=='C1' and limit is None:return completed_scene(destination,scope,scene)
        raise FileExistsError(str(destination))
    stage=destination.with_name(destination.name+'.incomplete.'+str(os.getpid()));stage.mkdir(parents=True,exist_ok=False)
    core.reset(scene)
    if legacy is not None:legacy.reset(scene);legacy.model=core.model;legacy.device=a.device
    guard=install_no_gt_guard();observer=MotionObserver();totals=Counter();timing_totals=Counter()
    input_chain=hashlib.sha256();state_checks=Counter();exact=0;count=0;last=None;loop_start=time.perf_counter()
    try:
        with ExitStack() as stack,observer.installed():
            output=stack.enter_context((stage/'tracks.jsonl').open('x'))
            timing=stack.enter_context((stage/'timings.jsonl').open('x'))
            audits=stack.enter_context((stage/'input_audit.jsonl').open('x'))
            checks=stack.enter_context((stage/'state_checks.jsonl').open('x'))
            reference=stack.enter_context(Path(evidence['original_R2_prediction']).open())
            iterator=iter(cache.frames())
            for index in range(evidence['timestamps']):
                if limit is not None and index>=limit:break
                begin=time.perf_counter();frame=next(iterator);read_s=time.perf_counter()-begin
                timestamp=frame['timestamp']
                if timestamp!=index+1:raise ValueError('NONCAUSAL_OR_MISSING_TEST_TIMESTAMP')
                if (set(map(str,frame['record']['image_ids']))!={'1','2'}
                        or set(map(str,frame['record']['rgb_sha256']))!={'1','2'}):
                    raise ValueError('TWO_SYNCHRONOUS_ORIGINAL_RGB_INPUTS_REQUIRED')
                obs=replay_observations(frame);signature=observation_signature(obs);observer.begin()
                if legacy is not None:
                    from worldmot.models.world_object_model_inputs import model_inputs
                    new=model_inputs(core.bank.active(),obs,core.config,timestamp,a.device)
                    old=old_inputs(legacy.bank.active(),obs,legacy.config,timestamp,a.device)
                    if any(not torch.equal(x,y) for x,y in zip(new,old)):raise AssertionError('C0_R2_HEAD_INPUT_DIFFERENCE')
                t=time.perf_counter();result=core.step(obs,timestamp)
                if str(a.device).startswith('cuda'):torch.cuda.synchronize()
                core_s=time.perf_counter()-t
                # Archived IDs are read only after this timestamp's decision.
                reference_row=json.loads(next(reference))
                if reference_row['timestamp']!=timestamp:raise ValueError('R2_REFERENCE_TIMESTAMP_DIFFERENCE')
                totals.update(compare_candidates(reference_row,result))
                if observation_signature(obs)!=signature:raise AssertionError('BBOX_SCORE_OR_ROI_MUTATED')
                if legacy is not None:
                    legacy_result=legacy.step(obs,timestamp)
                    if result!=legacy_result:raise AssertionError('C0_FROZEN_R2_CORE_DIFFERENCE')
                    common_fields_equal(reference_row,result)
                    if set(core.bank.objects)!=set(legacy.bank.objects):raise AssertionError('C0_R2_OBJECT_SET_DIFFERENCE')
                    for gid,q in core.bank.objects.items():
                        for field in ('center','covariance','identity_embedding','support_points'):
                            if not np.array_equal(getattr(q,field),getattr(legacy.bank.objects[gid],field),equal_nan=True):
                                raise AssertionError('C0_R2_INTERNAL_STATE_DIFFERENCE '+field)
                    exact+=1
                numeric=validate_motion_bank(core.bank,timestamp) if variant=='C1' else {}
                if core.scene_id!=scene or core.bank.last_timestamp!=timestamp:raise AssertionError('SCENE_OR_BANK_RESET_INSIDE_VIDEO')
                for q in core.bank.objects.values():
                    state=q.spatial_state
                    if state is not None and state.last_trusted_geometry_timestamp==timestamp:
                        current=[o for o in obs if core.bank.last_assignments[o.key]==q.global_id]
                        if not any(o.geometry_valid and o.world_state_update_eligible for o in current):
                            raise AssertionError('INVALID_GEOMETRY_WROTE_MEASUREMENT')
                motion=observer.finish();state_checks.update(numeric)
                state_checks['evidence_status/'+result['bank']['evidence_status']]+=1
                result['variant']=variant
                serial_start=time.perf_counter();output.write(json.dumps(result,allow_nan=False)+'\n');output.flush()
                audit={'timestamp':timestamp,'candidate_count':len(obs),'unchanged_observation_sha256':signature,
                    'source':frame['source'],'image_ids':frame['record']['image_ids'],'rgb_sha256':frame['record']['rgb_sha256']}
                audits.write(json.dumps(audit,allow_nan=False)+'\n')
                checks.write(json.dumps({'timestamp':timestamp,**numeric,**motion,'bank_reset_count':1,'GT_in_inference':False},allow_nan=False)+'\n')
                input_chain.update(json.dumps(audit,sort_keys=True).encode())
                row={'timestamp':timestamp,'cache_read_and_join_s':read_s,'tracking_core_wall_s':core_s,
                    **core.last_timings,**{k:v for k,v in motion.items() if k.endswith('_s')},
                    'prediction_and_audit_output_s':time.perf_counter()-serial_start}
                row['complete_timestamp_wall_s']=time.perf_counter()-begin
                timing.write(json.dumps(row,allow_nan=False)+'\n');timing.flush()
                timing_totals.update({k:v for k,v in row.items() if k.endswith('_s')})
                count+=1;last=timestamp
                if count%100==0:print(json.dumps({'scene':scene,'variant':variant,'timestamps':count}),flush=True)
            if limit is None:
                if next(iterator,None) is not None or reference.readline():raise ValueError('EXTRA_TEST_FRAMES')
        if source_fingerprint()!=source:raise ValueError('INFERENCE_SOURCE_CHANGED_DURING_SCENE')
        if count!=(evidence['timestamps'] if limit is None else min(limit,evidence['timestamps'])):raise ValueError('INCOMPLETE_SCENE')
        if guard['blocked_GT_reads']:raise AssertionError('INFERENCE_TRIED_TO_READ_GT')
        files={name:sha256(stage/name) for name in ['tracks.jsonl','timings.jsonl','input_audit.jsonl','state_checks.jsonl']}
        manifest={'schema':'worldmot_v4_c1_prediction_manifest_v1','status':'COMPLETE' if limit is None else 'PROFILE_ONLY',
            'scene':scene,'variant':variant,'method_id':'FULL_WORLD_OBJECT_V3','first_timestamp':1,'last_timestamp':last,
            'timestamps':count,'expected_timestamps':evidence['timestamps'],'full_video':limit is None,
            'checkpoint_sha256':FULL_SHA,'frozen_head':checkpoint,'source_sha256':source,'source_commit':execution_commit,
            'base_commit':BASE_COMMIT,'tracking_config':core.config.to_dict(),'config_sha256':canonical_sha(core.config.to_dict()),
            'scope_lock_sha256':sha256(a.output/'scope_v4_c1_TEST.json'),
            'authorization_sha256':sha256(a.output/'authorization_v4_c1_TEST.json'),
            'cache_manifest_sha256':evidence['cache_manifest_sha256'],'GMT_raw_manifest_sha256':evidence['raw_manifest_sha256'],
            'candidate_count':totals['output_candidates'],'candidate_comparison':dict(totals),'ROI_mutations':0,
            'input_chain_sha256':input_chain.hexdigest(),'tracks_sha256':files['tracks.jsonl'],'output_sha256':files,
            'GT_in_inference':False,'GT_read_guard':guard,'causal':True,'bank_reset_count':1,'clip_boundary_resets':0,
            'C0_R2_exact_frames':exact,'motion_checks':dict(state_checks),'motion_call_counts':dict(observer.total),
            'execution_device':a.device,'head_precision':'float32','started_UTC':started_UTC,'completed_UTC':now(),
            'worker_wall_s':time.perf_counter()-job_started,'scene_loop_wall_s':time.perf_counter()-loop_start,
            'timing_sums':dict(timing_totals),'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            'GPU_peak_allocated_bytes':int(torch.cuda.max_memory_allocated()) if str(a.device).startswith('cuda') else None,
            'fps_definition':'CACHED_GEOMETRY_AND_DETECTOR_REPLAY; includes input validation and output IO; NOT RGB-to-ID',
            'cached_paired_times_per_s':count/max(timing_totals['complete_timestamp_wall_s'],1e-12)}
        write_json(stage/'manifest.json',manifest)
        if destination.exists():raise FileExistsError('NO_COMPLETED_SCENE_OVERWRITE')
        stage.rename(destination)
        for path in destination.iterdir():path.chmod(0o400)
        return manifest
    except BaseException as error:
        write_json(stage/'failure.json',{'status':'FAILED','scene':scene,'variant':variant,'completed_timestamps':count,
            'source_sha256':source,'error':repr(error),'traceback':traceback.format_exc(),'GT_in_inference':False})
        raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['register','preflight','scene','coverage','profile'],required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--r2-root',type=Path)
    p.add_argument('--instruction',type=Path);p.add_argument('--scene',type=int);p.add_argument('--device',default='cpu')
    a=p.parse_args();a.output=a.output.resolve()
    if a.stage=='register':register(a);return
    import torch
    torch.set_num_threads(1)
    scope_path=a.output/'scope_v4_c1_TEST.json';authorization=a.output/'authorization_v4_c1_TEST.json'
    operation='preflight' if a.stage in ('preflight','profile') else 'replay'
    sid=1 if a.stage in ('preflight','profile') else a.scene
    scope,approval=require_scope(scope_path,authorization,operation,sid)
    if a.stage=='preflight':
        report=a.output/'v4_c1_test_preflight.json'
        if report.exists():
            existing=json.loads(report.read_text())
            if existing['status']=='PASSED' and existing['source_sha256']==source_fingerprint():return
            raise RuntimeError('PRESERVE_FAILED_PREFLIGHT_DO_NOT_OVERWRITE')
        repo=Path(__file__).resolve().parents[2]
        legacy,old_inputs,archive_sha=frozen_legacy(repo,a.output/'preflight/frozen_reference')
        c0=run_scene(a,scope,1,'C0',legacy,old_inputs,a.output/'preflight/C0/scene_01')
        if c0['C0_R2_exact_frames']!=scope['scenes']['1']['timestamps']:raise AssertionError('FULL_SCENE1_C0_REQUIRED')
        c1=run_scene(a,scope,1)
        write_json(report,{'status':'PASSED','source_sha256':source_fingerprint(),'checkpoint_sha256':FULL_SHA,
            'config':scope['config'],'scope_lock_sha256':sha256(scope_path),'approval':approval,
            'C0_full_scene1_exact_frames':c0['C0_R2_exact_frames'],'R2_package_archive_sha256':archive_sha,
            'C0_manifest_sha256':sha256(a.output/'preflight/C0/scene_01/manifest.json'),
            'C1_scene1_manifest_sha256':sha256(a.output/'predictions/V4_C1/scene_01/manifest.json'),
            'C1_scene1_reused_as_official_output':True,'C1_motion_checks':c1['motion_checks'],
            'C1_idempotence':c1['motion_call_counts'],'GT_in_inference':False,'parameters_changed':False,
            'detection_bbox_score_roi_differences':0,'created_UTC':now()})
        print(json.dumps({'preflight':'PASSED','C0_exact':c0['C0_R2_exact_frames'],'C1_scene1_complete':c1['timestamps']}),flush=True)
    elif a.stage=='profile':
        pre=json.loads((a.output/'v4_c1_test_preflight.json').read_text())
        if pre['status']!='PASSED':raise RuntimeError('PREFLIGHT_REQUIRED')
        label=a.device.replace(':','_')
        result=run_scene(a,scope,1,destination=a.output/'hardware_profile'/label,limit=32)
        print(json.dumps({'profile_device':a.device,'cached_rate':result['cached_paired_times_per_s'],'worker_wall_s':result['worker_wall_s']}),flush=True)
    elif a.stage=='scene':
        pre=json.loads((a.output/'v4_c1_test_preflight.json').read_text())
        if pre['status']!='PASSED' or pre['source_sha256']!=source_fingerprint():raise RuntimeError('CURRENT_SOURCE_PREFLIGHT_REQUIRED')
        locks=a.output/'scene_locks';locks.mkdir(exist_ok=True)
        with (locks/f'scene_{sid:02d}.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            result=run_scene(a,scope,sid)
        print(json.dumps({'scene':sid,'status':result['status'],'timestamps':result['timestamps']}),flush=True)
    else:
        manifests=[completed_scene(a.output/f'predictions/V4_C1/scene_{sid:02d}',scope,sid) for sid in range(1,23)]
        if sum(m['timestamps'] for m in manifests)!=TEST_TIMES or sum(m['candidate_count'] for m in manifests)!=TEST_CANDIDATES:
            raise ValueError('FULL22_COVERAGE_MISMATCH')
        write_json(a.output/'v4_c1_fulltest_coverage.json',{'status':'COMPLETE','variant':'C1','scenes':manifests,
            'scene_count':22,'timestamps':TEST_TIMES,'RGB_images':2*TEST_TIMES,'candidate_count':TEST_CANDIDATES,
            'candidate_difference_count':0,'ROI_mutations':0,'scope_lock_sha256':sha256(scope_path),
            'source_sha256':source_fingerprint(),'checkpoint_sha256':FULL_SHA,'all_outputs_frozen_before_GT':True,
            'GT_in_inference':False,'causal':True,'created_UTC':now()})
        print(json.dumps({'coverage':'COMPLETE','scenes':22,'timestamps':TEST_TIMES,'candidates':TEST_CANDIDATES}),flush=True)


if __name__=='__main__':main()
