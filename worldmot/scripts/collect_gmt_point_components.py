#!/usr/bin/env python3
"""Collect verified component gates and existing official comparators."""
from __future__ import annotations
import argparse
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import statistics

NAS=Path('/data1/liuyeqiang/nas/A100/WorldMOT')


def artifact(path):
    return {'path':str(path.resolve()),'bytes':path.stat().st_size,
            'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def cache_progress(path):
    rows=[]
    for directory in sorted(path.glob('scene_*')):
        record={'scene':int(directory.name.split('_')[-1])}
        for filename in ['manifest.json','progress.json']:
            p=directory/filename
            if not p.exists():continue
            try:record[filename]=json.loads(p.read_text())
            except (FileNotFoundError,json.JSONDecodeError):record[filename]={'status':'TRANSIENT_SNAPSHOT_READ_UNAVAILABLE'}
        m=record.get('manifest.json',{});q=record.get('progress.json',{})
        rows.append({'scene':record['scene'],'status':m.get('status','RUNNING_WITHOUT_FINAL_MANIFEST'),
            'completed_timestamps':int(m.get('frames',m.get('timestamps',0))) if m.get('status')=='COMPLETE' else 0,
            'checkpointed_timestamps':int(q.get('done',q.get('frames',m.get('frames',m.get('timestamps',0)))))})
    return {'root':str(path),'completed_scenes':sum(r['status']=='COMPLETE' for r in rows),
            'completed_timestamps':sum(r['completed_timestamps'] for r in rows),
            'checkpointed_timestamps':sum(r['checkpointed_timestamps'] for r in rows),'scenes':rows}


def live_processes():
    allowed={Path('/data1/liuyeqiang')/name/'worldmot' for name in [
        'WWW_fulltrain','WWW_capacity','WWW_gmt_detector_20261023','WWW_vggt_points_20261023',
        'WWW_gmt_point_comparison_20261023']}
    scripts=['scripts/run_gmt_feature_queue.py','scripts/export_gmt_raw_detections.py',
             'scripts/run_gmt_training_queue.py',
             'scripts/run_gmt_v1_queue.py','scripts/run_gmt_2d_queue.py','scripts/run_gmt_2d_fullvideo.py',
             'scripts/calibrate_vggt_point_train.py','scripts/run_point_validation_queue.py',
             'scripts/benchmark_vggt_constant_position_cache.py',
             'scripts/run_vggt_point_fullvideo.py','scripts/run_fullvideo.py',
             'scripts/orchestrate_fulltrain_test.py','scripts/run_capacity_test.py']
    rows=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            if proc.stat().st_uid!=os.getuid() or (proc/'cwd').resolve() not in allowed:continue
            command=(proc/'cmdline').read_bytes().split(b'\0')
            if not any(s.encode() in command for s in scripts):continue
            status=(proc/'status').read_text()
            state=next(line.split(':',1)[1].strip() for line in status.splitlines() if line.startswith('State:'))
            env=(proc/'environ').read_bytes().split(b'\0')
            gpu=next((e.split(b'=',1)[1].decode() for e in env if e.startswith(b'CUDA_VISIBLE_DEVICES=')),None)
            rows.append({'pid':int(proc.name),'cwd':str((proc/'cwd').resolve()),'state':state,
                         'physical_GPU_visible_devices':gpu,'command':[s.decode(errors='replace') for s in command if s]})
        except (OSError,StopIteration):continue
    return sorted(rows,key=lambda x:x['pid'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'results')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    stamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
    gates={}
    files={
        'gmt_native_candidate_parity':NAS/'gmt_detector_comparison_20261023/raw_detector_preflight_v1.json',
        'gmt_immutable_replay':NAS/'gmt_detector_comparison_20261023/cache_replay_audit_v1.json',
        'vggt_official_track_forward':NAS/'vggt_point_association_20261023/backend_preflight_v2.json',
        'raw_GMT_complete_TRAIN_integrity':NAS/'gmt_detector_comparison_20261023/raw_train_integrity_v1.json',
        'bbox_only_visibility_evidence':NAS/'gmt_detector_comparison_20261023/train_v1_visibility_evidence_audit_v2.json',
        'bbox_mask_geometry_sampling_parity':NAS/'gmt_detector_comparison_20261023/geometry_sampling_parity_v2.json',
        'constant_position_model_parity':NAS/'vggt_point_association_20261023/constant_position_parity_benchmark_v1.json',
        'constant_position_real_E2E_parity':NAS/'vggt_point_association_20261023/position_cache_e2e_parity_v1.json'}
    for key,path in files.items():
        if path.exists():
            data=json.loads(path.read_text())
            gates[key]={'artifact':artifact(path),'result':data}
    report={'status':'VERIFIED_COMPONENTS_AND_TRAIN_DIAGNOSTICS_FULL_TEST_PENDING','generated_utc':stamp,'gates':gates,
            'new_fullvideo_identity_metrics':'NOT_AVAILABLE',
            'notes':['Independent detector parity and cache replay do not prove association quality.',
                     'VGGT forward smoke points are a grid; no person identity gain is claimed.',
                     'Original inference/evaluation continues; capacity TEST waiter paused.',
                     'User authorizes up to six concurrent V100 cards; actual occupancy admission required.'],
            'preserved_engineering_failure':{
                'path':str(NAS/'vggt_point_association_20261023/backend_preflight_v1.log'),
                'cause':'Official memory-optimized Aggregator returns None for unused layers; adapter initially cast every layer to float.',
                'fix':'Preserve unused None entries when casting the layers consumed by pretrained TrackHead.',
                'subsequent_real_forward':'v2 preflight; status must be checked above'}}
    (a.output/'gmt_vggt_component_preflight.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    baselines={}
    root=NAS/'fulltrain_20261022'
    methods=['gmt_native','v1','v1_observed_bbox','v2_joint_seed20261022','v2_joint_seed20261023','v2_joint_seed20261024',
             'v2_no_omega_seed20261022','v2_no_appearance_seed20261022','v2_two_stage_seed20261022','v2_no_geometry_seed20261022']
    for method in methods:
        path=root/'test_evaluation'/method/'metrics.json'
        if not path.exists():continue
        data=json.loads(path.read_text())
        if data['status']!='COMPLETE_VALID_SCOPE':continue
        expected_box_source='tracks_2d_observation_boxes_by_camera' if method=='v1_observed_bbox' else 'tracks_2d_by_camera'
        if data.get('raw_GT_modified') or data.get('custom_metrics_used') or data['box_source']!=expected_box_source:
            raise RuntimeError('Invalid original comparison protocol')
        prediction_method='v1' if method=='v1_observed_bbox' else method
        mans=[json.loads(m.read_text()) for m in sorted((root/'test'/prediction_method).glob('scene_*/manifest.json'))]
        if len(mans)!=22 or not all(m['status']=='COMPLETE' and m['full_video'] and not m['GT_in_inference'] for m in mans):
            raise RuntimeError('Cannot claim full original video coverage')
        assert sum(m['frames'] for m in mans)==29019
        baselines[method]={'source_metrics':artifact(path),'TrackEval':data['TrackEval'],
            'VisionTrack_MATLAB':data['VisionTrack_MATLAB'],'blocked':data['blocked'],
            'evaluated_scenes':data['evaluated_scenes'],'original_GT':True,
            'box_source':data['box_source'],
            'full_inference_scenes':22,'full_inference_timestamps':29019,
            'real_e2e_fps':29019/sum(m['real_e2e_seconds'] for m in mans),
            'timing_scope':mans[0]['timing_scope']}
    (a.output/'gmt_vggt_original_official_baselines.json').write_text(json.dumps(
        {'generated_utc':stamp,'status':'ORIGINAL_METHODS_AVAILABLE_NEW_METHODS_PENDING',
         'methods':baselines,'TEST_used_for_new_method_selection':False},indent=2,allow_nan=False)+'\n')
    processes=live_processes()
    cards=sorted({g for row in processes if not row['state'].startswith('T')
                  for g in (row['physical_GPU_visible_devices'] or '').split(',') if g})
    status={'generated_utc':stamp,'goal_complete':False,'live_processes':processes,
        'live_owned_GPU_cards':cards,'user_maximum_physical_cards':6,
        'raw_GMT_TRAIN':cache_progress(NAS/'gmt_detector_comparison_20261023/raw_train_v1'),
        'raw_GMT_TEST':cache_progress(NAS/'gmt_detector_comparison_20261023/raw_test_v1'),
        'GMT_Omega_TRAIN_objects':cache_progress(NAS/'gmt_detector_comparison_20261023/fulltrain_objects_v1'),
        'TRAIN_target_scenes':24,'TRAIN_target_timestamps':28754,
        'new_TEST_methods':'NOT_STARTED_PENDING_TRAIN_VALIDATION_FREEZE',
        'point_default_TRAIN_temporal':cache_progress(NAS/'vggt_point_association_20261023/train_temporal_default_fullscene8_v1'),
        'GMT_V1_TRAIN_validation':cache_progress(NAS/'gmt_detector_comparison_20261023/train_v1_validation_shared_v2'),
        'preserved_V1_failed_validation':cache_progress(NAS/'gmt_detector_comparison_20261023/train_v1_validation_shared_gpu0_v1'),
        'remaining':['Full detector/object cache integrity and supervision audit',
                     'Matched-budget GMTDet V2 joint/no-geometry fitting and TRAIN-only selection/freeze',
                     'A0/A1 2D controls and A2/A3 complete TRAIN validation and fixed selection',
                     'All new 22-video TEST streams and formal TrackEval/MATLAB evaluation',
                     'Final shared-candidate attribution, comparable exclusive-GPU timing and failure figures']}
    queued=NAS/'gmt_detector_comparison_20261023/association_training_status.json'
    queues={'association_training_queue':queued,
        'V1_fixed_rule_queue':NAS/'gmt_detector_comparison_20261023/v1_fixed_rule_queue_status.json',
        'point_calibration_gpu3':NAS/'vggt_point_association_20261023/calibration_v1/gpu3_status.json',
        'point_calibration_gpu7':NAS/'vggt_point_association_20261023/calibration_v1/gpu7_status.json',
        'point_fullvideo_queue':NAS/'vggt_point_association_20261023/point_fullvideo_queue_status.json',
        'two_D_control_queue':NAS/'final_comparison_20261023/2d_controls_v1/queue_status.json'}
    for key,path in queues.items():
        if path.exists():
            record=json.loads(path.read_text());record['verified_live_process']=any(p['pid']==record.get('pid') for p in processes)
            status[key]=record
    raw_verified=NAS/'gmt_detector_comparison_20261023/raw_train_integrity_v1.json'
    if raw_verified.exists():
        raw=json.loads(raw_verified.read_text())
        status['raw_TRAIN_integrity']={'artifact':artifact(raw_verified),
            **{k:raw[k] for k in ['status','timestamps','images','observations','total_shard_bytes']}}
    if len(cards)>6:raise RuntimeError('More than the six authorized physical cards are in use')
    (a.output/'gmt_vggt_live_status.json').write_text(json.dumps(status,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'gates':{k:v['result']['status'] for k,v in gates.items()},
                      'original_official_methods':list(baselines),'generated_utc':stamp}),flush=True)


if __name__=='__main__':main()
