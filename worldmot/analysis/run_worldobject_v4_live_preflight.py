#!/usr/bin/env python3
"""Finite real TRAIN RGB-to-ID timing; no GT and no inference cache substitute."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from worldobject_v4_audit_common import write_json,code_fingerprint,timing_summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--r2-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:0')
    p.add_argument('--variants',nargs='+',choices=['C0','C1','C2','C3','C4','C5'],default=['C0','C1','C2','C3','C4'])
    p.add_argument('--C5-quality-gate',type=Path);p.add_argument('--limit',type=int,default=61)
    a=p.parse_args()
    if not 1<=a.limit<=100:raise ValueError('FINITE_LIVE_PREFLIGHT_ONLY')
    if 'C5' in a.variants:
        if not a.C5_quality_gate:raise ValueError('C5_REQUIRES_PERSISTENT_STATE_TRAIN_GAIN_GATE')
        gate=json.loads(a.C5_quality_gate.read_text())
        if not gate.get('C5_live_quality_gate_passed'):raise ValueError('C5_STATE_QUALITY_GATE_NOT_PASSED')
    import torch
    from PIL import Image
    from worldmot.data.visiontrack import VisionTrack
    from worldmot.models.gmt_detection_cache import ImmutableGMTDetectionCache
    from worldmot.models.gmt_detector_adapter import GMTDetectorAdapter,GMT_PYTHON
    from worldmot.models.omega_geometry_backend import OmegaGeometryBackend
    from worldmot.models.world_object_v4_checkpoint import load_frozen_r2_head
    from worldmot.runtime.online_world_object_tracker import OnlineWorldObjectTracker
    from worldmot.runtime.world_object_tracking_core import WorldObjectTrackingCore
    from worldmot.tracking.world_object_v4_variants import state_configs
    torch.set_num_threads(1);repo=Path(__file__).resolve().parents[2];fp=code_fingerprint(repo)
    a.output.mkdir(parents=True,exist_ok=False);worker_started=time.perf_counter()
    raw=ImmutableGMTDetectionCache(a.r2_root.parent/'gmt_detector_comparison_20261023/raw_train_v1')
    head,proof=load_frozen_r2_head(a.r2_root/'train/FULL_WORLD_OBJECT_V3/train_gate.json',raw.lock,a.device)
    detector=GMTDetectorAdapter(device=a.device,python=GMT_PYTHON)
    try:
        manifest=detector.manifest()
        if any(manifest.get(k)!=raw.lock[k] for k in ('checkpoint_sha256','config_sha256','threshold','precision')):
            raise ValueError('LIVE_DETECTOR_CONTRACT_MISMATCH')
        geometry=OmegaGeometryBackend('/data1/liuyeqiang/WWW_omega_runtime_20261016/official_vggt_omega',
            '/data1/liuyeqiang/WWW_omega_runtime_20261016/weights/vggt_omega_1b_512.pt',device=a.device)
        if geometry.status!='READY':raise ValueError('VERIFIED_OMEGA512_UNAVAILABLE: '+geometry.reason)
        metadata=a.r2_root/'metadata_only_dataset'
        payload=json.loads((metadata/'annotations/train.json').read_text())
        if payload.get('annotations'):raise ValueError('PHYSICAL_METADATA_ONLY_LIVE_INPUT_REQUIRED')
        ds=VisionTrack(metadata,split='train',load_annotations=False);results={}
        for variant in a.variants:
            config=state_configs(raw.feature_dim)[variant]
            tracker=OnlineWorldObjectTracker(detector,geometry,config,
                core=WorldObjectTrackingCore(config,head,a.device),device=a.device)
            folder=a.output/variant;folder.mkdir();timing_rows=[];calls=0;scope=[]
            # Equal GPU/adapter warmup for every variant; discard ALL tracking
            # state before measured scene origins. These are not future inputs
            # to the measured tracker, and no GT is loaded.
            tracker.reset(4)
            for pair in ds.synchronized_pairs(4,limit=5):
                images={v:np.asarray(Image.open(ds.image_path(r)).convert('RGB')) for v,r in pair.images.items()}
                tracker.step(images,pair.frame_id)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats();variant_started=time.perf_counter()
            for sid in (4,8,12):
                tracker.reset(sid);path=folder/f'scene_{sid:02d}.jsonl';scene_count=0
                scene_calls=0;reactivations=0;aligned_calls=0;last_request=None;max_gap=0
                with path.open('x') as stream:
                    for pair in ds.synchronized_pairs(sid,limit=a.limit):
                        started=time.perf_counter();t=time.perf_counter()
                        images={v:np.asarray(Image.open(ds.image_path(r)).convert('RGB')) for v,r in pair.images.items()}
                        read_s=time.perf_counter()-t
                        row=tracker.step(images,pair.frame_id);torch.cuda.synchronize()
                        output_started=time.perf_counter();stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
                        output_s=time.perf_counter()-output_started
                        stages=dict(tracker.last_timings)
                        latency=time.perf_counter()-started
                        timing_rows.append({'RGB_read_s':read_s,'final_output_s':output_s,'RGB_to_final_ID_s':latency,
                            **stages})
                        requested=row.get('omega_evidence_request',{}).get('action','REQUEST_OMEGA')=='REQUEST_OMEGA'
                        calls+=int(requested);scene_calls+=int(requested)
                        if requested:
                            if last_request is not None:
                                gap=pair.frame_id-last_request;max_gap=max(max_gap,gap)
                                reactivations+=int(gap>1)
                            last_request=pair.frame_id
                            aligned_calls+=int(row['alignment']['success'])
                        scene_count+=1
                scope.append({'scene':sid,'paired_timestamps':scene_count,'full_video':False,
                    'Omega_calls':scene_calls,'Omega_skips':scene_count-scene_calls,
                    'nonconsecutive_reconstruction_reactivations':reactivations,
                    'aligned_Omega_calls':aligned_calls,'failed_alignment_calls':scene_calls-aligned_calls,
                    'maximum_reconstruction_gap_frame_ticks':max_gap})
            total=sum(r['RGB_to_final_ID_s'] for r in timing_rows)
            model_time=sum(r.get('omega_model_only_s',0.) for r in timing_rows)
            results[variant]={'scope':scope,'synchronized_dual_camera_times':len(timing_rows),'omega_calls':calls,
                'LIVE_RGB_to_final_ID_FPS':len(timing_rows)/total,
                'variant_loop_wall_FPS_excluding_shared_initialization':len(timing_rows)/(time.perf_counter()-variant_started),
                'Omega_model_only_per_executed_call_FPS':calls/model_time if model_time else None,
                'timings':timing_summary(timing_rows),'GPU_peak_allocated_bytes_parent':int(torch.cuda.max_memory_allocated()),
                'GPU_peak_scope':'parent Ω/head allocator only; GMT child process excluded, not total GPU memory',
                'initial_frames_included':True,'initialization_excluded_from_per_timestamp_latency':True,
                'equal_warmup_per_variant':{'TRAIN_scene':4,'timestamps':5,'tracking_state_discarded_before_measured_origins':True},
                'timing_scope':'real RGB decode + real GMT subprocess/IPC wait + conditional Ω + alignment/support + assignment/state + flushed final-ID serialization',
                'sparse_LIVE_reactivation_observed':any(x['nonconsecutive_reconstruction_reactivations'] for x in scope),
                'single_video_latency_not_six_GPU_total_throughput':True}
            write_json(folder/'timing_summary.json',results[variant])
            print(json.dumps({'live_variant_complete':variant,**results[variant]}),flush=True)
        if code_fingerprint(repo)!=fp:raise ValueError('SOURCE_CHANGED_DURING_LIVE_PREFLIGHT')
        write_json(a.output/'v4_live_runtime.json',{'status':'COMPLETE_FINITE_TRAIN_LIVE_PREFLIGHT','source_sha256':fp,
            'checkpoint':proof,'variants':results,'GT_used':False,'TEST_read':False,'model_initialization_plus_worker_wall_s':time.perf_counter()-worker_started,
            'device':torch.cuda.get_device_name(),'CUDA_VISIBLE_DEVICES':__import__('os').environ.get('CUDA_VISIBLE_DEVICES'),
            'detector':manifest,'geometry':geometry.diagnostics(),'full_video_LIVE_FPS_claimed':False})
    finally:detector.close()


if __name__=='__main__':main()
