#!/usr/bin/env python3
"""Causal RGB + shared GMT detections -> temporal/cross-view point Global IDs."""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
from PIL import Image
import torch
from worldmot.data.visiontrack import VisionTrack
from worldmot.models.gmt_detection_cache import ImmutableGMTDetectionCache
from worldmot.models.gmt_detector_adapter import GMTDetectorAdapter,GMT_PYTHON,detection_rows
from worldmot.models.person_detector import sha256_file
from worldmot.models.vggt_point_backend import VGGTPointBackend
from worldmot.tracking.point_identity_memory import PointIdentityMemory
from worldmot.tracking.point_to_instance_association import PointAssociationSettings
from cache_fulltrain import atomic_json
from experiment_provenance import source_stamp
from preserve_inference_attempts import prepare_scene


def stats(a):
    x=np.asarray(a,np.float64)
    return {'count':len(x),'total_s':float(x.sum()),'p50_s':float(np.median(x)) if len(x) else None,
            'p95_s':float(np.quantile(x,.95)) if len(x) else None}


def encode(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    raise TypeError(type(value).__name__)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--metadata',required=True)
    p.add_argument('--dataset-root',default='/data/DATASETS/TRACKING/JDE/VisionTrack')
    p.add_argument('--split',required=True,choices=['train','test'])
    p.add_argument('--raw-cache',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--method',choices=['temporal','joint'],default='joint')
    p.add_argument('--detector-mode',choices=['replay','live_verified'],default='live_verified')
    p.add_argument('--scenes',nargs='+',type=int)
    p.add_argument('--limit',type=int)
    p.add_argument('--lock')
    p.add_argument('--train-settings',help='TRAIN-only settings selected by the registered window scan')
    p.add_argument('--resource-mode',choices=['unspecified','exclusive_gpu','shared_gpu'],default='unspecified')
    p.add_argument('--points',type=int,choices=[8,16,32],default=16)
    p.add_argument('--point-log-stride',type=int,default=100)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--constant-position-cache',action='store_true')
    a=p.parse_args()
    torch.set_num_threads(2)
    if a.split=='test' and a.limit:raise ValueError('TEST must run complete videos')
    meta=json.loads(Path(a.metadata).read_text())
    if meta.get('annotations'):raise ValueError('Inference must use GT-free metadata')
    settings=PointAssociationSettings(points_per_person=a.points)
    if a.train_settings:
        if a.split!='train' or a.lock:raise ValueError('TRAIN settings cannot bypass the TEST freeze lock')
        selected=json.loads(Path(a.train_settings).read_text())
        if selected['TEST_used_for_selection']:raise ValueError('TEST selection forbidden')
        settings=PointAssociationSettings(**selected['settings'][a.method])
    if a.lock:
        lock=json.loads(Path(a.lock).read_text())
        if lock['status']!='FROZEN_FROM_TRAIN_VALIDATION' or lock['TEST_used_for_selection']:
            raise ValueError('Invalid point association freeze lock')
        settings=PointAssociationSettings(**lock['settings'][a.method])
        if lock['source_sha256']!=source_stamp()['sha256']:raise RuntimeError('Point source changed after freeze')
        if lock.get('runner_sha256')!=sha256_file(__file__):raise RuntimeError('Point runner changed after freeze')
        if a.split=='test' and lock.get('TEST_metadata_sha256')!=sha256_file(a.metadata):raise RuntimeError('TEST metadata differs from freeze')
        if lock.get('constant_position_cache',False)!=a.constant_position_cache:raise RuntimeError('Position encoding mode differs from TRAIN freeze')
    elif a.split=='test':raise ValueError('TEST requires TRAIN-only frozen settings')
    cache=ImmutableGMTDetectionCache(a.raw_cache)
    if cache.lock['split']!=a.split:raise ValueError('Raw detector cache split mismatch')
    if a.split=='test' and lock.get('TEST_detector_lock_sha256')!=sha256_file(cache.lock_path):raise RuntimeError('TEST candidates differ from freeze')
    ds=VisionTrack(a.dataset_root,annotation=a.metadata,split=a.split,load_annotations=False)
    detector=GMTDetectorAdapter(device=a.device,python=GMT_PYTHON) if a.detector_mode=='live_verified' else None
    backend=VGGTPointBackend(device=a.device,cache_position_encoding=a.constant_position_cache)
    memory=PointIdentityMemory(backend,settings,cross_view=a.method=='joint')
    root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
    signature={'split':a.split,'method':'vggt_point_'+a.method,'metadata_sha256':sha256_file(a.metadata),
        'source_sha256':source_stamp()['sha256'],'runner_sha256':sha256_file(__file__),
        'settings':settings.manifest(),'detector_cache_lock_sha256':sha256_file(cache.lock_path),
        'VGGT_checkpoint_sha256':backend.manifest()['checkpoint_sha256'],'detector_mode':a.detector_mode,
        'constant_position_cache':a.constant_position_cache,
        'resource_mode':a.resource_mode,'train_settings_sha256':sha256_file(a.train_settings) if a.train_settings else None,
        'lock_sha256':sha256_file(a.lock) if a.lock else None}
    try:
        for sid in a.scenes or ds.scene_ids():
            out=root/f'scene_{sid:02d}'
            if not prepare_scene(out,signature):continue
            memory.reset(sid);pairs=list(ds.synchronized_pairs(sid,limit=a.limit))
            atomic_json(out/'manifest.json',{'status':'RUNNING',**signature,'scene':sid,
                'GT_in_inference':False,'expected_frames':len(pairs)})
            durations=[];reads=[];stage_values=collections.defaultdict(list)
            seen_ids=set();candidate_count=0;output_count=0;detector_shards=set();wall=time.perf_counter()
            with (out/'tracks.jsonl').open('w') as f,(out/'timings.jsonl').open('w') as tf:
                for frame_index,pair in enumerate(pairs):
                    start=time.perf_counter()
                    images={v:np.asarray(Image.open(ds.image_path(r)).convert('RGB')) for v,r in pair.images.items()}
                    read_s=time.perf_counter()-start;image_ids={v:r.image_id for v,r in pair.images.items()}
                    begin=time.perf_counter();cached=cache.get(sid,pair.frame_id,images,image_ids)
                    replay_s=time.perf_counter()-begin
                    detector_shards.add(cache.last_diagnostics['raw_candidate_shard_sha256'])
                    detector_s=0.
                    if detector is not None:
                        begin=time.perf_counter();dets=detector.infer(images);detector_s=time.perf_counter()-begin
                        if json.dumps(detection_rows(dets),sort_keys=True)!=json.dumps(detection_rows(cached),sort_keys=True):
                            raise RuntimeError('Live detector differs from the shared immutable candidates')
                    else:dets=cached
                    row=memory.step(images,dets,pair.frame_id,sid)
                    calls=row['point_forward_calls']
                    model_s=sum(c['timings'].get('aggregator_s',0)+c['timings'].get('track_head_s',0) for c in calls)
                    stages={'rgb_read_s':read_s,'raw_cache_verify_s':replay_s,'GMT_detector_s':detector_s,
                        'VGGT_aggregator_s':sum(c['timings'].get('aggregator_s',0) for c in calls),
                        'VGGT_track_head_s':sum(c['timings'].get('track_head_s',0) for c in calls),
                        'point_preprocess_s':sum(c['timings'].get('preprocess_s',0) for c in calls),
                        'point_postprocess_s':sum(c['timings'].get('postprocess_s',0) for c in calls),
                        'point_memory_total_s':row['memory_s'],'VGGT_model_only_s':model_s}
                    # Save actual trajectories on a fixed schedule, plus every
                    # decision/evidence row. Sampling is independent of GT.
                    if frame_index>=8 and frame_index%a.point_log_stride:row.pop('point_predictions',None)
                    row['image_ids']={str(v):int(i) for v,i in image_ids.items()}
                    row['detections_by_camera']={str(v):[d.to_dict() for d in ds] for v,ds in dets.items()}
                    row['raw_candidate_shard_sha256']=cache.last_diagnostics['raw_candidate_shard_sha256']
                    row['runtime']={'frame_stages':stages,'timing_mode':a.detector_mode}
                    begin=time.perf_counter();f.write(json.dumps(row,default=encode,allow_nan=False)+'\n');f.flush()
                    stages['output_serialization_s']=time.perf_counter()-begin
                    total=time.perf_counter()-start;durations.append(total);reads.append(read_s)
                    for k,v in stages.items():stage_values[k].append(v)
                    tf.write(json.dumps({'timestamp':pair.frame_id,'rgb_to_delivered_ids_s':total,
                        'stages_s':stages,'queue_s':0.,'synchronous':True})+'\n');tf.flush()
                    candidate_count+=row['raw_candidate_count'];output_count+=row['output_count']
                    seen_ids.update(r['global_id'] for values in row['tracks_2d_by_camera'].values() for r in values)
                    if (frame_index+1)%16==0:
                        progress={'scene':sid,'frames':frame_index+1,'expected':len(pairs),
                            'raw_candidates':candidate_count,'outputs':output_count,'global_ids':len(seen_ids),
                            'timing_mode':a.detector_mode,'delivered_timestamps_per_s':len(durations)/sum(durations)}
                        atomic_json(out/'progress.json',progress);print(json.dumps(progress),flush=True)
            seconds=time.perf_counter()-wall
            report={'status':'COMPLETE',**signature,'scene':sid,'frames':len(pairs),'expected_frames':len(pairs),
                'full_video':len(pairs)==len(ds.frames(sid)),'GT_in_inference':False,
                'raw_candidates':candidate_count,'delivered_boxes':output_count,'global_ids':len(seen_ids),
                'detector':detector.manifest() if detector else cache.manifest(),'VGGT':backend.manifest(),
                'e2e':stats(durations),'stages':{k:stats(v) for k,v in stage_values.items()},
                'prediction_sha256':sha256_file(out/'tracks.jsonl'),'timings_sha256':sha256_file(out/'timings.jsonl'),
                'raw_candidate_shard_sha256':sorted(detector_shards),'elapsed_seconds':seconds,
                'real_e2e_seconds':seconds if detector else None,
                'real_e2e_fps':len(pairs)/seconds if detector else None,
                'association_replay_FPS':len(pairs)/seconds if not detector else None,
                'fps_unit':'synchronized dual-camera timestamps/second',
                'gpu':torch.cuda.get_device_name(a.device),'visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
                'timing_scope':'RGB read + shared candidate verification + live GMT detector including IPC if enabled + all anchored VGGT forwards + point sampling/assignment/memory + flushed final ID output; excludes model initialization and offline evaluation'}
            atomic_json(out/'manifest.json',report)
            print(json.dumps({k:report[k] for k in ['status','scene','frames','raw_candidates','delivered_boxes','real_e2e_fps','association_replay_FPS']}),flush=True)
    finally:
        if detector is not None:detector.close()


if __name__=='__main__':main()
