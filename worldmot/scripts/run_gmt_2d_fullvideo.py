#!/usr/bin/env python3
"""Shared pre-GTA current detections -> independent causal 2D controls."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import time
import numpy as np
from PIL import Image
import torch
from worldmot.data.visiontrack import VisionTrack
from worldmot.models.person_detector import sha256_file
from worldmot.models.person_detector_factory import make_person_detector
from cache_fulltrain import atomic_json
from experiment_provenance import source_stamp
from preserve_inference_attempts import prepare_scene
from gmt_2d_control import Control,Settings


def stats(values):
    x=np.asarray(values,float)
    return {'count':len(x),'total_s':float(x.sum()),'p50_s':float(np.median(x)) if len(x) else None,
        'p95_s':float(np.quantile(x,.95)) if len(x) else None}


def main():
    p=argparse.ArgumentParser();p.add_argument('--metadata',required=True);p.add_argument('--raw-cache',required=True)
    p.add_argument('--output',required=True,type=Path);p.add_argument('--split',choices=['train','test'],required=True)
    p.add_argument('--method',choices=['iou_motion','appearance_motion'],required=True)
    p.add_argument('--lock');p.add_argument('--settings-file');p.add_argument('--scenes',type=int,nargs='+')
    p.add_argument('--device',default='cuda:0');p.add_argument('--limit',type=int)
    p.add_argument('--detector-mode',choices=['live_verified','replay'],default='live_verified')
    p.add_argument('--resource-mode',choices=['shared_gpu','exclusive_gpu','unspecified'],default='unspecified');a=p.parse_args()
    torch.set_num_threads(2)
    if a.split=='test' and a.limit:raise ValueError('Full TEST videos required')
    module=Path(__file__).with_name('gmt_2d_control.py');provenance=source_stamp()
    settings=Settings(appearance=a.method=='appearance_motion')
    if a.settings_file:
        if a.split!='train':raise ValueError('TEST settings come only from the TRAIN freeze lock')
        settings=Settings(**json.loads(Path(a.settings_file).read_text())['settings'][a.method])
    if a.lock:
        lock=json.loads(Path(a.lock).read_text())
        assert lock['status']=='FROZEN_FROM_COMPLETE_TRAIN_VALIDATION' and not lock['TEST_used_for_selection']
        assert lock['source_sha256']==provenance['sha256'] and lock['control_module_sha256']==sha256_file(module)
        assert lock['runner_sha256']==sha256_file(__file__)
        if a.split=='test':assert lock['TEST_metadata_sha256']==sha256_file(a.metadata)
        settings=Settings(**lock['settings'][a.method])
    elif a.split=='test':raise ValueError('TEST requires TRAIN-only freeze')
    assert settings.appearance==(a.method=='appearance_motion')
    metadata=json.loads(Path(a.metadata).read_text());assert not metadata.get('annotations')
    ds=VisionTrack('/data/DATASETS/TRACKING/JDE/VisionTrack',annotation=a.metadata,split=a.split,load_annotations=False)
    detector=make_person_detector('gmt_pre_gta',a.device,None,None,raw_cache=a.raw_cache,mode=a.detector_mode)
    control=Control(settings);a.output.mkdir(parents=True,exist_ok=True)
    signature={'split':a.split,'method':a.method,'source_sha256':provenance['sha256'],
        'runner_sha256':sha256_file(__file__),'control_module_sha256':sha256_file(module),
        'settings':settings.manifest(),'metadata_sha256':sha256_file(a.metadata),
        'raw_cache_lock_sha256':sha256_file(Path(a.raw_cache)/'DETECTOR_LOCK.json'),
        'detector_mode':a.detector_mode,'resource_mode':a.resource_mode,'GT_in_inference':False}
    if a.split=='test':assert signature['raw_cache_lock_sha256']==lock['TEST_detector_lock_sha256']
    try:
        for sid in a.scenes or ds.scene_ids():
            folder=a.output/f'scene_{sid:02d}'
            if not prepare_scene(folder,signature):continue
            pairs=list(ds.synchronized_pairs(sid,limit=a.limit));control.reset(sid)
            atomic_json(folder/'manifest.json',{'status':'RUNNING',**signature,'scene':sid,'expected_frames':len(pairs)})
            stages=defaultdict(list);durations=[];ids=set();boxes=0;start=time.perf_counter()
            with (folder/'tracks.jsonl').open('w') as f,(folder/'timings.jsonl').open('w') as tf:
                for index,pair in enumerate(pairs):
                    begin=time.perf_counter()
                    images={v:np.asarray(Image.open(ds.image_path(r)).convert('RGB')) for v,r in pair.images.items()}
                    read=time.perf_counter()-begin
                    detector.set_context(sid,pair.frame_id,{v:r.image_id for v,r in pair.images.items()})
                    t=time.perf_counter();dets=detector.infer(images);detection_s=time.perf_counter()-t
                    t=time.perf_counter();row=control.step(dets,pair.frame_id,sid);association_s=time.perf_counter()-t
                    row['image_ids']={str(v):r.image_id for v,r in pair.images.items()}
                    row['detections_by_camera']={str(v):[d.to_dict() for d in values] for v,values in dets.items()}
                    t=time.perf_counter();f.write(json.dumps(row)+'\n');f.flush();output_s=time.perf_counter()-t
                    total=time.perf_counter()-begin;durations.append(total)
                    frame_stages={'rgb_read_s':read,'detector_and_candidate_verification_s':detection_s,
                        '2D_association_s':association_s,'output_serialization_s':output_s}
                    for k,v in frame_stages.items():stages[k].append(v)
                    tf.write(json.dumps({'timestamp':pair.frame_id,'rgb_to_delivered_ids_s':total,'stages_s':frame_stages})+'\n');tf.flush()
                    boxes+=row['output_count'];ids.update(o['global_id'] for values in row['tracks_2d_by_camera'].values() for o in values)
                    if (index+1)%128==0:
                        atomic_json(folder/'progress.json',{'frames':index+1,'expected':len(pairs),'boxes':boxes,'global_ids':len(ids)})
            seconds=time.perf_counter()-start
            report={'status':'COMPLETE',**signature,'scene':sid,'frames':len(pairs),'expected_frames':len(pairs),
                'full_video':len(pairs)==len(ds.frames(sid)),'delivered_boxes':boxes,'global_ids':len(ids),
                'e2e':stats(durations),'stages':{k:stats(v) for k,v in stages.items()},
                'real_e2e_fps':len(pairs)/seconds if a.detector_mode=='live_verified' else None,
                'real_e2e_seconds':seconds if a.detector_mode=='live_verified' else None,
                'association_replay_FPS':len(pairs)/seconds if a.detector_mode=='replay' else None,
                'prediction_sha256':sha256_file(folder/'tracks.jsonl'),'detector':detector.manifest(),
                'visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'trainable_association_parameters':0,
                'fps_unit':'synchronized dual-camera timestamps/second',
                'timing_scope':'RGB decode + real GMT detector including IPC and exact candidate checks + 2D association + flushed final IDs; initialization and offline evaluation excluded'}
            atomic_json(folder/'manifest.json',report)
            print(json.dumps({k:report[k] for k in ['status','scene','frames','delivered_boxes','real_e2e_fps','association_replay_FPS']}),flush=True)
    finally:
        if hasattr(detector,'close'):detector.close()


if __name__=='__main__':main()
