#!/usr/bin/env python3
"""Immutable GT-free GMT pre-GTA candidates and independent ROI feature shards."""
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
from worldmot.models.gmt_detector_adapter import GMTDetectorAdapter
from worldmot.models.person_detector import sha256_file
import worldmot.models.gmt_detector_adapter as adapter_module

SCHEMA='gmt_pre_gta_rgb_person_roi_v1'


def save_json(path,data):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    os.replace(tmp,path)


def export_scene(detector,metadata,scene,output,dataset_root,split,signature,limit=None,shard_size=128):
    rows=[r for r in metadata['images'] if r['video_id']==scene]
    grouped=collections.defaultdict(dict)
    for r in rows:
        key=(r['frame_id'],r['view_id'])
        if r['view_id'] in grouped[r['frame_id']]:raise RuntimeError('Duplicate image key')
        grouped[r['frame_id']][r['view_id']]=r
    times=sorted(grouped)
    if any(set(g)!=set([1,2]) for g in grouped.values()):raise ValueError('Incomplete synchronized camera input')
    if limit:times=times[:limit]
    out=output/f'scene_{scene:02d}';out.mkdir(parents=True,exist_ok=True)
    manifest=out/'manifest.json'
    previous=json.loads(manifest.read_text()) if manifest.exists() else {}
    if previous:
        if previous['signature']!=signature:raise RuntimeError('Existing immutable candidate cache signature differs')
        if previous.get('status')=='COMPLETE':
            for sh in previous['shards']:
                if sha256_file(out/sh['file'])!=sh['sha256']:raise RuntimeError('Immutable shard changed')
            print(json.dumps({'scene':scene,'status':'EXISTING_VERIFIED_COMPLETE'}),flush=True)
            return
    start=time.perf_counter();saved=[];completed=[];observations=0;buffer=[]
    for p in sorted(out.glob('shard_*.json')):
        entry=json.loads(p.read_text())
        if entry['signature']!=signature or sha256_file(out/entry['file'])!=entry['sha256']:
            raise RuntimeError('Resume requires identical candidates and source')
        if entry['timestamps']!=times[len(completed):len(completed)+len(entry['timestamps'])]:
            raise RuntimeError('Cache resume is not a complete causal prefix')
        saved.append(entry);completed+=entry['timestamps'];observations+=entry['observations']
    save_json(manifest,{'status':'RUNNING','schema':SCHEMA,'signature':signature,'scene':scene,
                        'split':split,'expected_frames':len(times),'GT_in_cache':False})
    def flush():
        nonlocal observations
        if not buffer:return
        offsets=[0];cameras=[];indices=[];boxes=[];scores=[];features=[]
        for record,detections in buffer:
            for view,ds in sorted(detections.items()):
                for d in ds:
                    cameras.append(view);indices.append(d.detection_id);boxes.append(d.bbox_xywh)
                    scores.append(d.score);features.append(d.detector_roi_feature)
            offsets.append(len(boxes))
        path=out/f'shard_{len(saved):04d}.npz'
        if path.exists():raise FileExistsError('Uncommitted shard retained; inspect before recovery: '+str(path))
        tmp=path.with_suffix('.npz.tmp')
        dim=detector.manifest()['roi_feature_dim']
        with tmp.open('xb') as f:
            np.savez_compressed(f,frame_offsets=np.asarray(offsets,np.int64),
                timestamps=np.asarray([x[0]['timestamp'] for x in buffer],np.int32),
                camera_ids=np.asarray(cameras,np.int16),detection_ids=np.asarray(indices,np.int32),
                bbox=np.asarray(boxes,np.float32).reshape(-1,4),scores=np.asarray(scores,np.float32),
                roi_features=np.asarray(features,np.float32).reshape(-1,dim))
        os.replace(tmp,path)
        entry={'schema':SCHEMA,'signature':signature,'file':path.name,'sha256':sha256_file(path),
               'bytes':path.stat().st_size,'timestamps':[x[0]['timestamp'] for x in buffer],
               'observations':len(boxes),'records':[x[0] for x in buffer]}
        save_json(path.with_suffix('.json'),entry)
        observations+=len(boxes);completed.extend(entry['timestamps']);saved.append(entry);buffer.clear()
        save_json(out/'progress.json',{'scene':scene,'frames':len(completed),'expected':len(times),
            'observations':observations,'worker_wall_s':time.perf_counter()-start})
        print(json.dumps({'scene':scene,'frames':len(completed),'expected':len(times),'observations':observations}),flush=True)
    for t in times[len(completed):]:
        started=time.perf_counter()
        images={v:np.asarray(Image.open(dataset_root/split/r['file_name']).convert('RGB')) for v,r in grouped[t].items()}
        rgb_s=time.perf_counter()-started
        dets=detector.infer(images)
        record={'timestamp':t,'image_ids':{str(v):int(r['id']) for v,r in grouped[t].items()},
                'original_shapes':{str(v):list(im.shape) for v,im in images.items()},
                'rgb_sha256':{str(v):hashlib.sha256(np.ascontiguousarray(im).tobytes()).hexdigest() for v,im in images.items()},
                'rgb_read_s':rgb_s,'detector_timings':dict(detector.last_timings),
                'candidate_counts':{str(v):len(ds) for v,ds in dets.items()},
                'coordinate_transforms':dict(detector.last_diagnostics)}
        buffer.append((record,dets))
        if len(buffer)>=shard_size:flush()
    flush()
    save_json(manifest,{'status':'COMPLETE','schema':SCHEMA,'signature':signature,'scene':scene,'split':split,
        'frames':len(completed),'expected_frames':len(times),'full_video':not bool(limit),
        'observations':observations,'GT_in_cache':False,'track_ids_in_cache':False,'GTA_calls':0,
        'shards':[{k:s[k] for k in ['file','sha256','bytes','timestamps','observations']} for s in saved],
        'detector':detector.manifest(),'export_wall_s':time.perf_counter()-start,
        'timing_label':'detector export; not tracking-system E2E FPS'})


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--metadata',required=True,type=Path)
    p.add_argument('--dataset-root',type=Path,default=Path('/data/DATASETS/TRACKING/JDE/VisionTrack'))
    p.add_argument('--split',required=True,choices=['train','test'])
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--scenes',nargs='+',required=True,type=int)
    p.add_argument('--limit',type=int)
    p.add_argument('--shard-frames',type=int,default=128)
    args=p.parse_args()
    if args.split=='test' and args.limit:raise ValueError('TEST requires complete videos')
    meta=json.loads(args.metadata.read_text())
    if meta.get('annotations'):raise ValueError('GT identity is forbidden in raw candidate cache input')
    import torch
    torch.set_num_threads(2)
    detector=GMTDetectorAdapter()
    m=detector.manifest()
    signature={'schema':SCHEMA,'split':args.split,'metadata_sha256':sha256_file(args.metadata),
        'checkpoint_sha256':m['checkpoint_sha256'],'config_sha256':m['config_sha256'],
        'adapter_sha256':sha256_file(adapter_module.__file__),'exporter_sha256':sha256_file(__file__),
        'threshold':m['threshold'],'precision':m['precision'],'roi_feature_dim':m['roi_feature_dim']}
    args.output.mkdir(parents=True,exist_ok=True)
    lock=args.output/'DETECTOR_LOCK.json'
    try:
        with lock.open('x') as f:f.write(json.dumps(signature,indent=2)+'\n')
    except FileExistsError:
        # A small cross-process read may meet another writer at creation.
        for attempt in range(20):
            try:previous=json.loads(lock.read_text());break
            except json.JSONDecodeError:
                if attempt==19:raise
                time.sleep(.1)
        if previous!=signature:raise RuntimeError('Shared detector lock differs')
    for s in args.scenes:
        export_scene(detector,meta,s,args.output,args.dataset_root,args.split,signature,args.limit,args.shard_frames)


if __name__=='__main__':main()
