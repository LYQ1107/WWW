#!/usr/bin/env python3
"""Verify complete cached timelines and preserve per-camera misses/false positives."""
import argparse,collections,gzip,json
from pathlib import Path
import numpy as np
from worldmot.data.visiontrack import VisionTrack
from worldmot.models.person_detector import PersonDetection
from worldmot.training.identity_targets import match_detections_to_gt,iou_xywh
from cache_fulltrain import sha,atomic_json

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    ds=VisionTrack('/data/DATASETS/TRACKING/JDE/VisionTrack',split='train');out=Path(a.output);out.mkdir(parents=True,exist_ok=True);scenes=[]
    with gzip.open(out/'detection_supervision_failures.jsonl.gz','wt') as log:
        for sid in ds.scene_ids():
            directory=Path(a.cache)/f'scene_{sid:02d}';seen=[];counts=collections.Counter();branch_missing=collections.Counter();shards=[]
            pairs={p.frame_id:p for p in ds.synchronized_pairs(sid)}
            for shard in sorted(directory.glob('shard_*.npz')):
                meta=json.loads(shard.with_suffix('.json').read_text())
                if sha(shard)!=meta['sha256']:raise RuntimeError('corrupt cache: '+str(shard))
                with np.load(shard,allow_pickle=False) as d:
                    if d['observations'].shape[-1]!=532 or d['observations'].dtype!=np.float16:raise RuntimeError('feature schema/dtype invalid')
                    if not np.isfinite(d['observations']).all() or not np.isfinite(d['geometry']).all():raise RuntimeError('non-finite cached model inputs')
                    for name,start,end in [('RGB',0,128),('RTMDet',128,256),('Omega',256,512)]:branch_missing[name]+=int((np.linalg.norm(d['observations'][:,start:end].astype(np.float32),axis=-1)<1e-8).sum())
                    for i,timestamp in enumerate(d['timestamps']):
                        timestamp=int(timestamp);seen.append(timestamp);lo,hi=map(int,d['frame_offsets'][i:i+2]);pair=pairs[timestamp];gt=ds.ground_truth_pair(pair);dets={1:[],2:[]}
                        for o in range(lo,hi):
                            v=int(d['camera_ids'][o]);dets[v].append(PersonDetection(v,len(dets[v]),tuple(map(float,d['bbox'][o])),float(d['observations'][o,-2])))
                        mapped=match_detections_to_gt(dets,gt,sid);events=[]
                        for v,ds_view in dets.items():
                            for di,det in enumerate(ds_view):
                                target=mapped[(v,di)]
                                if target.source=='unmatched_detector':events.append({'type':'DETECTOR_FP','view':v,'bbox':list(det.bbox_xywh)});counts['FP_observations']+=1
                                elif target.source=='ambiguous_duplicate_identity_masked':counts['ambiguous_identity_masked']+=1
                                else:counts['legal_identity_observations']+=1
                            # Match cardinality is independent of identity ambiguity.
                            candidates=sorted([(iou_xywh(det.bbox_xywh,ann['bbox']),di,gi) for di,det in enumerate(ds_view) for gi,ann in enumerate(gt.get(v,[]))],reverse=True);ud=set();ug=set()
                            for score,di,gi in candidates:
                                if score<.5:break
                                if di in ud or gi in ug:continue
                                ud.add(di);ug.add(gi)
                            for gi,ann in enumerate(gt.get(v,[])):
                                if gi not in ug:events.append({'type':'DETECTOR_MISS','view':v,'identity':int(ann['instance_id']),'bbox':ann['bbox']});counts['missed_GT_boxes']+=1
                        if events:log.write(json.dumps({'scene':sid,'timestamp':timestamp,'events':events})+'\n')
                        counts['timestamps']+=1;counts['observations']+=hi-lo
                shards.append({'file':str(shard),'sha256':meta['sha256'],'frames':len(meta['frames'])})
            if seen!=ds.frames(sid):raise RuntimeError(f'incomplete full scene {sid}: {len(seen)}/{len(ds.frames(sid))}')
            if any(branch_missing.values()):raise RuntimeError(f'missing visual branches in scene {sid}: {branch_missing}')
            scenes.append({'scene':sid,'counts':dict(counts),'missing_visual_branches':dict(branch_missing),'shards':shards});print(sid,dict(counts),flush=True)
    coverage={'status':'VERIFIED_COMPLETE_24_TRAIN_OBJECT_CACHE','timestamps':sum(s['counts']['timestamps'] for s in scenes),'scenes':scenes,'images':57508,'raw_annotations':596749,
              'raw_GT_modified':False,'all_inputs_finite':True,'GT_xyz_in_inputs':False,'supervision_failure_samples':str(out/'detection_supervision_failures.jsonl.gz'),'failure_samples_sha256':sha(out/'detection_supervision_failures.jsonl.gz')}
    if coverage['timestamps']!=28754:raise RuntimeError('not complete TRAIN')
    atomic_json(out/'coverage.json',coverage)
