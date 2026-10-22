#!/usr/bin/env python3
"""Read-only full TRAIN identity demand before persistent-query fitting."""
import argparse
from collections import Counter
from pathlib import Path
import json
from worldmot.data.visiontrack import VisionTrack
from cache_fulltrain import atomic_json, sha


def audit(dataset_root, capacity):
    ds=VisionTrack(dataset_root,split='train');scenes=[];ambiguous_keys=0
    for sid in ds.scene_ids():
        frames=[]
        for pair in ds.synchronized_pairs(sid):
            identities=set()
            for annotations in ds.ground_truth_pair(pair).values():
                counts=Counter(int(a['instance_id']) for a in annotations)
                ambiguous_keys+=sum(n>1 for n in counts.values())
                identities.update(identity for identity,n in counts.items() if n==1)
            frames.append((pair.frame_id,identities))
        demand={}
        for length in [8,16,32]:
            clips=[(frames[start][0],len(set().union(*(x[1] for x in frames[start:start+length]))))
                   for start in range(0,len(frames),length)]
            first,count=max(clips,key=lambda row:row[1])
            demand[str(length)]={'maximum_legal_GT_identities':count,'worst_clip_first_timestamp':first}
        active=Counter();largest=0;largest_timestamp=None
        for index,(timestamp,identities) in enumerate(frames):
            active.update(identities)
            if index>=65:
                active.subtract(frames[index-65][1]);active=+active
            if len(active)>largest:largest=len(active);largest_timestamp=timestamp
        scenes.append({'scene':sid,'timestamps':len(frames),'clip_identity_demand':demand,
            'max_legal_GT_identities_in_65_timestamp_window':largest,
            'worst_window_last_timestamp':largest_timestamp})
    maximum=max(d['maximum_legal_GT_identities'] for s in scenes for d in s['clip_identity_demand'].values())
    report={'status':'GT_CLIP_CAPACITY_SUFFICIENT' if maximum<=capacity else 'CAPACITY_CHANGE_REQUIRED_BEFORE_TRAINING',
        'split':'train','scenes':scenes,'timestamps':sum(s['timestamps'] for s in scenes),
        'query_capacity':capacity,'maximum_clip_identity_demand':maximum,'ambiguous_image_identity_keys_masked':ambiguous_keys,
        'raw_annotation_sha256':sha(ds.annotation_path),'raw_GT_modified':False,'TEST_read':False,
        'limitation':'GT-only identity demand does not account for false-positive births or mistaken duplicate queries; those remain runtime audits.'}
    if report['timestamps']!=28754 or len(scenes)!=24:raise RuntimeError('not full 24-scene TRAIN')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset-root',default='/data/DATASETS/TRACKING/JDE/VisionTrack')
    p.add_argument('--capacity',type=int,default=128);p.add_argument('--output',required=True);a=p.parse_args()
    report=audit(a.dataset_root,a.capacity);atomic_json(a.output,report)
    print(json.dumps({k:v for k,v in report.items() if k!='scenes'},indent=2))
    if report['status']!='GT_CLIP_CAPACITY_SUFFICIENT':raise RuntimeError('insufficient registered Query capacity')
