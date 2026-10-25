#!/usr/bin/env python3
"""Validate immutable delivered TEST boxes against the canonical GMT inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from verify_worldobject_v3_train_replays import current_candidates


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--method',required=True);parser.add_argument('--output',required=True,type=Path)
    a=parser.parse_args();root=a.root
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from worldmot.data.visiontrack import VisionTrack
    from worldmot.models.gmt_detection_cache import ImmutableGMTDetectionCache
    from worldmot.runtime.experiment_authorization import require_experiment_authorization
    authorization=root/('authorization_v4.json' if a.method=='GEOMETRY_ONLY_ABLATION_V3' else 'authorization_v3.json')
    approval=require_experiment_authorization(authorization,'diagnostics','test',a.method)
    metadata=root/'metadata_only_dataset/annotations/test.json'
    if json.loads(metadata.read_text()).get('annotations',[])!=[]:raise ValueError('ANNOTATION_FREE_METADATA_REQUIRED')
    ds=VisionTrack(root/'metadata_only_dataset',split='test',load_annotations=False)
    raw=ImmutableGMTDetectionCache(root.parent/'gmt_detector_comparison_20261023/raw_test_v1')
    if ds.scene_ids()!=list(range(1,23)):raise ValueError('FULL22_METADATA_REQUIRED')
    rows=[]
    for sid in ds.scene_ids():
        directory=root/'predictions'/a.method/f'scene_{sid:02d}'
        manifest=json.loads((directory/'manifest.json').read_text())
        if manifest['status']!='COMPLETE' or manifest['method_id']!=a.method or manifest['GT_in_inference'] is not False:
            raise ValueError('IMMUTABLE_COMPLETE_METHOD_REQUIRED')
        digest=hashlib.sha256();count=0;det_count=0;max_bbox=0.;max_score=0.
        with (directory/'tracks.jsonl').open('rb') as stream:
            for pair in ds.synchronized_pairs(sid):
                line=stream.readline();digest.update(line);row=json.loads(line)
                if row['timestamp']!=pair.frame_id or row['scene_id']!=sid or row['GT_in_inference'] is not False:
                    raise ValueError('OUTPUT_TIMESTAMP_OR_CONTEXT_MISMATCH')
                actual=current_candidates(row)
                detections=raw.get(sid,pair.frame_id,image_ids={v:r.image_id for v,r in pair.images.items()})
                wanted={(v,d.detection_id):d for v,items in detections.items() for d in items}
                if set(actual)!=set(wanted):raise ValueError('CURRENT_DETECTION_KEY_OR_RETAINED_OUTPUT_MISMATCH')
                for key,(box,score) in actual.items():
                    bbox_delta=float(np.max(np.abs(np.asarray(box)-np.asarray(wanted[key].bbox_xywh))))
                    score_delta=abs(score-wanted[key].score);max_bbox=max(max_bbox,bbox_delta);max_score=max(max_score,score_delta)
                    if bbox_delta>1e-5 or score_delta>1e-6:raise ValueError('DELIVERED_BBOX_OR_SCORE_CHANGED')
                count+=1;det_count+=len(actual)
            if stream.readline():raise ValueError('EXTRA_DELIVERED_TIMESTAMPS')
        if digest.hexdigest()!=manifest['tracks_sha256'] or count!=manifest['timestamps']:
            raise ValueError('OUTPUT_SHA_OR_COUNT_MISMATCH')
        rows.append({'scene':sid,'timestamps':count,'current_detections':det_count,
            'max_bbox_abs_difference':max_bbox,'max_score_abs_difference':max_score,'tracks_sha256':digest.hexdigest()})
        print(json.dumps({'TEST_candidates_verified_scene':sid,'method':a.method,'timestamps':count}),flush=True)
    if sum(x['timestamps'] for x in rows)!=29019:raise ValueError('FULL_TEST_TIMESTAMP_TOTAL_MISMATCH')
    result={'status':'ALL22_ACTUAL_CURRENT_TEST_CANDIDATES_VERIFIED','method':a.method,'approval':approval,
        'observed_at':time.time(),'scene_count':22,'timestamps':29019,'current_detections':sum(x['current_detections'] for x in rows),
        'scenes':rows,'GT_read':False,'original_GT_modified':False,'predictions_modified':False,
        'bbox_score_abs_tolerance':[1e-5,1e-6],'canonical_detector_lock_sha256':raw.manifest()['cache_lock_sha256'],
        'metadata_sha256':hashlib.sha256(metadata.read_bytes()).hexdigest(),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({'status':result['status'],'output':str(a.output)}),flush=True)


if __name__=='__main__':main()
