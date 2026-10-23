#!/usr/bin/env python3
"""Verify real cached candidates/features reproduce the independent detector."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from worldmot.models.gmt_detection_cache import ImmutableGMTDetectionCache
from worldmot.models.gmt_detector_adapter import GMTDetectorAdapter,detection_rows


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cache',required=True,type=Path)
    p.add_argument('--metadata',required=True,type=Path)
    p.add_argument('--scene',default=8,type=int)
    p.add_argument('--frames',default=8,type=int)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--worker',action='store_true')
    args=p.parse_args()
    import torch
    torch.set_num_threads(2)
    from worldmot.models.gmt_detector_adapter import GMT_PYTHON
    detector=GMTDetectorAdapter(python=GMT_PYTHON if args.worker else None)
    cache=ImmutableGMTDetectionCache(args.cache)
    meta=json.loads(args.metadata.read_text())
    rows=[r for r in meta['images'] if r['video_id']==args.scene]
    times=sorted({r['frame_id'] for r in rows})[:args.frames]
    results=[]
    try:
        for t in times:
            current={r['view_id']:r for r in rows if r['frame_id']==t}
            ims={v:np.asarray(Image.open(Path('/data/DATASETS/TRACKING/JDE/VisionTrack')/'train'/r['file_name']).convert('RGB')) for v,r in current.items()}
            ids={v:r['id'] for v,r in current.items()}
            cached=cache.get(args.scene,t,ims,ids)
            live=detector.infer(ims)
            left=json.dumps(detection_rows(cached),sort_keys=True)
            right=json.dumps(detection_rows(live),sort_keys=True)
            assert left==right,'Candidates or frozen ROI descriptors changed on replay'
            results.append({'timestamp':t,'counts':{v:len(ds) for v,ds in live.items()},
                            'exact_candidate_and_feature_replay':True,
                            'shard_sha256':cache.last_diagnostics['raw_candidate_shard_sha256']})
        # A valid cache from a different time must fail an image-ID check.
        try:cache.get(args.scene,times[0],image_ids={1:-1,2:-2})
        except ValueError:pass
        else:raise AssertionError('Wrong frame/image IDs were accepted')
        report={'status':'PASS_REAL_IMMUTABLE_GMT_REPLAY','records':results,
                'cache_lock_sha256':hashlib.sha256(cache.lock_path.read_bytes()).hexdigest(),
                'detector':detector.manifest(),'interpreter_worker':args.worker,
                'GT_in_cache':False,'GTA_calls':0}
        args.output.parent.mkdir(parents=True,exist_ok=True)
        if args.output.exists():raise FileExistsError(args.output)
        args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(json.dumps({k:report[k] for k in ['status','GT_in_cache','GTA_calls','interpreter_worker']}),flush=True)
    finally:detector.close()


if __name__=='__main__':main()
