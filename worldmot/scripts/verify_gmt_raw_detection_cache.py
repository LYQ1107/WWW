#!/usr/bin/env python3
"""Deep full-split integrity, timestamp, camera and no-identity cache audit."""
from __future__ import annotations
import argparse
import collections
import json
from pathlib import Path
import numpy as np
from worldmot.models.person_detector import sha256_file
from worldmot.models.gmt_detector_adapter import GMT_CHECKPOINT_SHA256


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True,type=Path)
    p.add_argument('--metadata',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();lock=json.loads((a.cache/'DETECTOR_LOCK.json').read_text())
    meta=json.loads(a.metadata.read_text())
    if meta.get('annotations') or lock['metadata_sha256']!=sha256_file(a.metadata):
        raise RuntimeError('Expected exact GT-free metadata')
    if lock['checkpoint_sha256']!=GMT_CHECKPOINT_SHA256:raise RuntimeError('Wrong GMT weights')
    expected=collections.defaultdict(dict)
    for row in meta['images']:
        key=(row['video_id'],row['frame_id'])
        if row['view_id'] in expected[key]:raise ValueError('Duplicate metadata image')
        expected[key][row['view_id']]=row['id']
    allowed={'frame_offsets','timestamps','camera_ids','detection_ids','bbox','scores','roi_features'}
    report={'status':'RUNNING','metadata_sha256':sha256_file(a.metadata),
        'cache_lock_sha256':sha256_file(a.cache/'DETECTOR_LOCK.json'),'split':lock['split'],
        'GT_in_cache':False,'track_ids_in_cache':False,'GTA_calls':0,'scenes':[],
        'scope':'all stored candidates/features and metadata; exact live detector replay is a separate component gate'}
    for sid in sorted(v['id'] for v in meta['videos']):
        directory=a.cache/f'scene_{sid:02d}';manifest=json.loads((directory/'manifest.json').read_text())
        if (manifest['status']!='COMPLETE' or not manifest['full_video'] or manifest['signature']!=lock
            or manifest['GT_in_cache'] or manifest['track_ids_in_cache'] or manifest['GTA_calls']):
            raise RuntimeError('Incomplete or inconsistent raw GMT scene')
        seen=[];observations=0;artifacts=[];zero_features=0
        for shard in manifest['shards']:
            path=directory/shard['file']
            if sha256_file(path)!=shard['sha256']:raise RuntimeError('Raw candidate shard checksum mismatch')
            info=json.loads(path.with_suffix('.json').read_text())
            if info['signature']!=lock or info['timestamps']!=shard['timestamps'] or info['sha256']!=shard['sha256']:
                raise RuntimeError('Raw sidecar does not agree with the immutable lock')
            with np.load(path,allow_pickle=False) as archive:
                if set(archive.files)!=allowed:raise RuntimeError('Unexpected cache field; GT/identity arrays forbidden')
                data={k:archive[k] for k in archive.files}
            if (not np.isfinite(data['bbox']).all() or not np.isfinite(data['scores']).all()
                or not np.isfinite(data['roi_features']).all()):raise RuntimeError('Nonfinite raw detector values')
            if data['roi_features'].shape!=(len(data['scores']),lock['roi_feature_dim']):
                raise RuntimeError('Invalid raw ROI feature shape')
            if (data['bbox'][:,2:]<=0).any() or (data['scores']<=lock['threshold']).any():
                raise RuntimeError('Candidate filter/coordinates differ from native contract')
            zero_features+=int((np.linalg.norm(data['roi_features'],axis=1)==0).sum())
            if not np.array_equal(data['timestamps'],info['timestamps']):raise RuntimeError('Timestamp sidecar mismatch')
            if data['frame_offsets'][0]!=0 or data['frame_offsets'][-1]!=len(data['scores']):
                raise RuntimeError('Invalid frame offsets')
            for index,timestamp in enumerate(data['timestamps']):
                t=int(timestamp);seen.append(t);record=info['records'][index]
                want={str(v):int(i) for v,i in expected[sid,t].items()}
                if record['image_ids']!=want or set(want)!={'1','2'}:raise RuntimeError('Missing or wrong camera/image identity')
                lo,hi=map(int,data['frame_offsets'][index:index+2])
                for view in [1,2]:
                    mask=data['camera_ids'][lo:hi]==view
                    ids=data['detection_ids'][lo:hi][mask]
                    if not np.array_equal(ids,np.arange(len(ids))):raise RuntimeError('Detection indices are not stable per camera')
                    if len(ids)!=record['candidate_counts'][str(view)]:raise RuntimeError('Candidate count differs from stored frame')
                    if len(record['rgb_sha256'][str(view)])!=64:raise RuntimeError('Missing original RGB fingerprint')
                if record['coordinate_transforms'].get('forbidden_calls'):raise RuntimeError('Tracking entered the raw detector')
            observations+=len(data['scores'])
            artifacts.append({'path':str(path),'bytes':path.stat().st_size,'sha256':shard['sha256'],
                              'sidecar_sha256':sha256_file(path.with_suffix('.json'))})
        if seen!=sorted(t for s,t in expected if s==sid) or manifest['frames']!=len(seen):
            raise RuntimeError('Full video timestamp coverage differs from source metadata')
        report['scenes'].append({'scene':sid,'timestamps':len(seen),'images':len(seen)*2,
            'observations':observations,'zero_ROI_features':zero_features,
            'manifest_sha256':sha256_file(directory/'manifest.json'),'shards':artifacts})
        print(json.dumps({k:v for k,v in report['scenes'][-1].items() if k!='shards'}),flush=True)
    report.update(status='VERIFIED_COMPLETE_IMMUTABLE_RAW_GMT_SPLIT',
        timestamps=sum(s['timestamps'] for s in report['scenes']),
        images=sum(s['images'] for s in report['scenes']),
        observations=sum(s['observations'] for s in report['scenes']),
        total_shard_bytes=sum(sh['bytes'] for s in report['scenes'] for sh in s['shards']))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.output.exists():raise FileExistsError(a.output)
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
