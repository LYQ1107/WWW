"""SHA-verified original-GMT FIT12 inputs; no GT parser or learned predictor.

Only the native storage adapters are reused from earlier experiments. Their
prediction modules, model weights, results and training logs are never loaded.
"""
from collections import Counter
import gzip
import json
from pathlib import Path
import numpy as np
import torch
from common import ROOT,sha256
from pirga_indexed_store import StoreReader,dense_matrix

FIT=(4,5,6,7,8,9,12,13,17,20,21,24)


def boxes_from_trace(path,expected_sha):
    if sha256(path)!=expected_sha:raise ValueError('Physical detector provenance changed')
    result={}
    with gzip.open(path,'rt') as stream:
        for line in stream:
            row=json.loads(line);key=tuple(row['source']);box=row['detection']['bbox']
            if key in result:raise ValueError('Duplicated physical observation')
            result[key]=[box[0],box[1],box[0]+box[2],box[1]+box[3]]
    return result


def indexed_calls(scene,coverage):
    spec=coverage['per_video'][str(scene)];store=StoreReader(spec['input_root'],spec['manifest_sha256'])
    root=Path(spec['input_root']).parent;capture=json.loads((root/'manifest.json').read_text())
    boxes=boxes_from_trace(root/'detector_provenance.jsonl.gz',capture['artifacts']['detector_provenance.jsonl.gz'])
    rows=list(store.db.execute('SELECT id,image,detection,frame,camera,variant FROM features ORDER BY id'))
    if [r[0] for r in rows]!=list(range(len(rows))):raise ValueError('Physical feature index discontinuity')
    frames=torch.tensor([r[3] for r in rows]);cameras=torch.tensor([r[4] for r in rows]);physical=torch.tensor([r[5] is None for r in rows])
    all_boxes=torch.tensor([boxes[(r[1],r[2])] for r in rows],dtype=torch.float32)
    # Admission and every immutable byte have been checked; avoid O(reference)
    # SQLite queries per call. The solver/math is independently replayed below.
    try:
        for (body,) in store.db.execute('SELECT body FROM calls ORDER BY call_id'):
            call=json.loads(body)
            if call['stage']!='run_global_tracker_plus' or not call['query_features']:continue
            q=call['query_features'];h=call['reference_features'];ids=torch.tensor(call['candidate_ids'],dtype=torch.long)
            yield dict(scene=scene,keys=call['query_keys'],frame=int(frames[q[0]]),camera=int(cameras[q[0]]),
                q=torch.from_numpy(np.array(store.vectors[q],dtype=np.float32,copy=True)),qboxes=all_boxes[q],
                h=torch.from_numpy(np.array(store.vectors[h],dtype=np.float32,copy=True)),hboxes=all_boxes[h],
                history_ids=torch.tensor(call['reference_tracks'],dtype=torch.long),history_frames=frames[h],history_cameras=cameras[h],
                physical=physical[h],candidates=ids,native_scores=torch.from_numpy(dense_matrix(call['matrix_shape'],call['matrix_entries'])),
                thresholds=torch.tensor(call['thresholds']),original_ids=call['accepted_ids'],precision='FP16 stored -> FP32; exact original FP32 scores')
    finally:store.db.close()


def shard_calls():
    from cv_pil_dev12_shards import NativeShardAssets
    pins=json.loads((ROOT/'results/cv_pil_v3_native_validation/P1_SOURCE_PINS.json').read_text())['release']
    if pins['manifest_sha']!='a4d2014edcc920f45753e0f76c378bf08f25c22758e3d8770572dc36632c2173':raise ValueError('TRAIN12 source pin changed')
    assets=NativeShardAssets(**pins)
    for call in assets.calls.values():
        if call['stage']!='run_global_tracker_plus' or not call['queries']:continue
        q=[assets.observations[tuple(k)] for k in call['queries']]
        h=[assets.observations[tuple(ref['source'])] for ref in call['references']]
        def boxes(observations):
            return torch.stack([o['native_box'] for o in observations]).reshape(-1,4) if observations else torch.empty(0,4)
        yield dict(scene=12,keys=[[12,*k] for k in call['queries']],frame=q[0]['frame'],camera=q[0]['camera'],
            q=torch.stack([assets.feature(spec) for spec in call['query_feature_keys']]),qboxes=boxes(q),
            h=torch.stack([assets.feature(spec) for spec in call['reference_feature_keys']]) if h else torch.empty(0,1152),hboxes=boxes(h),
            history_ids=torch.tensor([r['track'] for r in call['references']],dtype=torch.long),
            history_frames=torch.tensor([o['frame'] for o in h]),history_cameras=torch.tensor([o['camera'] for o in h]),
            physical=torch.tensor([spec['variant'] is None for spec in call['reference_feature_keys']],dtype=torch.bool),
            candidates=torch.tensor(call['track_ids'],dtype=torch.long),native_scores=call['track_scores'].clone(),
            thresholds=torch.tensor(call['thresholds']) if 'thresholds' in call else torch.tensor([.1*sum(r['track']==identity for r in call['references']) for identity in call['track_ids']]),
            original_ids=call['decision']['accepted_track_ids'],precision='sealed original FP32 vectors and scores')


def all_calls(coverage):
    for scene in FIT:
        yield from shard_calls() if scene==12 else indexed_calls(scene,coverage)
