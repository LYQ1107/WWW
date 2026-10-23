#!/usr/bin/env python3
"""Real paired-forward parity/speed audit for a constant sinusoidal grid.

No official source or pretrained weight is modified. This is a component
benchmark, never an MOT E2E-FPS result or a person-identity accuracy result.
"""
import argparse
import contextlib
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
from worldmot.models.gmt_detection_cache import ImmutableGMTDetectionCache
from worldmot.models.vggt_point_backend import VGGTPointBackend
from worldmot.models.person_detector import sha256_file
from worldmot.tracking.person_point_sampler import PersonPointSampler


def main():
    p=argparse.ArgumentParser();p.add_argument('--metadata',required=True,type=Path)
    p.add_argument('--cache',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--scene',default=8,type=int);p.add_argument('--repeats',default=3,type=int)
    a=p.parse_args()
    if a.scene not in [4,8,12,16,20,24]:raise ValueError('TRAIN validation only')
    import torch
    torch.set_num_threads(2)
    meta=json.loads(a.metadata.read_text())
    if meta.get('annotations'):raise ValueError('GT-free metadata required')
    records=[r for r in meta['images'] if r['video_id']==a.scene]
    stamps=sorted({r['frame_id'] for r in records})[:2]
    by_key={(r['view_id'],r['frame_id']):r for r in records if r['frame_id'] in stamps}
    images={k:np.asarray(Image.open(Path('/data/DATASETS/TRACKING/JDE/VisionTrack/train')/r['file_name']).convert('RGB')) for k,r in by_key.items()}
    cache=ImmutableGMTDetectionCache(a.cache);dets=cache.get(a.scene,stamps[0])
    sampler=PersonPointSampler(16)
    requests=[]
    for source,target in [(1,2),(2,1)]:
        keys=[(source,stamps[0]),(source,stamps[1]),(target,stamps[1])]
        queries=np.concatenate([sampler.sample(images[keys[0]],d,stamps[0]).xy for d in dets[source]])
        requests.append((keys,[images[k] for k in keys],queries))
    backend=VGGTPointBackend()
    import vggt.heads.track_modules.base_track_predictor as module
    original=module.get_2d_sincos_pos_embed
    constant_cache={};calls={'misses':0,'hits':0}
    def cached(embed_dim,grid_size,return_grid=False):
        grid=tuple(grid_size) if isinstance(grid_size,tuple) else (grid_size,grid_size)
        key=(embed_dim,grid,return_grid,'cuda:0')
        if key not in constant_cache:
            values=original(embed_dim,grid_size,return_grid=return_grid)
            constant_cache[key]=tuple(x.to('cuda:0') for x in values) if isinstance(values,tuple) else values.to('cuda:0')
            calls['misses']+=1
        else:calls['hits']+=1
        return constant_cache[key]
    @contextlib.contextmanager
    def mode(optimized):
        module.get_2d_sincos_pos_embed=cached if optimized else original
        try:yield
        finally:module.get_2d_sincos_pos_embed=original
    reference=None;results={};parity=[]
    a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.output.exists():raise FileExistsError(a.output)
    a.output.write_text(json.dumps({'status':'RUNNING','backend':backend.manifest()},indent=2))
    for optimized in [False,True]:
        samples=[]
        for repeat in range(a.repeats+1):
            outputs=[];begin=time.perf_counter()
            with mode(optimized):
                for keys,rgb,queries in requests:
                    outputs.append(backend.track(rgb,queries,image_keys=keys,reference_key=keys[0],current_timestamp=stamps[1]))
            elapsed=time.perf_counter()-begin
            if reference is None:reference=outputs
            for anchor,(expected,actual) in enumerate(zip(reference,outputs)):
                exact=all(np.array_equal(getattr(expected,k),getattr(actual,k)) for k in ['xy','visibility','confidence','valid'])
                difference=float(np.max(np.abs(expected.xy-actual.xy)))
                parity.append({'optimized':optimized,'repeat':repeat,'anchor':anchor,'exact':exact,'maximum_xy_difference':difference})
                if not exact:raise AssertionError('Constant encoding cache changed actual pretrained point outputs')
            sample={'repeat':repeat,'warm':repeat>0,'paired_point_pipeline_s':elapsed,
                'aggregator_s':sum(o.timings['aggregator_s'] for o in outputs),
                'track_head_s':sum(o.timings['track_head_s'] for o in outputs),
                'model_only_s':sum(o.timings['aggregator_s']+o.timings['track_head_s'] for o in outputs)}
            samples.append(sample);print(json.dumps({'optimized':optimized,**sample}),flush=True)
        warm=[s for s in samples if s['warm']]
        results['constant_cache' if optimized else 'original']={'samples':samples,
            'paired_model_median_s':float(np.median([s['model_only_s'] for s in warm])),
            'paired_point_pipeline_median_s':float(np.median([s['paired_point_pipeline_s'] for s in warm]))}
    final={'status':'PASS_REAL_PRETRAINED_PAIRED_FORWARD_EXACT_PARITY','backend':backend.manifest(),
        'benchmark_source_sha256':sha256_file(__file__),'official_track_utils_sha256':sha256_file(backend.source/'vggt/heads/track_modules/utils.py'),
        'metadata_sha256':sha256_file(a.metadata),'scene':a.scene,'timestamps':stamps,
        'queries_per_anchor':[len(q) for _,_,q in requests],'results':results,'parity':parity,
        'constant_cache_calls':calls,'constant_cache_bytes':sum(x.numel()*x.element_size() for x in constant_cache.values()),
        'model_only_speedup':results['original']['paired_model_median_s']/results['constant_cache']['paired_model_median_s'],
        'MOT_E2E_FPS':'NOT_MEASURED','identity_accuracy':'NOT_MEASURED',
        'scope':'Two actual anchored three-image forwards per synchronized timestamp; RGB predecoded, no detector; only fixed sinusoidal position tensor memoized, not images/depth/tracks.'}
    a.output.write_text(json.dumps(final,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:final[k] for k in ['status','model_only_speedup','constant_cache_calls','constant_cache_bytes','MOT_E2E_FPS']}),flush=True)


if __name__=='__main__':main()
