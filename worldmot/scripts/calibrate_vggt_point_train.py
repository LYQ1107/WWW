#!/usr/bin/env python3
"""Finite preregistered TRAIN point scan with exact-request model reuse."""
import argparse
from collections import Counter
import itertools
import json
import os
from pathlib import Path
import time
import numpy as np
from PIL import Image
import torch
from worldmot.data.visiontrack import VisionTrack
from worldmot.models.gmt_detection_cache import ImmutableGMTDetectionCache
from worldmot.models.person_detector import sha256_file
from worldmot.models.vggt_point_backend import VGGTPointBackend
from worldmot.tracking.point_identity_memory import PointIdentityMemory
from worldmot.tracking.point_to_instance_association import PointAssociationSettings
from cache_fulltrain import atomic_json
from experiment_provenance import source_stamp
from preserve_inference_attempts import prepare_scene
from point_request_cache import RequestCache
from score_train_association_windows import score_rows,OBJECTIVE

HERE=Path(__file__).resolve().parents[1]
BASE=Path('/data1/liuyeqiang/nas/A100/WorldMOT')


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--scenes',type=int,nargs='+',required=True);p.add_argument('--label',required=True)
    a=p.parse_args();torch.set_num_threads(2)
    budget_path=HERE/'configs/gmt_point_train_budget_20261023.json';budget=json.loads(budget_path.read_text())
    assert set(a.scenes)<=set(budget['validation_scenes']) and len(a.scenes)==len(set(a.scenes))
    root=a.root;root.mkdir(parents=True,exist_ok=True);provenance=source_stamp()
    driver_sha=sha256_file(__file__);budget_sha=sha256_file(budget_path)
    metadata=BASE/'fulltrain_20261022/audit/train_metadata.json'
    ds=VisionTrack('/data/DATASETS/TRACKING/JDE/VisionTrack',annotation=metadata,split='train',load_annotations=False)
    gt=VisionTrack('/data/DATASETS/TRACKING/JDE/VisionTrack',split='train')
    cache=ImmutableGMTDetectionCache(BASE/'gmt_detector_comparison_20261023/raw_train_v1')
    status=root/f'{a.label}_status.json'
    registration={'source_provenance':provenance,'driver_sha256':driver_sha,'budget_sha256':budget_sha,
        'metadata_sha256':sha256_file(metadata),'cache_lock_sha256':sha256_file(cache.lock_path),
        'scenes':a.scenes,'label':a.label,'TEST_used_for_selection':False,'objective':OBJECTIVE}
    atomic_json(root/f'{a.label}_registration_{time.time_ns()}.json',registration)
    def state(stage,**extra):
        atomic_json(status,{'stage':stage,'pid':os.getpid(),'source_sha256':provenance['sha256'],
            'driver_sha256':driver_sha,'budget_sha256':budget_sha,'scenes':a.scenes,
            'updated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            'visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'TEST_used_for_selection':False,
            'timing_label':'TRAIN_ASSOCIATION_REQUEST_REPLAY_NOT_E2E','goal_complete':False,**extra})
        print(json.dumps({'stage':stage,**extra}),flush=True)
    try:
        state('INITIALIZING_REAL_PRETRAINED_BACKEND')
        backend=VGGTPointBackend(cache_position_encoding=True)
        signature={'core_source_sha256':provenance['sha256'],'request_cache_driver_sha256':sha256_file(HERE/'scripts/point_request_cache.py'),
            'checkpoint_sha256':backend.manifest()['checkpoint_sha256'],'precision':['float16 aggregator','float32 TrackHead'],
            'constant_position_cache':True}
        request_cache=RequestCache(backend,root/'requests',signature)
        configs=list(itertools.product(budget['points_per_person'],budget['minimum_confidence'],budget['minimum_support']))
        results=[];total_windows=0
        for method in budget['point_methods']:
            for points,confidence,support in configs:
                settings=PointAssociationSettings(**budget['fixed_point_settings'],points_per_person=points,
                    minimum_confidence=confidence,minimum_support=support)
                name=f'k{points}_c{confidence:.1f}_s{support:.1f}'
                for sid in a.scenes:
                    pairs=list(ds.synchronized_pairs(sid));length=budget['window_length']
                    assert len(pairs)>=length
                    for label,offset in [('start',0),('middle',(len(pairs)-length)//2),('end',len(pairs)-length)]:
                        if source_stamp()!=provenance or sha256_file(__file__)!=driver_sha or sha256_file(budget_path)!=budget_sha:
                            raise RuntimeError('Registered source/driver/budget changed during calibration')
                        selected=pairs[offset:offset+length]
                        folder=root/'windows'/method/name/f'scene_{sid:02d}'/label
                        run_signature={'split':'train','method':method,'settings':settings.manifest(),
                            'source_sha256':provenance['sha256'],'driver_sha256':driver_sha,'budget_sha256':budget_sha,
                            'scene':sid,'window':label,'timestamps':[p.frame_id for p in selected],
                            'raw_cache_lock_sha256':registration['cache_lock_sha256'],'metadata_sha256':registration['metadata_sha256']}
                        if prepare_scene(folder,run_signature):
                            atomic_json(folder/'manifest.json',{'status':'RUNNING',**run_signature})
                            memory=PointIdentityMemory(request_cache,settings,cross_view=method=='joint');memory.reset(sid)
                            rows=[]
                            with (folder/'tracks.jsonl').open('w') as stream:
                                for pair in selected:
                                    images={v:np.asarray(Image.open(ds.image_path(r)).convert('RGB')) for v,r in pair.images.items()}
                                    image_ids={v:r.image_id for v,r in pair.images.items()}
                                    detections=cache.get(sid,pair.frame_id,images,image_ids)
                                    row=memory.step(images,detections,pair.frame_id,sid)
                                    row['image_ids']={str(v):i for v,i in image_ids.items()}
                                    row['detections_by_camera']={str(v):[d.to_dict() for d in values] for v,values in detections.items()}
                                    # Complete actual trajectories remain in the model-request cache.
                                    row.pop('point_predictions',None)
                                    stream.write(json.dumps(row,default=lambda x:x.tolist() if hasattr(x,'tolist') else x)+'\n')
                                    rows.append(row)
                            counts=score_rows(rows,gt,sid)
                            atomic_json(folder/'manifest.json',{'status':'COMPLETE',**run_signature,'counts':counts,
                                'GT_in_inference':False,'GT_only_in_posthoc_score':True,'objective':OBJECTIVE,
                                'prediction_sha256':sha256_file(folder/'tracks.jsonl'),'request_cache':request_cache.manifest(),
                                'real_e2e_fps':None,'association_replay_FPS':'NOT_REPORTED_FOR_CALIBRATION'})
                        saved=json.loads((folder/'manifest.json').read_text())
                        if sha256_file(folder/'tracks.jsonl')!=saved['prediction_sha256']:raise RuntimeError('Saved TRAIN window changed')
                        results.append({'method':method,'config':name,'scene':sid,'window':label,
                            'settings':settings.manifest(),'counts':saved['counts'],'manifest':str(folder/'manifest.json'),
                            'manifest_sha256':sha256_file(folder/'manifest.json')})
                        total_windows+=1
                        state('FINITE_TRAIN_WINDOW_SCAN',method=method,config=name,scene=sid,window=label,
                            completed_windows=total_windows,expected_windows=2*18*len(a.scenes)*3,
                            request_cache=request_cache.manifest())
        result=root/f'{a.label}_results.json'
        atomic_json(result,{'status':'COMPLETE_TRAIN_CALIBRATION_PARTITION','registration':registration,'results':results,
            'backend':backend.manifest(),'request_cache':request_cache.manifest(),'TEST_used_for_selection':False})
        state('COMPLETE_TRAIN_CALIBRATION_PARTITION',completed_windows=total_windows,results_sha256=sha256_file(result),
              request_cache=request_cache.manifest())
    except BaseException as error:
        state('FAILED_REQUIRES_ENGINEERING_FIX',error=repr(error));raise


if __name__=='__main__':main()
