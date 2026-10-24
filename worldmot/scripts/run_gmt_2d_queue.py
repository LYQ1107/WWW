#!/usr/bin/env python3
"""Finite TRAIN-only 2D calibration, complete validation, frozen TEST controls."""
import argparse
from collections import Counter,defaultdict
import concurrent.futures
import itertools
import json
import os
from pathlib import Path
import subprocess
import time
from worldmot.data.visiontrack import VisionTrack
from worldmot.models.gmt_detection_cache import ImmutableGMTDetectionCache
from worldmot.models.person_detector import sha256_file
from cache_fulltrain import atomic_json
from experiment_provenance import source_stamp
from preserve_inference_attempts import prepare_scene
from gmt_2d_control import Control,Settings
from score_train_association_windows import score_rows,rank_record,OBJECTIVE

HERE=Path(__file__).resolve().parents[1]
NAS=Path('/data1/liuyeqiang/nas/A100/WorldMOT')
PYTHON='/data1/liuyeqiang/envs/worldmot311/bin/python'


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path)
    p.add_argument('--train-only',action='store_true',help='finish the TRAIN gate and freeze TEST settings, then stop before TEST inference')
    p.add_argument('--gpus',type=int,nargs='+',default=[0,2,6]);a=p.parse_args()
    assert len(set(a.gpus))==len(a.gpus) and set(a.gpus)<={0,2,6}
    root=a.root;root.mkdir(parents=True,exist_ok=True);(root/'logs').mkdir(exist_ok=True)
    status=root/'queue_status.json'
    if status.exists():
        old=json.loads(status.read_text());proc=Path('/proc')/str(old.get('pid',-1))
        if proc.exists() and b'scripts/run_gmt_2d_queue.py' in (proc/'cmdline').read_bytes().split(b'\0'):
            raise RuntimeError('2D queue already running')
    budget_path=HERE/'configs/gmt_point_train_budget_20261023.json';budget=json.loads(budget_path.read_text())
    provenance=source_stamp()
    watched=[Path(__file__),HERE/'scripts/gmt_2d_control.py',HERE/'scripts/run_gmt_2d_fullvideo.py',
        HERE/'scripts/score_train_association_windows.py',budget_path]
    script_hashes={str(x):sha256_file(x) for x in watched}
    def guard():
        if source_stamp()!=provenance or any(sha256_file(p)!=h for p,h in script_hashes.items()):
            raise RuntimeError('Registered 2D source/runner/budget changed')
    def state(stage,**fields):
        atomic_json(status,{'stage':stage,'pid':os.getpid(),'source_sha256':provenance['sha256'],
            'script_sha256':script_hashes,'TEST_used_for_selection':False,'gpus':a.gpus,'goal_complete':False,
            'updated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**fields})
        print(json.dumps({'stage':stage,**fields}),flush=True)
    def run(label,command,gpu=None):
        guard();env={**os.environ,'PYTHONPATH':str(HERE),'CUDA_VISIBLE_DEVICES':str(gpu) if gpu is not None else '',
            'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'2'}
        with (root/'logs'/f'{label}.log').open('a') as f:
            f.write('\nCOMMAND '+json.dumps(command)+'\n');f.flush()
            subprocess.run(command,cwd=HERE,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    raw_root=NAS/'gmt_detector_comparison_20261023';base=NAS/'fulltrain_20261022'
    raw=ImmutableGMTDetectionCache(raw_root/'raw_train_v1')
    ds=VisionTrack('/data/DATASETS/TRACKING/JDE/VisionTrack',annotation=base/'audit/train_metadata.json',split='train',load_annotations=False)
    gt=VisionTrack('/data/DATASETS/TRACKING/JDE/VisionTrack',split='train')
    configs={'iou_motion':[Settings(**budget['control_fixed'],minimum_iou=x) for x in budget['control_A0_minimum_iou']],
        'appearance_motion':[Settings(**budget['control_fixed'],appearance=True,minimum_iou=x,minimum_cosine=y)
            for x,y in itertools.product(budget['control_A1_minimum_iou'],budget['control_A1_minimum_cosine'])]}
    atomic_json(root/f'registration_{time.time_ns()}.json',{'source_provenance':provenance,'script_sha256':script_hashes,
        'budget':budget,'objective':OBJECTIVE,'settings':{k:[s.manifest() for s in v] for k,v in configs.items()},
        'TEST_used_for_selection':False,'raw_TRAIN_lock_sha256':sha256_file(raw.lock_path)})
    try:
        totals=defaultdict(Counter);records=[]
        state('FINITE_TRAIN_2D_WINDOW_SCAN')
        for sid in budget['validation_scenes']:
            pairs=list(ds.synchronized_pairs(sid));n=budget['window_length']
            for label,offset in [('start',0),('middle',(len(pairs)-n)//2),('end',len(pairs)-n)]:
                guard();data=[]
                for pair in pairs[offset:offset+n]:
                    image_ids={v:r.image_id for v,r in pair.images.items()}
                    data.append((pair,raw.get(sid,pair.frame_id,image_ids=image_ids),image_ids))
                for method,candidates in configs.items():
                    for index,settings in enumerate(candidates):
                        signature={'split':'train','method':method,'settings':settings.manifest(),
                            'source_sha256':provenance['sha256'],'script_sha256':script_hashes,'scene':sid,
                            'timestamps':[p.frame_id for p,_,_ in data],'raw_TRAIN_lock_sha256':sha256_file(raw.lock_path)}
                        folder=root/'windows'/method/f'config_{index:02d}'/f'scene_{sid:02d}'/label
                        if prepare_scene(folder,signature):
                            control=Control(settings);rows=[]
                            with (folder/'tracks.jsonl').open('w') as f:
                                for pair,dets,image_ids in data:
                                    row=control.step(dets,pair.frame_id,sid)
                                    row['image_ids']={str(v):i for v,i in image_ids.items()}
                                    row['detections_by_camera']={str(v):[d.to_dict() for d in values] for v,values in dets.items()}
                                    f.write(json.dumps(row)+'\n');rows.append(row)
                            counts=score_rows(rows,gt,sid)
                            atomic_json(folder/'manifest.json',{'status':'COMPLETE',**signature,'counts':counts,
                                'prediction_sha256':sha256_file(folder/'tracks.jsonl'),'GT_in_inference':False,
                                'GT_only_in_posthoc_score':True,'timing_label':'CACHED_ASSOCIATION_TRAIN_DIAGNOSTIC_NO_E2E'})
                        saved=json.loads((folder/'manifest.json').read_text())
                        assert sha256_file(folder/'tracks.jsonl')==saved['prediction_sha256']
                        totals[method,index].update(saved['counts'])
                        records.append({'method':method,'config':index,'scene':sid,'window':label,'counts':saved['counts'],
                            'manifest_sha256':sha256_file(folder/'manifest.json')})
                state('FINITE_TRAIN_2D_WINDOW_SCAN',scene=sid,window=label,completed_trials=len(records),expected_trials=216)
        selected={};ranking={}
        for method,candidates in configs.items():
            ranked=sorted(range(len(candidates)),key=lambda i:rank_record(totals[method,i],candidates[i].manifest()),reverse=True)
            selected[method]=candidates[ranked[0]].manifest()
            ranking[method]=[{'config':i,'settings':candidates[i].manifest(),'counts':dict(totals[method,i]),
                'selection_rank':list(rank_record(totals[method,i],candidates[i].manifest()))} for i in ranked]
        selection=root/'window_selection.json'
        atomic_json(selection,{'status':'SELECTED_ON_TRAIN_WINDOWS_PENDING_COMPLETE_VALIDATION','settings':selected,
            'objective':OBJECTIVE,'ranking':ranking,'records':records,'source_sha256':provenance['sha256'],'TEST_used_for_selection':False})
        # Complete real videos with the selected rules; TRAIN replay has no E2E claim.
        for method in configs:
            state('COMPLETE_SIX_SCENE_2D_TRAIN_VALIDATION',method=method,settings=selected[method])
            output=root/'train_validation'/method
            run(f'validate_{method}',[PYTHON,'-u','scripts/run_gmt_2d_fullvideo.py','--metadata',str(base/'audit/train_metadata.json'),
                '--raw-cache',str(raw_root/'raw_train_v1'),'--split','train','--method',method,'--settings-file',str(selection),
                '--scenes',*map(str,budget['validation_scenes']),'--device','cpu','--detector-mode','replay','--output',str(output)])
            for sid in budget['validation_scenes']:
                m=json.loads((output/f'scene_{sid:02d}/manifest.json').read_text())
                assert m['status']=='COMPLETE' and m['full_video'] and not m['GT_in_inference']
            run(f'evaluate_train_{method}',[PYTHON,'scripts/evaluate_fullvideo.py','--split','train',
                '--audit',str(base/'audit/data_audit.json'),'--predictions',str(output),'--scenes',*map(str,budget['validation_scenes']),
                '--output',str(root/'train_evaluation'/method)])
        state('VERIFY_FULL22_GT_FREE_TEST_CANDIDATES')
        integrity=root/'raw_TEST_integrity.json'
        if not integrity.exists():
            run('verify_raw_TEST',[PYTHON,'scripts/verify_gmt_raw_detection_cache.py','--cache',str(raw_root/'raw_test_v1'),
                '--metadata',str(base/'audit/test_metadata.json'),'--output',str(integrity)])
        audit=json.loads(integrity.read_text());assert audit['status']=='VERIFIED_COMPLETE_IMMUTABLE_RAW_GMT_SPLIT' and audit['split']=='test' and audit['timestamps']==29019
        lock=root/'frozen_test_lock.json'
        payload={'status':'FROZEN_FROM_COMPLETE_TRAIN_VALIDATION','TEST_used_for_selection':False,'settings':selected,
            'source_sha256':provenance['sha256'],'control_module_sha256':sha256_file(HERE/'scripts/gmt_2d_control.py'),
            'runner_sha256':sha256_file(HERE/'scripts/run_gmt_2d_fullvideo.py'),
            'selection_sha256':sha256_file(selection),'TEST_metadata_sha256':sha256_file(base/'audit/test_metadata.json'),
            'TEST_detector_lock_sha256':sha256_file(raw_root/'raw_test_v1/DETECTOR_LOCK.json'),
            'TRAIN_official_metrics_sha256':{m:sha256_file(root/'train_evaluation'/m/'metrics.json') for m in configs},
            'TEST_raw_integrity_sha256':sha256_file(integrity)}
        if lock.exists() and json.loads(lock.read_text())!=payload:raise RuntimeError('Do not overwrite a different 2D freeze')
        atomic_json(lock,payload);state('FROZEN_2D_FULL22_TEST_RUNNING',lock_sha256=sha256_file(lock),settings=selected)
        if a.train_only:
            state('TRAIN_GATE_COMPLETE_TEST_PENDING',lock_sha256=sha256_file(lock),settings=selected,
                  note='TRAIN-only gate requested; no TEST inference was started')
            return
        def worker(index):
            scenes=list(range(1,23))[index::len(a.gpus)]
            for method in configs:
                run(f'test_{method}_gpu{a.gpus[index]}',[PYTHON,'-u','scripts/run_gmt_2d_fullvideo.py',
                    '--metadata',str(base/'audit/test_metadata.json'),'--raw-cache',str(raw_root/'raw_test_v1'),
                    '--split','test','--method',method,'--lock',str(lock),'--resource-mode','shared_gpu',
                    '--scenes',*map(str,scenes),'--output',str(root/'test'/method)],a.gpus[index])
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(a.gpus)) as pool:
            for f in [pool.submit(worker,i) for i in range(len(a.gpus))]:f.result()
        for method in configs:
            paths=list((root/'test'/method).glob('scene_*/manifest.json'))
            manifests=[json.loads(p.read_text()) for p in paths]
            assert len(manifests)==22 and all(m['status']=='COMPLETE' and m['full_video'] for m in manifests)
            assert sum(m['frames'] for m in manifests)==29019
            run(f'evaluate_test_{method}',[PYTHON,'scripts/evaluate_fullvideo.py','--split','test',
                '--audit',str(base/'audit/data_audit.json'),'--predictions',str(root/'test'/method),
                '--output',str(root/'test_evaluation'/method)])
        state('COMPLETE_BOTH_2D_FORMAL_CONTROLS',timing_limitation='Accuracy runs share GPUs; exclusive speed audit still required')
    except BaseException as error:
        state('FAILED_REQUIRES_ENGINEERING_FIX',error=repr(error));raise


if __name__=='__main__':main()
