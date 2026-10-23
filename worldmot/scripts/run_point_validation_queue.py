#!/usr/bin/env python3
"""Wait for finite TRAIN scan, validate selected settings, freeze and evaluate TEST."""
import argparse
from collections import Counter,defaultdict
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import time
from worldmot.models.person_detector import sha256_file
from cache_fulltrain import atomic_json
from experiment_provenance import source_stamp
from score_train_association_windows import rank_record,OBJECTIVE

HERE=Path(__file__).resolve().parents[1]
NAS=Path('/data1/liuyeqiang/nas/A100/WorldMOT')
PYTHON='/data1/liuyeqiang/envs/worldmot311/bin/python'


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--calibration-root',type=Path,default=None)
    p.add_argument('--gpus',type=int,nargs='+',default=[3,7])
    p.add_argument('--test-gpus',type=int,nargs='+',default=[0,2,3,4,6,7]);a=p.parse_args()
    assert set(a.gpus)=={3,7} and len(set(a.test_gpus))==len(a.test_gpus) and set(a.test_gpus)<={0,2,3,4,6,7}
    root=a.root;root.mkdir(parents=True,exist_ok=True);logs=root/'logs';logs.mkdir(exist_ok=True)
    calibration=a.calibration_root or (root/'calibration_v1');provenance=source_stamp()
    watched=[Path(__file__),HERE/'scripts/run_vggt_point_fullvideo.py',HERE/'scripts/score_train_association_windows.py',
        HERE/'scripts/calibrate_vggt_point_train.py',HERE/'scripts/point_request_cache.py',
        HERE/'scripts/ram_point_request_cache.py',HERE/'configs/gmt_point_train_budget_20261023.json']
    registered={str(p):sha256_file(p) for p in watched};budget=json.loads(watched[-1].read_text())
    status=root/'point_fullvideo_queue_status.json'
    if status.exists():
        old=json.loads(status.read_text());proc=Path('/proc')/str(old.get('pid',-1))
        if proc.exists() and b'scripts/run_point_validation_queue.py' in (proc/'cmdline').read_bytes().split(b'\0'):
            raise RuntimeError('Verified point full-video queue already running')
    def guard():
        if source_stamp()!=provenance or any(sha256_file(p)!=h for p,h in registered.items()):
            raise RuntimeError('Registered point source/runner/scan/budget changed')
    def state(stage,**extra):
        atomic_json(status,{'stage':stage,'pid':os.getpid(),'source_sha256':provenance['sha256'],
            'registered_script_sha256':registered,'gpus':a.gpus,'test_gpus':a.test_gpus,
            'TEST_used_for_selection':False,'goal_complete':False,
            'calibration_root':str(calibration),
            'updated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**extra})
        print(json.dumps({'stage':stage,**extra}),flush=True)
    def run(label,command,gpu=None):
        guard()
        # Admit extra point jobs only when they fit alongside existing jobs.
        # GPU cards remain within the six already authorized physical IDs.
        while gpu is not None:
            free=int(subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
            available=next(int(x.split()[1])//1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))
            if free>=16000 and available>=16000:break
            print(json.dumps({'stage':'WAITING_MEMORY_ADMISSION','gpu':gpu,'free_GPU_MB':free,'available_RAM_MB':available}),flush=True)
            time.sleep(30);guard()
        env={**os.environ,'PYTHONPATH':str(HERE),'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'2',
             'CUDA_VISIBLE_DEVICES':str(gpu) if gpu is not None else ''}
        with (logs/f'{label}.log').open('a') as f:
            f.write('\nCOMMAND '+json.dumps(command)+'\n');f.flush()
            subprocess.run(command,cwd=HERE,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    base=NAS/'fulltrain_20261022';raw=NAS/'gmt_detector_comparison_20261023'
    try:
        while True:
            guard();complete=[]
            for label in ['gpu3','gpu7']:
                path=calibration/f'{label}_results.json';s=calibration/f'{label}_status.json'
                if s.exists() and json.loads(s.read_text())['stage']=='FAILED_REQUIRES_ENGINEERING_FIX':
                    raise RuntimeError('Point calibration partition failed: '+str(s))
                if path.exists() and json.loads(path.read_text())['status']=='COMPLETE_TRAIN_CALIBRATION_PARTITION':complete.append(label)
            state('WAITING_COMPLETE_REGISTERED_TRAIN_WINDOW_SCAN',complete_partitions=complete)
            if len(complete)==2:break
            time.sleep(30)
        totals=defaultdict(Counter);settings={};keys=set();result_artifacts=[]
        for label in ['gpu3','gpu7']:
            path=calibration/f'{label}_results.json';data=json.loads(path.read_text());registration=data['registration']
            assert registration['source_provenance']==provenance and not registration['TEST_used_for_selection']
            assert registration['budget_sha256']==sha256_file(watched[-1])
            assert registration['driver_sha256']==sha256_file(HERE/'scripts/calibrate_vggt_point_train.py')
            result_artifacts.append({'path':str(path),'sha256':sha256_file(path)})
            for row in data['results']:
                key=(row['method'],row['config'],row['scene'],row['window']);assert key not in keys;keys.add(key)
                assert sha256_file(row['manifest'])==row['manifest_sha256']
                totals[row['method'],row['config']].update(row['counts']);settings[row['method'],row['config']]=row['settings']
        assert len(keys)==2*18*6*3 and len(totals)==36
        selected={};ranking={}
        for method in ['temporal','joint']:
            names=[name for m,name in totals if m==method]
            names.sort(key=lambda n:rank_record(totals[method,n],settings[method,n]),reverse=True)
            selected[method]=settings[method,names[0]]
            ranking[method]=[{'config':n,'settings':settings[method,n],'counts':dict(totals[method,n]),
                'selection_rank':list(rank_record(totals[method,n],settings[method,n]))} for n in names]
        selection=root/'point_window_selection_v1.json'
        atomic_json(selection,{'status':'SELECTED_ON_TRAIN_WINDOWS_PENDING_COMPLETE_VALIDATION','settings':selected,
            'ranking':ranking,'objective':OBJECTIVE,'source_sha256':provenance['sha256'],
            'partitions':result_artifacts,'TEST_used_for_selection':False})
        state('SELECTED_POINT_COMPLETE_TRAIN_VALIDATION_RUNNING',settings=selected)
        def validate(index):
            method=['temporal','joint'][index]
            run(f'point_train_{method}',[PYTHON,'-u','scripts/run_vggt_point_fullvideo.py','--metadata',str(base/'audit/train_metadata.json'),
                '--split','train','--raw-cache',str(raw/'raw_train_v1'),'--scenes',*map(str,budget['validation_scenes']),
                '--method',method,'--train-settings',str(selection),'--constant-position-cache','--resource-mode','shared_gpu',
                '--output',str(root/'selected_train_fullvideo_v1'/method)],a.gpus[index])
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            for f in [pool.submit(validate,i) for i in range(2)]:f.result()
        validation=[]
        for method in ['temporal','joint']:
            for sid in budget['validation_scenes']:
                path=root/'selected_train_fullvideo_v1'/method/f'scene_{sid:02d}/manifest.json';m=json.loads(path.read_text())
                assert m['status']=='COMPLETE' and m['full_video'] and not m['GT_in_inference'] and m['settings']==selected[method]
                validation.append({'method':method,'scene':sid,'manifest_sha256':sha256_file(path)})
            run(f'point_train_eval_{method}',[PYTHON,'scripts/evaluate_fullvideo.py','--split','train','--audit',str(base/'audit/data_audit.json'),
                '--predictions',str(root/'selected_train_fullvideo_v1'/method),'--scenes',*map(str,budget['validation_scenes']),
                '--output',str(root/'selected_train_evaluation_v1'/method)])
        integrity=root/'point_raw_TEST_integrity_v1.json'
        if not integrity.exists():run('verify_point_raw_TEST',[PYTHON,'scripts/verify_gmt_raw_detection_cache.py',
            '--cache',str(raw/'raw_test_v1'),'--metadata',str(base/'audit/test_metadata.json'),'--output',str(integrity)])
        verified=json.loads(integrity.read_text());assert verified['status']=='VERIFIED_COMPLETE_IMMUTABLE_RAW_GMT_SPLIT' and verified['timestamps']==29019
        lock=root/'point_final_TEST_lock_v1.json'
        payload={'status':'FROZEN_FROM_TRAIN_VALIDATION','settings':selected,'TEST_used_for_selection':False,
            'source_sha256':provenance['sha256'],'runner_sha256':sha256_file(HERE/'scripts/run_vggt_point_fullvideo.py'),
            'constant_position_cache':True,'TEST_metadata_sha256':sha256_file(base/'audit/test_metadata.json'),
            'TEST_detector_lock_sha256':sha256_file(raw/'raw_test_v1/DETECTOR_LOCK.json'),'raw_TEST_integrity_sha256':sha256_file(integrity),
            'selection_sha256':sha256_file(selection),'complete_TRAIN_validation':validation,
            'TRAIN_official_metrics_sha256':{m:sha256_file(root/'selected_train_evaluation_v1'/m/'metrics.json') for m in ['temporal','joint']}}
        if lock.exists() and json.loads(lock.read_text())!=payload:raise RuntimeError('Do not overwrite a different point TEST lock')
        atomic_json(lock,payload)
        # Balance only by GT-free video length, preserving every video's order.
        meta=json.loads((base/'audit/test_metadata.json').read_text());lengths=Counter(r['video_id'] for r in meta['images'])
        partitions=[[] for _ in a.test_gpus];loads=[0]*len(partitions)
        for sid in sorted(lengths,key=lambda s:(-lengths[s],s)):
            index=min(range(len(loads)),key=lambda i:(loads[i],i));partitions[index].append(sid);loads[index]+=lengths[sid]
        for method in ['joint','temporal']:
            state('FROZEN_POINT_FULL22_TEST_RUNNING',method=method,lock_sha256=sha256_file(lock),partitions=partitions)
            def test_worker(index):
                run(f'point_test_{method}_gpu{a.test_gpus[index]}',[PYTHON,'-u','scripts/run_vggt_point_fullvideo.py',
                    '--metadata',str(base/'audit/test_metadata.json'),'--split','test','--raw-cache',str(raw/'raw_test_v1'),
                    '--method',method,'--lock',str(lock),'--constant-position-cache','--resource-mode','shared_gpu',
                    '--scenes',*map(str,partitions[index]),'--output',str(root/'test_v1'/method)],a.test_gpus[index])
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(a.test_gpus)) as pool:
                for f in [pool.submit(test_worker,i) for i in range(len(a.test_gpus))]:f.result()
            records=[json.loads(p.read_text()) for p in (root/'test_v1'/method).glob('scene_*/manifest.json')]
            assert len(records)==22 and all(m['status']=='COMPLETE' and m['full_video'] for m in records) and sum(m['frames'] for m in records)==29019
            run(f'point_test_evaluate_{method}',[PYTHON,'scripts/evaluate_fullvideo.py','--split','test','--audit',str(base/'audit/data_audit.json'),
                '--predictions',str(root/'test_v1'/method),'--output',str(root/'test_evaluation_v1'/method)])
        state('COMPLETE_POINT_FORMAL_TEST_EVALUATIONS',timing_limitation='Full TEST may share GPUs; retain exclusive paired speed audit separately')
    except BaseException as error:
        state('FAILED_REQUIRES_ENGINEERING_FIX',error=repr(error));raise


if __name__=='__main__':main()
