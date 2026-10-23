#!/usr/bin/env python3
"""Fixed-rule V1: complete TRAIN validation, freeze, full TEST, official eval."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import time
from cache_fulltrain import atomic_json,sha
from experiment_provenance import source_stamp

HERE=Path(__file__).resolve().parents[1]
PYTHON='/data1/liuyeqiang/envs/worldmot311/bin/python'
BASE=Path('/data1/liuyeqiang/nas/A100/WorldMOT/fulltrain_20261022')
VAL=[4,8,12,16,20,24]


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--gpus',nargs='+',type=int,default=[0,2,6]);a=p.parse_args()
    assert len(a.gpus)==len(set(a.gpus)) and set(a.gpus)<={0,2,6}
    root=a.root;root.mkdir(parents=True,exist_ok=True);logs=root/'logs';logs.mkdir(exist_ok=True)
    status=root/'v1_fixed_rule_queue_status.json'
    if status.exists():
        previous=json.loads(status.read_text());proc=Path('/proc')/str(previous.get('pid',-1))
        if proc.exists() and b'scripts/run_gmt_v1_queue.py' in (proc/'cmdline').read_bytes().split(b'\0'):
            raise RuntimeError('Verified V1 queue already running')
    provenance=source_stamp()
    def state(stage,**fields):
        atomic_json(status,{'stage':stage,'pid':os.getpid(),'updated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            'source_sha256':provenance['sha256'],'driver_sha256':sha(__file__),
            'gpus':a.gpus,'quality_selection':'fixed existing V1 defaults; no threshold search',
            'TEST_used_for_selection':False,'resource_mode':'shared_gpu','goal_complete':False,**fields})
        print(json.dumps({'stage':stage,**fields}),flush=True)
    def run(label,command,gpu=None):
        if source_stamp()!=provenance:raise RuntimeError('V1 source changed after queue registration')
        env={**os.environ,'PYTHONPATH':str(HERE),'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1'}
        if gpu is not None:env['CUDA_VISIBLE_DEVICES']=str(gpu)
        with (logs/f'{label}.log').open('a') as stream:
            stream.write('\nCOMMAND '+json.dumps(command)+'\n');stream.flush()
            subprocess.run(command,cwd=HERE,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
    def parallel(split,scenes,output,lock=None):
        def worker(index):
            assigned=scenes[index::len(a.gpus)]
            if not assigned:return
            command=[PYTHON,'-u','scripts/run_fullvideo.py','--metadata',str(BASE/'audit'/f'{split}_metadata.json'),
                '--split',split,'--method','v1','--detector','gmt_pre_gta','--gmt-raw-cache',str(root/f'raw_{split}_v1'),
                '--gmt-detector-mode','live_verified','--resource-mode','shared_gpu','--scenes',*map(str,assigned),'--output',str(output)]
            if lock:command+=['--lock',str(lock)]
            run(f'v1_{split}_gpu{a.gpus[index]}_v2',command,a.gpus[index])
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(a.gpus)) as pool:
            for future in [pool.submit(worker,i) for i in range(len(a.gpus))]:future.result()
        records=[]
        for scene in scenes:
            path=output/f'scene_{scene:02d}'/'manifest.json';m=json.loads(path.read_text())
            assert m['status']=='COMPLETE' and m['full_video'] and not m['GT_in_inference']
            assert m['source_sha256']==provenance['sha256'] and m['v1_output_boxes']=='current_detector'
            records.append({'scene':scene,'frames':m['frames'],'manifest_sha256':sha(path)})
        return records
    try:
        state('COMPLETE_SIX_SCENE_TRAIN_VALIDATION_RUNNING')
        val_output=root/'train_v1_validation_shared_v2'
        records=parallel('train',VAL,val_output)
        state('OFFICIAL_TRAIN_VALIDATION_EVALUATION',coverage=records)
        validation_metrics=root/'v1_train_validation_evaluation_v2'
        run('v1_train_official_evaluation_v2',[PYTHON,'scripts/evaluate_fullvideo.py','--split','train',
            '--audit',str(BASE/'audit/data_audit.json'),'--predictions',str(val_output),
            '--scenes',*map(str,VAL),'--output',str(validation_metrics)])
        state('VERIFY_COMPLETE_GT_FREE_TEST_CANDIDATES')
        integrity=root/'raw_test_integrity_v1.json'
        run('verify_raw_test_v1',[PYTHON,'scripts/verify_gmt_raw_detection_cache.py','--cache',str(root/'raw_test_v1'),
            '--metadata',str(BASE/'audit/test_metadata.json'),'--output',str(integrity)])
        verified=json.loads(integrity.read_text())
        assert verified['status']=='VERIFIED_COMPLETE_IMMUTABLE_RAW_GMT_SPLIT' and verified['timestamps']==29019 and verified['split']=='test'
        lock=root/'v1_fixed_rule_test_lock_v2.json'
        payload={'status':'FROZEN_FIXED_V1_AFTER_COMPLETE_TRAIN_VALIDATION','TEST_used_for_selection':False,
            'source_provenance':provenance,'TRAIN_validation_coverage':records,
            'TRAIN_validation_metrics_sha256':sha(validation_metrics/'metrics.json'),
            'raw_TEST_integrity_sha256':sha(integrity),'TEST_metadata_sha256':sha(BASE/'audit/test_metadata.json'),
            'association_training':'none; existing frozen rule/Hungarian defaults',
            'min_geometry_confidence':1.5,'min_geometry_points':16,'output_boxes':'current_detector',
            'original_failed_validation_preserved':'train_v1_validation_shared_gpu0_v1'}
        if lock.exists() and json.loads(lock.read_text())!=payload:raise RuntimeError('Do not overwrite a different freeze lock')
        atomic_json(lock,payload)
        state('FROZEN_V1_FULL22_TEST_RUNNING',lock_sha256=sha(lock))
        test_output=root/'test_gmt_v1_v2'
        coverage=parallel('test',list(range(1,23)),test_output,lock)
        assert sum(x['frames'] for x in coverage)==29019
        state('OFFICIAL_FULL22_TEST_EVALUATION',coverage=coverage)
        evaluation=root/'evaluation_gmt_v1_v2'
        run('v1_test_official_evaluation_v2',[PYTHON,'scripts/evaluate_fullvideo.py','--split','test',
            '--audit',str(BASE/'audit/data_audit.json'),'--predictions',str(test_output),'--output',str(evaluation)])
        state('COMPLETE_V1_FORMAL_EVALUATION',coverage=coverage,metrics_sha256=sha(evaluation/'metrics.json'),
              timing_limitation='Full-video accuracy runs share GPUs; require separate exclusive speed audit for method ranking')
    except BaseException as error:
        state('FAILED_REQUIRES_ENGINEERING_FIX',error=repr(error));raise


if __name__=='__main__':main()
