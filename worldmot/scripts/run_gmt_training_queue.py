#!/usr/bin/env python3
"""Wait for verified full cache, fit registered budgets, and freeze new weights.

This durable worker never starts TEST or reports partial-cache training as a
formal fit. Agent review/admission starts the subsequent independent TEST.
"""
from __future__ import annotations
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
VAL=[4,8,12,16,20,24]


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path)
    p.add_argument('--gpus',nargs='+',type=int,default=[2,6]);a=p.parse_args()
    if len(set(a.gpus))!=len(a.gpus) or not set(a.gpus)<={0,2,6}:
        raise ValueError('Use the registered detector-route cards, without occupying other users cards')
    root=a.root;root.mkdir(parents=True,exist_ok=True);(root/'logs').mkdir(exist_ok=True)
    status_path=root/'association_training_status.json'
    if status_path.exists():
        previous=json.loads(status_path.read_text());proc=Path('/proc')/str(previous.get('pid',-1))
        if proc.exists() and b'scripts/run_gmt_training_queue.py' in (proc/'cmdline').read_bytes().split(b'\0'):
            raise RuntimeError('A verified training controller already owns this root')
    budget_path=HERE/'configs/gmt_detector_association_budget_20261023.json'
    budget=json.loads(budget_path.read_text());budget_digest=sha(budget_path);provenance=source_stamp()
    cache=root/'fulltrain_objects_v1'
    def state(stage,**fields):
        atomic_json(status_path,{'stage':stage,'pid':os.getpid(),'updated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            'budget_sha256':budget_digest,'driver_sha256':sha(__file__),'goal_complete':False,**fields})
        print(json.dumps({'stage':stage,**fields}),flush=True)
    def run(name,command,gpu=None):
        if source_stamp()!=provenance:raise RuntimeError('Model/training source changed after queue registration')
        if sha(budget_path)!=budget_digest:raise RuntimeError('Registered training budget changed')
        env={**os.environ,'PYTHONPATH':str(HERE),'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1'}
        if gpu is not None:env['CUDA_VISIBLE_DEVICES']=str(gpu)
        with (root/'logs'/f'{name}.log').open('a') as stream:
            stream.write('\nCOMMAND '+json.dumps(command)+'\n');stream.flush()
            subprocess.run(command,cwd=HERE,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
    def fit(method,seed,path,scenes,gpu):
        command=[PYTHON,'scripts/train_world_query.py','--cache',str(cache),'--output',str(path),
            '--epochs','4','--seed',str(seed),'--ablation',method,'--scenes',*map(str,scenes)]
        if path.exists():
            import torch
            saved=torch.load(path,map_location='cpu',weights_only=True);m=saved['manifest']
            if (m.get('source_provenance')!=provenance or m.get('seed')!=seed
                or m.get('train_scenes')!=sorted(scenes) or m.get('detector_frontend')!='gmt_pre_gta_detector'
                or m['model_config'].get('ablation','joint')!=method or saved['epoch']>4):
                raise RuntimeError('Preserve mismatched checkpoint before attempting any resume')
            if saved['epoch']==4:return
            command+=['--resume',str(path)]
        run(path.stem,command,gpu)
    state('WAITING_FULL_24_GMT_OBJECT_CACHE')
    try:
        while True:
            complete=[];timestamps=0
            for sid in range(1,25):
                path=cache/f'scene_{sid:02d}'/'manifest.json'
                if not path.exists():continue
                try:m=json.loads(path.read_text())
                except (FileNotFoundError,json.JSONDecodeError):continue
                if m.get('status')=='COMPLETE':complete.append(sid);timestamps+=m['timestamps']
            state('WAITING_FULL_24_GMT_OBJECT_CACHE',complete_scenes=complete,timestamps=timestamps)
            if complete==list(range(1,25)) and timestamps==28754:break
            failed=[]
            for path in cache.glob('gpu*_status.json'):
                try:record=json.loads(path.read_text())
                except (FileNotFoundError,json.JSONDecodeError):continue
                if record['stage']=='FAILED_REQUIRES_ENGINEERING_FIX':failed.append(record)
            if failed:raise RuntimeError('Feature workers failed: '+repr(failed))
            time.sleep(30)
        state('VERIFYING_COMPLETE_GMT_OBJECT_CACHE')
        verification=root/'cache_verification'
        run('verify_all_GMT_objects',[PYTHON,'scripts/verify_fulltrain_cache.py','--cache',str(cache),'--output',str(verification)])
        verified=json.loads((verification/'coverage.json').read_text())
        if verified['status']!='VERIFIED_COMPLETE_24_TRAIN_OBJECT_CACHE' or verified['identity_label_mismatches'] or verified['timestamps']!=28754:
            raise RuntimeError('Full cache supervision gate failed')
        internal=root/'internal';internal.mkdir(exist_ok=True)
        train=[sid for sid in range(1,25) if sid not in VAL]
        state('INTERNAL_MATCHED_BUDGET_FITS',methods=['joint','no_geometry'],epochs=4,train_scenes=train,validation_scenes=VAL)
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(a.gpus)) as pool:
            def internal_worker(index):
                for method in ['joint','no_geometry'][index::len(a.gpus)]:
                    fit(method,20261022,internal/f'{method}.pt',train,a.gpus[index])
            fs=[pool.submit(internal_worker,i) for i in range(len(a.gpus))]
            for f in fs:f.result()
        state('TRAIN_ONLY_FIXED_BUDGET_SELECTION')
        selection=root/'selection'
        run('select_GMT_association',[PYTHON,'scripts/select_query_model.py','--cache',str(cache),
            '--internal-root',str(internal),'--output',str(selection),'--methods','joint','no_geometry','--candidate-epochs','4'],a.gpus[0])
        choice=json.loads((selection/'selection.json').read_text())
        if choice['status']!='FROZEN_FROM_TRAIN_VALIDATION' or choice['selected_epochs']!=4:
            raise RuntimeError('Unregistered refit budget or incomplete TRAIN validation')
        final=root/'final';final.mkdir(exist_ok=True)
        trials=[('joint',s) for s in [20261022,20261023,20261024]]+[('no_geometry',20261022)]
        state('FINAL_ALL24_MATCHED_BUDGET_FITS',trials=trials,epochs=4)
        # Per-card serial queues avoid accidentally overlapping two refits on
        # the same physical card. Other independent cards train in parallel.
        def worker(index):
            for method,seed in trials[index::len(a.gpus)]:
                fit(method,seed,final/f'{method}_seed{seed}.pt',list(range(1,25)),a.gpus[index])
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(a.gpus)) as pool:
            for future in [pool.submit(worker,i) for i in range(len(a.gpus))]:future.result()
        import torch
        checkpoints=[]
        for method,seed in trials:
            path=final/f'{method}_seed{seed}.pt';saved=torch.load(path,map_location='cpu',weights_only=True)
            m=saved['manifest']
            if source_stamp()!=provenance or m['source_provenance']!=provenance:
                raise RuntimeError('Source changed between registered fit and freezing')
            if not m['full_train_coverage'] or m['status']!='FULLTRAIN_COMPLETE' or m['detector_frontend']!='gmt_pre_gta_detector':
                raise RuntimeError('Subset/original-detector checkpoint cannot be frozen for new TEST')
            m.update(thresholds=choice['methods'][method]['thresholds'],formal_inference_allowed=True,
                     trained_checkpoint_sha256=sha(path),selection_sha256=sha(selection/'selection.json'))
            frozen=path.with_name(path.stem+'.frozen.pt');torch.save(saved,frozen)
            atomic_json(frozen.with_suffix('.json'),m)
            checkpoints.append({'method':method,'seed':seed,'path':str(frozen),'sha256':sha(frozen)})
        lock={'status':'FROZEN_BEFORE_INDEPENDENT_TEST','TEST_used_for_selection':False,
            'checkpoint_sha256':[r['sha256'] for r in checkpoints],'checkpoints':checkpoints,
            'source_provenance':provenance,'TRAIN_scenes':list(range(1,25)),
            'TEST_metadata_sha256':sha(Path('/data1/liuyeqiang/nas/A100/WorldMOT/fulltrain_20261022/audit/test_metadata.json')),
            'selection_sha256':sha(selection/'selection.json'),'budget_sha256':sha(budget_path),
            'supervision_verification_sha256':sha(verification/'coverage.json')}
        atomic_json(root/'final_test_lock.json',lock)
        state('FROZEN_GMT_ASSOCIATION_WEIGHTS_AWAIT_TEST_ADMISSION',checkpoints=checkpoints,TEST_started=False)
    except BaseException as error:
        state('FAILED_REQUIRES_ENGINEERING_FIX',error=repr(error));raise


if __name__=='__main__':main()
