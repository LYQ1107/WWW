#!/usr/bin/env python3
"""Durable experiment driver. Status reflects completed work, never config files."""
from __future__ import annotations
import argparse,collections,concurrent.futures,hashlib,json,os,subprocess,sys,time
from pathlib import Path
import torch
from cache_fulltrain import atomic_json,sha
from experiment_provenance import source_stamp

ROOT=Path(__file__).resolve().parents[1]
PY='/data1/liuyeqiang/envs/worldmot311/bin/python'
VISION_PY='/data1/liuyeqiang/envs/worldmot_mmdet311/bin/python'
GMT_PY='/home/liuyeqiang/anaconda3/envs/GMT/bin/python'
GPUS=[0,2,3,4]
METHODS=['joint','no_omega','no_appearance','two_stage','no_geometry']
TRAIN_SCENES=[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19,21,22,23]

def snapshot(path):
    # Some NAS servers briefly expose a missing directory entry while an
    # atomic snapshot is replaced. Immutable shards remain checksum-checked.
    for attempt in range(3):
        try:return json.loads(Path(path).read_text())
        except (FileNotFoundError,json.JSONDecodeError):
            if attempt<2:time.sleep(.1)
    return None

class Experiment:
    def __init__(self,root):
        self.root=Path(root);self.logs=self.root/'logs';self.logs.mkdir(exist_ok=True);self.events=[]
    def state(self,stage,**kw):
        data={'stage':stage,'updated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw}
        atomic_json(self.root/'experiment_status.json',data)
        with (self.root/'events.jsonl').open('a') as f:f.write(json.dumps(data)+'\n')
        print(json.dumps(data),flush=True)
    def command(self,name,cmd,gpu=None):
        env={**os.environ,'PYTHONPATH':str(ROOT),'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'2'}
        if gpu is not None:env['CUDA_VISIBLE_DEVICES']=str(gpu)
        started=time.time()
        with (self.logs/(name+'.log')).open('a') as f:
            f.write('\nCOMMAND '+json.dumps(cmd)+'\n');f.flush();result=subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f'{name} exited {result.returncode}; see {self.logs/(name+".log")}')
        return {'name':name,'elapsed_s':time.time()-started,'log':str(self.logs/(name+'.log'))}
    def parallel(self,jobs):
        queues=[[] for _ in GPUS]
        for i,job in enumerate(jobs):queues[i%len(GPUS)].append(job)
        def worker(idx):
            return [fn(GPUS[idx]) for fn in queues[idx]]
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(GPUS)) as pool:
            futures=[pool.submit(worker,i) for i in range(len(GPUS))]
            return [r for f in futures for r in f.result()]
    def wait_cache(self):
        expected={s['scene']:s['synchronized_frames'] for s in json.loads((self.root/'audit/data_audit.json').read_text())['train']['scenes']}
        dead_polls=collections.Counter()
        while True:
            manifests=[];progress=[]
            for sid in sorted(expected):
                p=self.root/'train_cache'/f'scene_{sid:02d}'/'manifest.json'
                m=snapshot(p)
                if m and m['status']=='COMPLETE' and m['timestamps']==expected[sid]:manifests.append(m)
                q=p.with_name('progress.json')
                latest=snapshot(q)
                if latest:progress.append(latest)
            done=sum(p['done'] for p in progress)
            self.state('TRAIN_CACHE_RUNNING',completed_scenes=len(manifests),completed_timestamps=done,expected_timestamps=sum(expected.values()))
            if len(manifests)==len(expected):break
            complete={m['scene'] for m in manifests}
            for worker in json.loads((self.root/'cache_workers.json').read_text())['workers']:
                if set(worker['scenes'])<=complete:continue
                proc=Path('/proc')/str(worker['pid'])/'cmdline'
                if not proc.exists() or b'scripts/cache_fulltrain.py' not in proc.read_bytes().split(b'\0'):
                    dead_polls[worker['pid']]+=1
                    if dead_polls[worker['pid']]>=3:raise RuntimeError('cache worker stopped before full coverage; see '+worker['log'])
                else:dead_polls[worker['pid']]=0
            # Wait is confined to the background driver; the agent can inspect
            # files and continue useful work without blocking a tool call.
            time.sleep(30)
        for m in manifests:
            for shard in m['shards']:
                p=self.root/'train_cache'/f'scene_{m["scene"]:02d}'/shard['file']
                if sha(p)!=shard['sha256']:raise RuntimeError('full cache checksum mismatch')
        atomic_json(self.root/'cache_complete.json',{'status':'COMPLETE','scenes':manifests,'timestamps':sum(m['timestamps'] for m in manifests),'schema':manifests[0]['schema']})
        self.state('TRAIN_CACHE_COMPLETE',completed_scenes=24,completed_timestamps=28754)
    def train(self,method,seed,output,epochs,scenes,gpu):
        output=Path(output)
        if output.exists():
            saved=torch.load(output,map_location='cpu',weights_only=True)
            if saved['manifest'].get('source_provenance')!=source_stamp():raise RuntimeError('existing training checkpoint source differs: '+str(output))
            if saved['epoch']==epochs and saved['manifest'].get('train_scenes')==sorted(scenes):return {'name':output.stem,'status':'EXISTING_COMPLETE_CHECKPOINT'}
        cmd=[PY,'scripts/train_world_query.py','--cache',str(self.root/'train_cache'),'--output',str(output),'--epochs',str(epochs),'--seed',str(seed),'--device','cuda:0','--ablation',method,'--scenes',*map(str,scenes)]
        if output.exists():cmd+=['--resume',str(output)]
        return self.command(output.stem,cmd,gpu)
    def collect(self):
        return self.command('collect_actual_artifacts',[PY,'scripts/collect_fulltrain_results.py','--root',str(self.root)])
    def run(self):
        self.wait_cache()
        capacity=self.root/'audit/query_capacity.json'
        if not capacity.exists():
            self.command('audit_TRAIN_query_capacity',[PY,'scripts/audit_train_query_capacity.py','--output',str(capacity)])
        if json.loads(capacity.read_text())['status']!='GT_CLIP_CAPACITY_SUFFICIENT':raise RuntimeError('TRAIN query capacity audit failed')
        rgb_summary=self.root/'audit/test_RGB_decode_summary.json'
        rgb=snapshot(rgb_summary)
        if not rgb or rgb.get('status')!='ALL_TEST_RGB_DECODE_VERIFIED':
            self.command('verify_complete_TEST_RGB',[PY,'scripts/audit_test_rgb.py','--metadata',str(self.root/'audit/test_metadata.json'),'--output',str(self.root/'audit')])
            rgb=json.loads(rgb_summary.read_text())
        if rgb['decoded_images']!=58038 or rgb['synchronized_timestamps']!=29019 or len(rgb['scenes'])!=22 or rgb['metadata_sha256']!=sha(self.root/'audit/test_metadata.json'):
            raise RuntimeError('TEST RGB preflight does not cover the registered complete split')
        verification=self.root/'cache_verification/coverage.json'
        if not verification.exists():
            self.command('verify_all_train_cache',[PY,'scripts/verify_fulltrain_cache.py','--cache',str(self.root/'train_cache'),'--output',str(self.root/'cache_verification')])
        verified=json.loads(verification.read_text())
        if (verified.get('status')!='VERIFIED_COMPLETE_24_TRAIN_OBJECT_CACHE' or verified.get('timestamps')!=28754
            or [s['scene'] for s in verified.get('scenes',[])]!=list(range(1,25))
            or verified.get('identity_label_mismatches')!=0
            or verified.get('identity_labels_checked')!=sum(s['counts']['observations'] for s in verified.get('scenes',[]))
            or verified.get('verification_source_sha256')!=sha(ROOT/'scripts/verify_fulltrain_cache.py')):
            raise RuntimeError('full TRAIN supervision verification is missing, failed or produced by different verification source')
        self.collect()
        internal=self.root/'internal';internal.mkdir(exist_ok=True)
        self.state('TRAIN_INTERNAL_RUNNING',train_scenes=TRAIN_SCENES,validation_scenes=[4,8,12,16,20,24])
        jobs=[lambda gpu,m=m:self.train(m,20261022,internal/f'{m}.pt',6,TRAIN_SCENES,gpu) for m in METHODS]
        self.parallel(jobs);self.state('TRAIN_INTERNAL_COMPLETE');self.collect()
        selected=self.root/'selection'
        if not (selected/'selection.json').exists() or json.loads((selected/'selection.json').read_text()).get('status')!='FROZEN_FROM_TRAIN_VALIDATION':
            self.state('TRAIN_VALIDATION_SELECTING')
            self.command('train_validation',[PY,'scripts/select_query_model.py','--cache',str(self.root/'train_cache'),'--internal-root',str(internal),'--output',str(selected),'--device','cuda:0'],GPUS[0])
        choice=json.loads((selected/'selection.json').read_text());epochs=int(choice['selected_epochs'])
        if epochs<3:raise RuntimeError('selected epoch budget does not cover all registered clip lengths; audit TRAIN validation before proceeding')
        self.state('TRAIN_VALIDATION_FROZEN',selected_epochs=epochs,selection_sha256=sha(selected/'selection.json'))
        self.collect()
        final=self.root/'final';final.mkdir(exist_ok=True)
        trials=[('joint',s) for s in [20261022,20261023,20261024]]+[(m,20261022) for m in METHODS[1:]]
        self.state('FINAL_FULLTRAIN_RUNNING',trials=trials,epochs=epochs,train_scenes=list(range(1,25)))
        self.parallel([lambda gpu,m=m,s=s:self.train(m,s,final/f'{m}_seed{s}.pt',epochs,list(range(1,25)),gpu) for m,s in trials])
        frozen=[];sha_set=[]
        for method,seed in trials:
            p=final/f'{method}_seed{seed}.pt';saved=torch.load(p,map_location='cpu',weights_only=True);m=saved['manifest']
            if not m.get('full_train_coverage') or m.get('status')!='FULLTRAIN_COMPLETE':raise RuntimeError('incomplete full TRAIN checkpoint')
            m['thresholds']=choice['methods'][method]['thresholds'];m['formal_inference_allowed']=True;m['selection_sha256']=sha(selected/'selection.json');m['trained_checkpoint_sha256']=sha(p)
            target=final/f'{method}_seed{seed}.frozen.pt';torch.save(saved,target);target.with_suffix('.json').write_text(json.dumps(m,indent=2));sha_set.append(sha(target));frozen.append({'method':method,'seed':seed,'path':str(target),'sha256':sha_set[-1],'train_timestamps':m['total_timestamps']})
        lock={'status':'FROZEN_BEFORE_INDEPENDENT_TEST','checkpoint_sha256':sha_set,'checkpoints':frozen,'selection_sha256':sha(selected/'selection.json'),'TEST_used_for_selection':False,'TRAIN_scenes':list(range(1,25)),'TEST_metadata_sha256':sha(self.root/'audit/test_metadata.json'),'TEST_RGB_preflight_sha256':sha(rgb_summary),'TEST_RGB_fingerprint_records_sha256':rgb['per_image_fingerprints_sha256'],'source_provenance':source_stamp()}
        atomic_json(self.root/'final_test_lock.json',lock);self.state('FINAL_FULLTRAIN_COMPLETE',checkpoints=frozen);self.collect()
        # Every method uses actual RGB, with independent causal memory and the
        # same complete TEST metadata. Cache replay is not an E2E substitute.
        test=self.root/'test';test.mkdir(exist_ok=True);specs=[('v1',None,None,None)]
        specs += [('v2',r['method'],r['seed'],r['path']) for r in frozen]
        jobs=[]
        for kind,method,seed,checkpoint in specs:
            label='v1' if kind=='v1' else f'v2_{method}_seed{seed}'
            cmd=[VISION_PY,'scripts/run_fullvideo.py','--split','test','--metadata',str(self.root/'audit/test_metadata.json'),'--method',kind,'--output',str(test/label),'--device','cuda:0']
            if checkpoint:cmd+=['--checkpoint',checkpoint,'--lock',str(self.root/'final_test_lock.json')]
            jobs.append(lambda gpu,cmd=cmd,label=label:self.command('TEST_'+label,cmd,gpu))
        # The independently frozen native GMT comparator has its own environment.
        cmd=[GMT_PY,'scripts/run_gmt_test_baseline.py','--split','test','--metadata',str(self.root/'audit/test_metadata.json'),'--output',str(test/'gmt_native')]
        jobs.append(lambda gpu,cmd=cmd:self.command('TEST_gmt_native',cmd,gpu))
        self.state('INDEPENDENT_TEST_RUNNING',methods=len(jobs),test_scenes=22,timestamps_per_method=29019)
        self.parallel(jobs);self.state('INDEPENDENT_TEST_INFERENCE_COMPLETE')
        submissions=self.root/'test_submissions';submissions.mkdir(exist_ok=True)
        for pred in sorted(test.iterdir()):
            if not pred.is_dir():continue
            self.command('EXPORT_'+pred.name,[PY,'scripts/export_fullvideo_predictions.py','--metadata',str(self.root/'audit/test_metadata.json'),'--predictions',str(pred),'--output',str(submissions/pred.name)])
        self.command('EXPORT_v1_observed_bbox',[PY,'scripts/export_fullvideo_predictions.py','--metadata',str(self.root/'audit/test_metadata.json'),'--predictions',str(test/'v1'),'--output',str(submissions/'v1_observed_bbox'),'--box-source','tracks_2d_observation_boxes_by_camera'])
        # Offline GT opens only after all frozen causal prediction streams exist.
        evaluations=self.root/'test_evaluation';evaluations.mkdir(exist_ok=True)
        for pred in sorted(test.iterdir()):
            if not pred.is_dir():continue
            self.command('EVAL_'+pred.name,[PY,'scripts/evaluate_fullvideo.py','--split','test','--audit',str(self.root/'audit/data_audit.json'),'--predictions',str(pred),'--output',str(evaluations/pred.name)])
        self.command('EVAL_v1_observed_bbox',[PY,'scripts/evaluate_fullvideo.py','--split','test','--audit',str(self.root/'audit/data_audit.json'),'--predictions',str(test/'v1'),'--output',str(evaluations/'v1_observed_bbox'),'--box-source','tracks_2d_observation_boxes_by_camera'])
        self.state('OFFICIAL_TEST_EVALUATION_COMPLETE',results=str(evaluations),goal_requires_artifact_review_and_github_push=True)
        self.command('analyze_full_TEST_failures',[PY,'scripts/analyze_fulltrain_test_failures.py','--root',str(self.root),'--render'])
        self.command('collect_complete_artifacts',[PY,'scripts/collect_fulltrain_results.py','--root',str(self.root),'--require-complete'])
        self.state('RESULTS_READY_FOR_REVIEW_AND_GITHUB_PUSH',goal_complete=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='/data1/liuyeqiang/nas/A100/WorldMOT/fulltrain_20261022');a=p.parse_args();ex=Experiment(a.root)
    try:ex.run()
    except Exception as e:
        ex.state('FAILED_NEEDS_AGENT_FIX',error=f'{type(e).__name__}: {e}');raise
