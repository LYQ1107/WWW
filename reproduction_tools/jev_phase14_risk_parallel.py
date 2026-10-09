import json,pathlib,os,subprocess,time,signal,sys
b=pathlib.Path('/data1/liuyeqiang/WWW_jev_phase14_runtime/20261010_v1');src=b/'source_diagnostic_v1';sys.path.insert(0,str(src/'reproduction_tools'))
from jev_phase14_queue import gpu_status
plan=json.load(open(b/'diagnostic_queue_v1/SCHEDULER_EXPANSION.json'));excluded={x['key'] for x in plan['active_children_preserved']};pid=plan['PID'];root=b/'risk_parallel_expansion_v1';root.mkdir(exist_ok=True);py='/home/liuyeqiang/anaconda3/envs/GMT/bin/python';jobs=[]
for phase in ['formal','onpolicy','oldloss_onpolicy','offpolicy']:
 for v in ['fixed_question','multi_question','set_transformer']:
  for s in [20261008,20261009,20261010]:
   key=f'risk_{phase}_{v}_{s}';p=b/'matched_TRAIN_native_risk_v1'/phase/f'{v}_seed{s}/RESULT.json'
   if key not in excluded and not p.exists():jobs.append(dict(key=key,phase=phase,variant=v,seed=s,result=str(p)))
active={};done=[];failed=[]
def save(name,d):
 p=root/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)
try:
 while jobs or active:
  for gpu,(proc,h,j) in list(active.items()):
   code=proc.poll()
   if code is None:continue
   h.close();j['returncode']=code;(done if code==0 and pathlib.Path(j['result']).exists() else failed).append(j);del active[gpu];print('RISK_FINISHED',j['key'],code,flush=True)
  candidates=[g for g in gpu_status() if g[0] not in active and g[2]>=8192];candidates.sort(key=lambda g:(g[1]>1000,g[3],g[1],g[0]))
  for gpu,used,free,util in candidates:
   if not jobs or len(active)>=8:break
   j=jobs.pop(0);log=root/(j['key']+'.log');h=log.open('a');e=os.environ.copy();e.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
   proc=subprocess.Popen([py,str(src/'reproduction_tools/jev_phase14_train_risk.py'),'--variant',j['variant'],'--seed',str(j['seed']),'--phase',j['phase']],cwd=src,env=e,stdout=h,stderr=subprocess.STDOUT);j['log']=str(log);active[gpu]=(proc,h,j);print('RISK_START',gpu,j['key'],flush=True)
  save('PROGRESS.json',dict(done=len(done),failed=len(failed),pending=len(jobs),active=[dict(GPU=g,key=v[2]['key'],pid=v[0].pid) for g,v in active.items()],actor_source_commit='062f2c5',scientific_matrix_unchanged=True));time.sleep(10)
 save('RESULT.json',dict(status='COMPLETE_WITH_FAILURES' if failed else 'COMPLETE',done=done,failed=failed,actor_source_commit='062f2c5',excluded_original_active=sorted(excluded)))
finally:
 os.kill(pid,signal.SIGCONT);print('ORIGINAL_SCHEDULER_RESUMED',pid,flush=True)
