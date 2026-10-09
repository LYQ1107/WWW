from pathlib import Path
import json,os,subprocess,time
src=Path('/data1/liuyeqiang/WWW_jev_phase14_runtime/20261010_v1/source_pipeline_v2');base=src.parent;root=base/'metrics_queue_v2';root.mkdir(exist_ok=True)
py='/home/liuyeqiang/anaconda3/envs/GMT/bin/python';p=json.load(open(src/'reports/JEV_PHASE14/FORMAL_PROTOCOL.json'))
jobs=[(phase,v,s) for phase in ['formal','onpolicy','offpolicy','oldloss_onpolicy'] for v in p['architectures'] for s in p['seeds']]
active={};done=[];failed=[]
def save(name,d):
 q=root/name;t=q.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(q)
while jobs or active:
 for key,(proc,h,job) in list(active.items()):
  code=proc.poll()
  if code is None:continue
  h.close();phase,v,s=job;result=base/f'{phase}_official_matlab_v2'/f'{v}_seed{s}/RESULT.json'
  (done if code==0 and result.exists() else failed).append(dict(phase=phase,variant=v,seed=s,returncode=code,result=str(result),log=str(root/(key+'.log'))));del active[key]
 for job in list(jobs):
  if len(active)>=3:break
  phase,v,s=job
  if not all((base/f'{phase}_online_v2'/f'{v}_seed{s}'/f'video{x:02d}/RESULT.json').exists() for x in [17,18,19]):continue
  jobs.remove(job);key=f'{phase}_{v}_{s}';h=(root/(key+'.log')).open('a');env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',PYTHONUNBUFFERED='1')
  code="import subprocess;subprocess.run("+repr([py,str(src/'reproduction_tools/jev_phase14_pooled.py'),'--variant',v,'--seed',str(s),'--phase',phase])+",check=True);subprocess.run("+repr([py,str(src/'reproduction_tools/jev_phase14_matlab.py'),'--variant',v,'--seed',str(s),'--phase',phase])+",check=True)"
  proc=subprocess.Popen([py,'-c',code],cwd=src,env=env,stdout=h,stderr=subprocess.STDOUT);active[key]=(proc,h,job);print('METRICS_START',key,flush=True)
 save('PROGRESS.json',dict(total=36,done=len(done),failed=len(failed),pending=len(jobs),active=[dict(key=k,pid=x[0].pid) for k,x in active.items()],source_commit='ca4d07f75a290f20aece8a1ce48c48a9fee503fa'))
 if not active and jobs and (base/'completion_queue_v2/RESULT.json').exists():
  failed.extend(dict(phase=j[0],variant=j[1],seed=j[2],reason='complete online prerequisite missing') for j in jobs);jobs=[]
 time.sleep(10)
save('RESULT.json',dict(status='COMPLETE_WITH_FAILURES' if failed else 'COMPLETE',done=done,failed=failed));print('METRICS_COMPLETE',len(done),len(failed),flush=True)
