"""Pinned actors; bounded free-memory scheduling and observable per-job heartbeats."""
from jev_phase17_common import *
from jev_phase15_queue import gpus

def pin(name):
 protect();destination=OUT/name;head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
 assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
 if destination.exists():
  assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=destination,text=True).strip()==head;return destination
 subprocess.run(['git','worktree','add','--no-checkout','--detach',str(destination),head],cwd=ROOT,check=True)
 subprocess.run(['git','read-tree','HEAD'],cwd=destination,check=True)
 keep=[];skip=[]
 for path in filter(None,subprocess.check_output(['git','ls-files','-z'],cwd=destination).split(b'\0')):
  s=path.decode();(skip if s.startswith('TrackEval/data/') or s.endswith(('.pth','.pt','.gz','.xz')) else keep).append(path)
 for option,paths in [('update-index',skip),('checkout-index',keep)]:
  cmd=['git',option]+(['--skip-worktree'] if option=='update-index' else [])+['-z','--stdin'];subprocess.run(cmd,input=b'\0'.join(paths)+b'\0',cwd=destination,check=True)
 save(OUT/(name+'.json'),dict(status='PINNED',path=str(destination),commit=head));return destination

def queue(source,jobs,name):
 source=Path(source);head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip();folder=OUT/name;folder.mkdir(parents=True,exist_ok=True)
 pending=list(jobs);active={};done=[];failed=[];launches=[];start=time.monotonic()
 while pending or active:
  for pid,(gpu,p,stream,job) in list(active.items()):
   code=p.poll()
   if code is None:continue
   stream.close();result=Path(job['result']);good=code==0 and result.exists() and read(result)['status']=='COMPLETE'
   if good:assert read(result)['binding']['source_commit']==head
   (done if good else failed).append(dict(job,pid=pid,returncode=code));del active[pid];print('DONE',job['key'],code,flush=True)
  snapshot=gpus();assigned={g:sum(g==a[0] for a in active.values()) for g,*_ in snapshot};new={g:0 for g,*_ in snapshot}
  while pending and len(active)<6:
   choices=[g for g in snapshot if g[2]-assigned[g[0]]*4096>=8192 and (g[4]==0 or assigned[g[0]] or g[3]<20)]
   if not choices:break
   choices.sort(key=lambda g:(g[4]+new[g[0]]>0,g[3]+new[g[0]]*10,g[1]+new[g[0]]*4096,g[0]));gpu,used,free,util,count=choices[0];job=pending.pop(0)
   if Path(job['result']).exists():
    r=read(job['result']);assert r['status']=='COMPLETE' and r['binding']['source_commit']==head;done.append(dict(job,resumed=True));continue
   protect();env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
   log=folder/(job['key']+'.log');stream=log.open('a');cmd=[PYTHON,str(source/'reproduction_tools'/job['script'])]+job['args'];p=subprocess.Popen(cmd,cwd=source,env=env,stdout=stream,stderr=subprocess.STDOUT)
   active[p.pid]=(gpu,p,stream,job);launches.append(dict(key=job['key'],pid=p.pid,GPU=gpu,free_MiB=free,processes=count,utilization=util,command=cmd,log=str(log),source_commit=head));assigned[gpu]+=1;new[gpu]+=1;print('LAUNCH',job['key'],gpu,p.pid,flush=True)
  save(folder/'PROGRESS.json',dict(status='RUNNING',total=len(jobs),done=len(done),failed=len(failed),pending=len(pending),active=[dict(pid=pid,GPU=g,key=j['key']) for pid,(g,p,s,j) in active.items()],seconds=time.monotonic()-start))
  if pending or active:time.sleep(10)
 result=dict(status='COMPLETE' if not failed else 'COMPLETE_WITH_FAILURES',binding=binding(),actor_commit=head,done=done,failed=failed,launches=launches,seconds=time.monotonic()-start);save(folder/'RESULT.json',result);save(folder/'PROGRESS.json',result);return result

def main():
 assert read(REPORTS/'FACTORIAL_ENGINEERING_CONTRACT.json')['status']=='PASS_B0_EXACT_NATIVE_REFERENCE'
 source=pin('source_factorial_v1');jobs=[]
 for scope,videos in [('cold',DEV),('prefix',TRAIN)]:
  for arm in ['B0','B1','B2','B3']:
   for v in videos:jobs.append(dict(key=f'{scope}_{arm}_video{v}',script='jev_phase17_factorial.py',args=['--video',str(v),'--arm',arm,'--scope',scope],result=str(OUT/'factorial'/scope/arm/f'video{v:02d}/RESULT.json')))
 queue(source,jobs,'factorial_queue_v1')
if __name__=='__main__':main()
