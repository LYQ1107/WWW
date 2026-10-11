"""Observable bounded processes from immutable actor code, prioritizing idle GPUs."""
from jev_phase19_common import *
from jev_phase15_queue import gpus

def queue(jobs,name):
 folder=OUT/name;folder.mkdir(parents=True,exist_ok=True);assert not (folder/'RESULT.json').exists();pending=list(jobs);active={};done=[];failed=[];launches=[];start=time.monotonic()
 while pending or active:
  for pid,(gpu,p,stream,j) in list(active.items()):
   rc=p.poll()
   if rc is None:continue
   stream.close();good=rc==0 and Path(j['result']).exists() and read(j['result'])['status']=='COMPLETE';(done if good else failed).append(dict(j,pid=pid,returncode=rc));del active[pid]
  snapshot=gpus();new={g[0]:0 for g in snapshot}
  cache_live=0;cache_pending=False
  if name=='census_queue_v1':
   progress=read(OUT/'cache_queue_v1/PROGRESS.json');cache_pending=progress.get('pending',0)>0
   for old in progress.get('active',[]):
    try:os.kill(old['pid'],0);cache_live+=1
    except ProcessLookupError:pass
   terminal=OUT/'cache_queue_v1/RESULT.json'
   if terminal.exists():
    bad={int(j['args'][1]) for j in read(terminal)['failed']}
    for j in pending.copy():
     if int(j['args'][1]) in bad:pending.remove(j);failed.append(dict(j,returncode=None,reason='UPSTREAM_CACHE_FAILURE_NO_NATIVE_EXECUTION'))
  while pending and len(active)+cache_live<6 and not cache_pending:
   if name=='census_queue_v1':
    ready=[j for j in pending if (OUT/'stage1_cache_v1'/f'video{int(j["args"][1]):02d}/RESULT.json').exists()]
    if not ready:break
   else:ready=pending
   choices=[g for g in snapshot if g[2]-sum(g[0]==a[0] for a in active.values())*8192>=8192 and (g[4]==0 or g[3]<20)]
   if not choices:break
   choices.sort(key=lambda g:(g[4]+new[g[0]]>0,g[3]+new[g[0]]*10,g[1]+new[g[0]]*8192,g[0]));gpu,used,free,util,count=choices[0];j=ready[0];pending.remove(j);protect();stream=(folder/(j['key']+'.log')).open('w');env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1');cmd=[PYTHON,str(ROOT/'reproduction_tools'/j['script'])]+j['args'];p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT);active[p.pid]=(gpu,p,stream,j);new[gpu]+=1;launches.append(dict(key=j['key'],pid=p.pid,GPU=gpu,free_MiB=free,foreign_processes=count,utilization=util,command=cmd));print('LAUNCH',j['key'],gpu,p.pid,flush=True)
  save(folder/'PROGRESS.json',dict(status='RUNNING',done=len(done),failed=len(failed),pending=len(pending),active=[dict(pid=pid,GPU=g,key=j['key']) for pid,(g,p,s,j) in active.items()],seconds=time.monotonic()-start))
  if active or pending:time.sleep(10)
 r=dict(status='COMPLETE' if not failed else 'ENGINEERING_FAIL',binding=binding(),done=done,failed=failed,launches=launches,seconds=time.monotonic()-start);save(folder/'RESULT.json',r);save(folder/'PROGRESS.json',r)
if __name__=='__main__':queue([dict(key=f'cache_video{v}',script='jev_phase19_cache.py',args=['--video',str(v)],result=str(OUT/'stage1_cache_v1'/f'video{v:02d}/RESULT.json')) for v in SCOPE],'cache_queue_v1')
