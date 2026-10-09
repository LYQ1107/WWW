"""Owned child jobs only; prioritize idle GPUs, bounded RAM, never kill others."""
import argparse,subprocess,time,json,os
from pathlib import Path
from jev_phase13_learning import *

def main(kind,max_jobs):
    source=binding();assert not source['dirty'];assert kind in ['formal','onpolicy','validation']
    if kind=='formal':cases=[(v,s) for v in VARIANTS for s in SEEDS]
    elif kind=='onpolicy':cases=[(v,s) for v in ['full','motip','camel','set_transformer'] for s in SEEDS]
    else:cases=[(v,s) for v in VARIANTS for s in SEEDS]
    pending=[];done=[];failed=[];active={};root=OUT/f'queue_{kind}_v1';root.mkdir(parents=True,exist_ok=True);assert not (root/'RESULT.json').exists()
    for v,s in cases:
        result=OUT/f'{kind if kind != "validation" else "formal"}_training_v1'/v/f'seed{s}'/'RESULT.json'
        complete=all((OUT/'formal_online_v1'/f'{v}_seed{s}'/f'video{x:02d}'/'RESULT.json').exists() for x in VAL) if kind=='validation' else result.exists() and (kind!='onpolicy' or all((OUT/'onpolicy_online_v1'/f'{v}_seed{s}'/f'video{x:02d}'/'RESULT.json').exists() for x in VAL))
        (done if complete else pending).append((v,s))
    def command(v,s):
        if kind=='formal':return [PYTHON,str(ROOT/'reproduction_tools/train_jev_stage2.py'),'--phase','formal','--variant',v,'--seed',str(s)]
        if kind=='onpolicy':return [PYTHON,str(ROOT/'reproduction_tools/run_jev_phase13_case.py'),'--variant',v,'--seed',str(s),'--phase','onpolicy']
        return [PYTHON,str(ROOT/'reproduction_tools/run_jev_phase13_case.py'),'--variant',v,'--seed',str(s),'--phase','validation']
    def gpu():
        out=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True);candidates=[]
        for line in out.strip().splitlines():
            i,free,util=map(int,line.split(','));ours=sum(j['gpu']==i for j in active.values())
            if free>=4096 and ours<2:candidates.append((int(util>70),ours,int(util>5),-free,util,i))
        return min(candidates)[-1] if candidates else None
    def eligible(v,s):
        if kind=='formal':return True
        if not (OUT/'formal_training_v1'/v/f'seed{s}'/'RESULT.json').exists():return False
        if kind=='onpolicy':return (OUT/'formal_training_v1/full'/f'seed{s}'/'RESULT.json').exists()
        return True
    def memory_ready(available):
        # Concurrent queues share the same host. Reserve the not-yet-loaded
        # part of each Phase XIII worker's expected RAM, including workers
        # spawned by another queue. This prevents a startup race in which
        # several processes all observe the same apparently free memory.
        deficit=0.
        for proc in Path('/proc').iterdir():
            if not proc.name.isdigit():continue
            try:
                cwd=os.readlink(str(proc/'cwd'))
                if not cwd.startswith(str(OUT)+'/source_'):continue
                args=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
                if not any(n in args for n in ['train_jev_stage2.py','run_jev_phase13_case.py','evaluate_jev_stage2_online.py']):continue
                if 'python' not in Path(args.split()[0]).name:continue
                rss=int(next(l.split()[1] for l in (proc/'status').read_text().splitlines() if l.startswith('VmRSS:')))/1048576
                estimate=5. if 'evaluate_jev_stage2_online.py' in args or '--phase validation' in args else 11.
                deficit+=max(0.,estimate-rss)
            except (OSError,StopIteration,ValueError):continue
        required=5. if kind=='validation' else 11.
        return available-deficit>=14.+required
    begin=time.monotonic();last=0.
    while pending or active:
        for key,j in list(active.items()):
            code=j['process'].poll()
            if code is None:continue
            j['handle'].close();entry={'variant':key[0],'seed':key[1],'gpu':j['gpu'],'returncode':code,'log':str(j['log'])};(done if code==0 else failed).append(entry);del active[key];print('PHASE13_QUEUE_FINISHED',kind,key,code,flush=True)
        available=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))/1048576
        ready=next(((i,v,s) for i,(v,s) in enumerate(pending) if eligible(v,s)),None)
        if ready and len(active)<max_jobs and memory_ready(available):
            g=gpu()
            if g is not None:
                index,v,s=ready;pending.pop(index);path=root/f'{v}_seed{s}.log';h=path.open('a');env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(g),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1');p=subprocess.Popen(command(v,s),cwd=ROOT,env=env,stdout=h,stderr=subprocess.STDOUT);active[v,s]={'process':p,'gpu':g,'handle':h,'log':path};print('PHASE13_QUEUE_STARTED',kind,v,s,'GPU',g,'PID',p.pid,flush=True)
        if time.monotonic()-last>=15:
            save(root/'PROGRESS.json',{'status':'RUNNING','binding':source,'done':done,'failed':failed,'pending':pending,'active':[{'variant':v,'seed':s,'gpu':j['gpu'],'PID':j['process'].pid,'log':str(j['log'])} for (v,s),j in active.items()],'seconds':time.monotonic()-begin});last=time.monotonic()
        time.sleep(1)
    save(root/'RESULT.json',{'status':'PASS' if not failed else 'FAIL','binding':source,'done':done,'failed':failed,'seconds':time.monotonic()-begin});assert not failed,failed;print('PHASE13_QUEUE_COMPLETE',kind,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['formal','onpolicy','validation'],required=True);p.add_argument('--max-jobs',type=int,default=12);a=p.parse_args();main(a.kind,a.max_jobs)
