"""Resource-aware queue; prioritize empty GPUs and preserve other processes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from jev_phase6_common import OUT,ROOT,save


def main(path,max_workers):
    tasks=json.loads(path.read_text()); pending=list(tasks);running=[];completed=[]
    external_path=path.parent/'EXTERNAL_RESERVATIONS.json'
    while pending or running:
        for entry in list(running):
            if entry['process'].poll() is None:continue
            entry['handle'].close(); running.remove(entry)
            completed.append({'id':entry['task']['id'],'gpu':entry['gpu'],'exit_code':entry['process'].returncode})
        snapshot=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
        cards=[]
        for line in snapshot.splitlines():
            index,uuid,used,free,util=[x.strip() for x in line.split(',')]
            cards.append((int(index),uuid,int(used),int(free),int(util)))
        allocations=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True)
        own_actual={};actual_by_pid={}
        for line in allocations.splitlines():
            uuid,pid,memory=[x.strip() for x in line.split(',')]
            if memory.isdigit():actual_by_pid[(uuid,int(pid))]=int(memory)
            if memory.isdigit() and any(int(pid)==e['process'].pid for e in running):
                own_actual[uuid]=own_actual.get(uuid,0)+int(memory)
        # Reserve requested capacity immediately, before a new process has
        # initialized CUDA and before nvidia-smi can reflect its allocation.
        for index,task in list(enumerate(pending)):
            if len(running)>=max_workers:break
            skip=task.get('skip_if')
            if skip and Path(skip['path']).exists() and json.loads(Path(skip['path']).read_text()).get('status') in skip['statuses']:
                pending.remove(task);completed.append({'id':task['id'],'gpu':None,'exit_code':0,'status':'INELIGIBLE','reason':skip})
                continue
            if any(not Path(p).exists() for p in task.get('requires',[])):continue
            eligible=[]
            for gpu,uuid,used,free,util in cards:
                own=[e for e in running if e['gpu']==gpu]
                reserved=sum(e['task']['reserve_mib'] for e in own)
                external_used=max(0,used-own_actual.get(uuid,0))
                if external_path.exists():
                    planned={};own_pids={e['process'].pid for e in running}
                    for reservation in json.loads(external_path.read_text()):
                        if reservation['gpu']!=gpu or reservation['pid'] in own_pids:continue
                        try:os.kill(reservation['pid'],0)
                        except ProcessLookupError:continue
                        cmdline=Path(f"/proc/{reservation['pid']}/cmdline")
                        if not cmdline.exists() or not cmdline.read_bytes():continue
                        planned[reservation['pid']]=max(planned.get(reservation['pid'],0),reservation['reserve_mib'])
                    # Unreserved processes already count in observed usage.
                    # Add each known external process's unallocated budget,
                    # never replace the total observed usage with its budget.
                    external_used+=sum(max(0,budget-actual_by_pid.get((uuid,pid),0)) for pid,budget in planned.items())
                available=min(free,used+free-external_used-reserved)
                if available>=task['reserve_mib']+1024:
                    eligible.append((bool(used or own),util,len(own),-available,gpu))
            if not eligible and not task.get('cpu_only'):continue
            gpu=None if task.get('cpu_only') else min(eligible)[-1]
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
            log=OUT/'queue_logs'/f"{task['id']}.log";log.parent.mkdir(parents=True,exist_ok=True)
            handle=log.open('w');proc=subprocess.Popen(task['argv'],cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT)
            running.append({'task':task,'process':proc,'handle':handle,'gpu':gpu,'log':str(log)})
            pending.remove(task)
        save(OUT/(path.stem+'_STATUS.json'),{'pending':[t['id'] for t in pending],
            'running':[{'id':e['task']['id'],'pid':e['process'].pid,'gpu':e['gpu'],'log':e['log']} for e in running],
            'completed':completed})
        if pending or running:time.sleep(10)
    if any(r['exit_code'] for r in completed):raise RuntimeError('one or more queue tasks failed; inspect retained logs')
    print(json.dumps({'status':'COMPLETE','tasks':len(completed)}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('tasks',type=Path);p.add_argument('--max-workers',type=int,default=6)
    a=p.parse_args();main(a.tasks,a.max_workers)
