"""One GPU grouped fit with a hard existing-quota cap and exact-state CPU continuation."""
import argparse,json,os,signal,subprocess,time
from pathlib import Path
from pirga_indexed_store import sha


def save(p,j):
    q=p.with_suffix('.tmp');q.write_text(json.dumps(j,indent=2)+'\n');q.replace(p)


def main():
    p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--protocol-sha',required=True);p.add_argument('--allocation',type=Path,required=True);p.add_argument('--allocation-sha',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if sha(a.protocol)!=a.protocol_sha or sha(a.allocation)!=a.allocation_sha:raise ValueError('Frozen protocol/allocation changed')
    allocation=json.loads(a.allocation.read_text());parent=allocation['parent_budget'];gpu=allocation['physical_gpu']
    if allocation['status']!='ALLOCATED_WITHIN_EXISTING_PARENT_CAP' or parent['recorded_usage']+allocation['previous_PIRGA_actual_GPU_hours']+allocation['maximum_GPU_wall_hours']>parent['cap'] or sha(parent['path'])!=parent['sha256']:raise ValueError('Parent quota changed or exceeded')
    if gpu in (0,1,4,5,6):raise ValueError('Reserved GPU excluded')
    if int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip()):raise ValueError('GPU occupied')
    a.out.mkdir(parents=True,exist_ok=False)
    with Path(str(a.allocation)+'.consumed').open('x') as f:f.write(str(a.out.resolve()))
    cmd=[allocation['python'],'-u',str(Path(__file__).with_name('train_pirga_grouped_phase1.py')),'--protocol',str(a.protocol),'--protocol-sha',a.protocol_sha,'--out',str(a.out/'fit'),'--device','cuda']
    ledger=dict(status='RUNNING_GPU',GPU_wall_hours=0.,GPU_cap=allocation['maximum_GPU_wall_hours'],physical_gpu=gpu,command=cmd,old_ledgers_reset=False)
    child=None;started=time.monotonic();lp=a.out/'RESOURCE_LEDGER.json';budget_stop=False
    try:
        with open(a.out/'GPU_FIT.log','x') as log:
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
            child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True);ledger['pid']=child.pid;save(lp,ledger)
            while child.poll() is None:
                elapsed=time.monotonic()-started;ledger['GPU_wall_hours']=elapsed/3600;save(lp,ledger)
                if sha(parent['path'])!=parent['sha256']:raise RuntimeError('Parent budget source changed')
                if elapsed>allocation['maximum_GPU_wall_hours']*3600-30:
                    budget_stop=True;os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=20)
                    except subprocess.TimeoutExpired:raise RuntimeError('Budget checkpoint did not finish within allowance')
                    break
                time.sleep(2)
            code=child.wait();ledger['GPU_wall_hours']=(time.monotonic()-started)/3600;ledger['GPU_exit_code']=code
            if code:raise RuntimeError('GPU training failed; preserve evidence, no automatic scientific retries')
        status=json.loads((a.out/'fit/STATUS.json').read_text())
        if status['status']=='RESOURCE_CHECKPOINT_READY_FOR_CPU_CONTINUATION' and budget_stop:
            ledger.update(status='RUNNING_CPU_CONTINUATION',GPU_pid=None,CPU_resume_snapshot_sha256=sha(a.out/'fit/RESUME.pth'));save(lp,ledger)
            cpu=cmd[:-1]+['cpu','--resume',str(a.out/'fit/RESUME.pth')]
            with open(a.out/'CPU_CONTINUATION.log','x') as log:
                c=subprocess.Popen(cpu,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8'),start_new_session=True);ledger['CPU_pid']=c.pid;save(lp,ledger)
                if c.wait():raise RuntimeError('CPU continuation failed')
        elif status['status']!='COMPLETE':raise RuntimeError('Unexpected trainer terminal state')
        ledger.update(status='COMPLETE_FIXED5EPOCHS',GPU_pid=None,CPU_pid=None)
    except BaseException as e:
        ledger.update(status='FAILED_RETAIN_EVIDENCE',reason=repr(e));raise
    finally:
        if child and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            ledger['GPU_wall_hours']=(time.monotonic()-started)/3600
        save(lp,ledger)

if __name__=='__main__':main()
