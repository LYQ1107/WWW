"""Bounded evidence controller, with visible progress and no implicit scientific pass."""
import argparse
import fcntl
from jev_oneweek_common import *


def status():
    registry=read(REPORTS/'EXPERIMENT_REGISTRY.json')
    progress={}
    for path in OUT.rglob('PROGRESS.json'):
        value=read(path)
        progress[str(path.relative_to(OUT))]={k:value.get(k) for k in
            ('status','key','frames','counts','elapsed','step','total_steps') if k in value}
    return dict(registry=registry,runtime_progress=progress,goal=read(REPORTS/'GO_NO_GO.json'),
                new_retained_MiB=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())/2**20)


def run_job(name,command,gpu=None):
    protect()
    env=dict(os.environ)
    if gpu is not None:env['CUDA_VISIBLE_DEVICES']=str(gpu)
    log=OUT/'controller_logs'/f'{name}_{time.time_ns()}.log';log.parent.mkdir(exist_ok=True)
    with log.open('x') as stream:
        with (OUT/'REGISTRY.lock').open('a') as registry_lock:
            fcntl.flock(registry_lock,fcntl.LOCK_EX)
            registry=read(REPORTS/'EXPERIMENT_REGISTRY.json')
            job=registry['jobs'].setdefault(name,dict(status='PLANNED',results=None))
            if str(job['status']).startswith('COMPLETE'):
                print('SKIP_COMPLETE',name,flush=True);return
            if job['status']=='RUNNING':
                pid=job.get('PID')
                if pid and Path('/proc',str(pid)).exists():raise RuntimeError('Job already running')
                raise RuntimeError('Stale RUNNING job needs recorded recovery, not blind restart')
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
            job.update(status='RUNNING',PID=process.pid,command=command,GPU=gpu,log=str(log))
            registry['status']='BOUNDED_EXPERIMENTS_RUNNING'
            save(REPORTS/'EXPERIMENT_REGISTRY.json',registry)
        code=process.wait()
    with (OUT/'REGISTRY.lock').open('a') as registry_lock:
        fcntl.flock(registry_lock,fcntl.LOCK_EX)
        registry=read(REPORTS/'EXPERIMENT_REGISTRY.json');job=registry['jobs'][name]
        job.update(status='COMPLETE' if code==0 else 'ENGINEERING_OR_SCIENCE_FAILURE_REVIEW_REQUIRED',exit_code=code,log=ref(log))
        save(REPORTS/'EXPERIMENT_REGISTRY.json',registry)
    if code:raise RuntimeError(name+' failed; retained log '+str(log))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--status',action='store_true')
    parser.add_argument('--run-script')
    parser.add_argument('--job')
    parser.add_argument('--gpu',type=int)
    parser.add_argument('--args',nargs=argparse.REMAINDER,default=[])
    a=parser.parse_args()
    if a.status or not a.run_script:
        print(json.dumps(status(),indent=2));return
    assert a.job
    path=ROOT/'reproduction_tools'/a.run_script
    assert path.parent==ROOT/'reproduction_tools' and path.name.startswith('jev_oneweek_') and path.exists()
    OUT.mkdir(parents=True,exist_ok=True)
    locks=OUT/'job_locks';locks.mkdir(exist_ok=True)
    assert '/' not in a.job and a.job not in ['.','..']
    with (locks/(a.job+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run_job(a.job,[PYTHON,str(path)]+a.args,a.gpu)


if __name__=='__main__':main()
