"""Transfer unlaunched stages while preserving the current GPU child."""
import argparse
import signal
import time
from common import *
from run_research_queue import process_identity


def validate_plan(original, replacement, observed):
    stages=original['queues'][replacement['original_queue']]
    index=next(i for i,s in enumerate(stages) if s['id']==replacement['existing_stage'])
    if observed[-1]['id']!=stages[index]['id'] or observed[-1]['status']!='RUNNING':
        raise ValueError('dispatcher advanced beyond expected child')
    expected={s['id']:s for s in stages[index+1:]}
    supplied=[s for values in replacement['queues'].values() for s in values]
    if len(supplied)!=len(expected) or {s['id'] for s in supplied}!=set(expected):
        raise ValueError('remaining stages must transfer exactly once')
    passed={s['id'] for s in observed if s['status']=='PASS'}
    for queue,values in replacement['queues'].items():
        local={s['id'] for s in values}
        for stage in values:
            old=expected[stage['id']]
            if {k:v for k,v in stage.items() if k!='depends_on'}!={k:v for k,v in old.items() if k!='depends_on'}:
                raise ValueError('stage command/options changed')
            if stage.get('depends_on',[])!=[d for d in old.get('depends_on',[]) if d in local]:
                raise ValueError('internal dependency changed')
            external=set(old.get('depends_on',[]))-local
            if external-passed-{replacement['existing_stage']}:
                raise ValueError('unverified external dependency')
            if replacement['existing_stage'] in external and queue!='gpu0':
                raise ValueError('current child evaluation must wait on GPU0')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--controller-pid',type=int,required=True)
    parser.add_argument('--existing-child-pid',type=int,required=True)
    args=parser.parse_args()
    plan=json.loads(args.plan.read_text())
    queue_path=RUNTIME/'queues/primary_vggt_inference_seed_pairs_v1_gpu0/manifest.json'
    queue=json.loads(queue_path.read_text());source=Path(queue['provenance']['source_worktree'])
    original_path=source/plan['original_plan']
    if sha256(original_path)!=queue['plan_sha256']:raise ValueError('original plan changed')
    original=json.loads(original_path.read_text());validate_plan(original,plan,queue['stages'])
    for stage in queue['stages']:
        if stage['status']=='PASS' and sha256(stage['log'])!=stage['log_sha256']:
            raise ValueError('completed prefix/gate log changed')
    if queue['pid']!=args.controller_pid or queue['stages'][-1]['pid']!=args.existing_child_pid:raise ValueError('wrong processes')
    parent=process_identity(args.controller_pid);child=process_identity(args.existing_child_pid)
    if parent is None or child is None or 'geo_tools/run_research_queue.py' not in parent[1] or plan['existing_stage'] not in child[1]:
        raise ValueError('named live process identities required')
    for file,digest in queue['provenance']['source_hashes'].items():
        if sha256(source/file)!=digest:raise ValueError('original source changed: '+file)
    uuid=subprocess.check_output(['nvidia-smi','-i','4','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    if any(line.split(',')[0].strip()==uuid for line in apps.splitlines()):raise ValueError('GPU4 is occupied')
    for values in plan['queues'].values():
        for stage in values:
            command=stage['command']
            if '--tag' in command and (RUNTIME/'online'/command[command.index('--tag')+1]).exists():
                raise FileExistsError('future output already exists')
    out=RUNTIME/'handoffs'/plan['id']
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    report=dict(status='PREPARED',controller_pid=args.controller_pid,controller_starttime=parent[0],
        existing_children={str(args.existing_child_pid):child[0]},plan=str(args.plan),plan_sha256=sha256(args.plan),
        original_queue=str(queue_path),original_queue_snapshot=queue,source_worktree=str(source),
        inference_source_unchanged=True,completed_prefix_log_hashes_checked=True,free_gpu4_uuid=uuid,provenance=provenance())
    held=False;launched=[]
    def save():save_json(out/'manifest.json',report)
    def launch(command,name,cwd):
        with (out/name).open('x') as log:
            process=subprocess.Popen(command,cwd=cwd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'))
        return process.pid
    try:
        os.kill(args.controller_pid,signal.SIGSTOP);held=True
        deadline=time.monotonic()+5
        while Path(f'/proc/{args.controller_pid}/stat').read_text().rsplit(')',1)[1].split()[0] not in {'T','t'}:
            if time.monotonic()>deadline:raise TimeoutError('stop not observed')
            time.sleep(.01)
        children=list(map(int,Path(f'/proc/{args.controller_pid}/task/{args.controller_pid}/children').read_text().split()))
        if children!=[args.existing_child_pid] or process_identity(args.existing_child_pid)!=child:raise ValueError('child changed')
        validate_plan(original,plan,json.loads(queue_path.read_text())['stages'])
        report.update(status='DISPATCHER_HELD_CHILD_UNINTERRUPTED',held_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save()
        for gpu in ('0','4'):
            command=[sys.executable,'-u',str(source/'geo_tools/run_research_queue.py'),'--plan',str(args.plan),'--queue','gpu'+gpu,'--gpu',gpu]
            if gpu=='0':command+=['--after-pid',str(args.existing_child_pid),'--after-command-match',plan['existing_stage']]
            launched.append(launch(command,'queue_gpu'+gpu+'.log',source))
        report['replacement_controller_pids']=launched;save()
        # Keep the original probe intact; a new probe follows replacement roots.
        probe_source=RUNTIME/'pinned_sources/resource_probe_v1'
        roots=[args.controller_pid,*launched,*[pid for pid in (24757,29388,3006) if process_identity(pid) is not None]]
        command=[sys.executable,'-u',str(probe_source/'geo_tools/monitor_project_resources.py'),'--tag','remaining_declared_jobs_balanced_handoff_v1','--interval','2','--controller-pids',*map(str,roots)]
        report['resource_probe_pid']=launch(command,'resource_probe.log',probe_source)
        report['status']='REPLACEMENT_QUEUES_RUNNING';save()
        while (state:=process_identity(args.existing_child_pid)) is not None and state[0]==child[0]:time.sleep(2)
        if process_identity(args.controller_pid)!=parent:raise ValueError('dispatcher identity changed')
        children=list(map(int,Path(f'/proc/{args.controller_pid}/task/{args.controller_pid}/children').read_text().split()))
        if any(process_identity(pid) is not None for pid in children):raise ValueError('unexpected live child')
        # The sole original GPU child has exited. A pending default SIGTERM
        # terminates the stopped dispatcher on resume before new work launches.
        os.kill(args.controller_pid,signal.SIGTERM);os.kill(args.controller_pid,signal.SIGCONT)
        deadline=time.monotonic()+5
        while process_identity(args.controller_pid) is not None:
            if time.monotonic()>deadline:raise TimeoutError('dispatcher termination not observed')
            time.sleep(.05)
        held=False
        report.update(status='ORIGINAL_DISPATCHER_SUPERSEDED_AFTER_CHILD_EXIT',existing_child_exited_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            original_dispatcher_resumed=False,model_processes_signaled=False,
            note='original queue retained as superseded, not relabeled complete; replacements own only original unlaunched stages and evaluation of the completed original child')
        save()
    except BaseException as error:
        report.update(status='HANDOFF_SUPERVISION_FAILED',error=repr(error),replacement_controller_pids=launched)
        if held and not launched and process_identity(args.controller_pid)==parent:
            os.kill(args.controller_pid,signal.SIGCONT);report['original_dispatcher_resumed']=True
        save();raise


if __name__=='__main__':main()
