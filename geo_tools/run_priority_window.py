"""Insert a serial priority experiment window after a current child completes."""
import argparse,datetime,json,os,signal,subprocess,sys,time
from pathlib import Path
from common import ROOT,RUNTIME,save_json,provenance
from run_research_queue import process_identity


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--controller-pid',type=int,required=True);parser.add_argument('--controller-match',required=True)
    parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--gpu',required=True);parser.add_argument('--tag',required=True);args=parser.parse_args()
    identity=process_identity(args.controller_pid)
    if identity is None or args.controller_match not in identity[1] or f'--gpu {args.gpu}' not in identity[1]:
        raise ValueError('not the named live project dispatcher on this GPU')
    out=RUNTIME/'priority_windows'/args.tag
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);report=dict(status='WAITING_FOR_EXISTING_CHILD',controller_pid=args.controller_pid,controller_starttime=identity[0],gpu=args.gpu,plan=str(args.plan),provenance=provenance())
    child=None;held=False
    def interrupt(signum,frame):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupt)
    try:
        os.kill(args.controller_pid,signal.SIGSTOP);held=True
        stop_deadline=time.monotonic()+5
        while True:
            stat=Path(f'/proc/{args.controller_pid}/stat').read_text()
            if stat[stat.rfind(')')+2:].split()[0] in {'T','t'}:break
            if time.monotonic()>stop_deadline:raise TimeoutError('dispatcher stop not observed; priority queue not launched')
            time.sleep(.01)
        children_path=Path(f'/proc/{args.controller_pid}/task/{args.controller_pid}/children')
        # Only the dispatcher is held. Its existing inference/evaluation child
        # continues uninterrupted; no competing GPU stage starts during it.
        children={int(pid):process_identity(int(pid)) for pid in children_path.read_text().split()}
        report['existing_children']={str(pid):state[0] if state else None for pid,state in children.items()}
        save_json(out/'manifest.json',report)
        for pid,state in children.items():
            if state is None:continue
            while True:
                current=process_identity(pid)
                if current is None or current[0]!=state[0]:break
                time.sleep(2)
        report['status']='RUNNING_PRIORITY_QUEUE';report['started_priority_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=args.gpu,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        with (out/'queue.log').open('x') as log:
            child=subprocess.Popen([sys.executable,str(ROOT/'geo_tools/run_research_queue.py'),'--plan',str(args.plan),'--queue',f'gpu{args.gpu}','--gpu',args.gpu],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            report['priority_controller_pid']=child.pid;save_json(out/'manifest.json',report)
            report['priority_queue_exit_code']=child.wait()
        report['status']='PRIORITY_COMMANDS_ENDED'
    except KeyboardInterrupt:
        report['status']='INTERRUPTED'
        raise
    finally:
        # On interruption, stop only this new owned process group before
        # allowing the original dispatcher to launch its next GPU job.
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            report['status']='INTERRUPTED'
        current=process_identity(args.controller_pid)
        if held and current is not None and current[0]==identity[0]:
            os.kill(args.controller_pid,signal.SIGCONT);report['original_dispatcher_resumed']=True
        else:report['original_dispatcher_resumed']=False
        save_json(out/'manifest.json',report)


if __name__=='__main__':main()
