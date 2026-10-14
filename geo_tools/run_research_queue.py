"""Execute a fixed experiment queue serially on one explicitly assigned GPU."""
import argparse
import datetime
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from common import ROOT, RUNTIME, save_json, sha256, provenance


def process_identity(pid):
    path = Path('/proc')/str(pid)
    try:
        stat = (path/'stat').read_text()
        # Process names can contain spaces: starttime is field22 after the ')'.
        fields = stat[stat.rfind(')')+2:].split()
        if fields[0] == 'Z':
            return None
        started = fields[19]
        command = (path/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
        return started, command
    except FileNotFoundError:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--queue', required=True)
    parser.add_argument('--gpu', required=True)
    parser.add_argument('--after-pid', type=int)
    parser.add_argument('--after-command-match', default='')
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if args.gpu not in plan['project_gpu_ids']:
        raise ValueError('GPU outside this project allocation')
    stages = plan['queues'][args.queue]
    out = RUNTIME/'queues'/f"{plan['id']}_{args.queue}"
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    report = dict(status='RUNNING', queue=args.queue, gpu=args.gpu, plan=str(args.plan), plan_sha256=sha256(args.plan),
                  provenance=provenance(), stages=[], pid=os.getpid())
    save_json(out/'manifest.json', report)
    if args.after_pid:
        identity = process_identity(args.after_pid)
        if identity is not None:
            if not args.after_command_match or args.after_command_match not in identity[1]:
                raise ValueError('existing process is not the specified project job')
            report['waits_for'] = dict(pid=args.after_pid,starttime=identity[0],command_match=args.after_command_match)
            save_json(out/'manifest.json', report)
            while True:
                current = process_identity(args.after_pid)
                if current is None or current[0] != identity[0]:
                    break
                time.sleep(10)
    completed = set()
    for stage in stages:
        if not re.fullmatch('[A-Za-z0-9_]+', stage['id']):
            raise ValueError('unsafe stage ID')
        log = RUNTIME/'logs'/f"{plan['id']}_{args.queue}_{stage['id']}.log"
        row = dict(id=stage['id'],command=stage['command'],log=str(log),status='RUNNING',started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        report['stages'].append(row)
        if not set(stage.get('depends_on', [])) <= completed:
            row['status']='SKIPPED_FAILED_DEPENDENCY'
            save_json(out/'manifest.json',report)
            continue
        env = dict(os.environ,CUDA_VISIBLE_DEVICES='' if stage.get('cpu_only') else args.gpu,
                   OMP_NUM_THREADS='1' if stage.get('cpu_only') else '4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        started = time.perf_counter()
        with log.open('x') as stream:
            child = subprocess.Popen([sys.executable, *stage['command']],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
            row['pid']=child.pid
            save_json(out/'manifest.json',report)
            print(json.dumps(dict(stage=stage['id'],pid=child.pid,log=str(log))),flush=True)
            code = child.wait()
        row.update(exit_code=code,seconds=time.perf_counter()-started,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),log_sha256=sha256(log))
        row['status']='PASS' if code==0 else 'FAILED'
        if code==0:
            completed.add(stage['id'])
        save_json(out/'manifest.json',report)
    report['status']='FINISHED' if all(row['status']=='PASS' for row in report['stages']) else 'FINISHED_WITH_FAILURES'
    # FINISHED means planned commands ended; it is not scientific success or
    # the completion audit. Every resulting run/metric needs independent audit.
    save_json(out/'manifest.json',report)


if __name__ == '__main__':
    main()
