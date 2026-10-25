#!/usr/bin/env python3
"""Give completed Full/Head TEST native scoring priority without stopping jobs.

Uses the unchanged registered evaluator/wrapper and equivalent job arguments.
The scheduler's three-CPU queue limit is relaxed for one priority scorer when
at least 16 GiB RAM is available. GPU workers and R2 source are never changed.
"""
import argparse
import builtins
import importlib.util
import json
from pathlib import Path
import sys
import time


def read(path):return json.loads(path.read_text())


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    repo=Path(__file__).resolve().parents[2];root=a.root
    launcher=repo/'worldmot/scripts/launch_worldobject_v3_job.py'
    spec=importlib.util.spec_from_file_location('frozen_R2_priority_eval_launcher',launcher)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.sum=lambda values:builtins.sum(bool(x) for x in values)
    todo=['FULL_WORLD_OBJECT_V3','HEAD_ONLY_NO_GEOMETRY_V3']
    while todo:
        for method in list(todo):
            name='evaluate_'+method;status=root/'jobs'/(name+'.status.json')
            if status.exists():todo.remove(method);continue
            inference=root/'jobs'/('infer_'+method+'.status.json')
            if not inference.exists() or read(inference).get('status')!='COMPLETE':continue
            available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
            if available<16*(1<<30):continue
            path=root/'job_specs'/(name+'.json')
            if not path.exists():
                template=read(root/'job_specs/evaluate_FULL_NO_GEOMETRY_V3.json')
                job={**template,'job_id':name,'arguments':[v.replace('FULL_NO_GEOMETRY_V3',method) for v in template['arguments']]}
                if job['gpu_uuid'] is not None or job['script']!='evaluate_fullvideo.py':raise ValueError('UNEXPECTED_PRIORITY_TEMPLATE')
                # Never overwrite a job spec created by the normal scheduler.
                try:
                    with path.open('x') as f:f.write(json.dumps(job,indent=2)+'\n')
                except FileExistsError:pass
            old=sys.argv;sys.argv=[str(launcher),'--job',str(path)]
            try:module.main()
            finally:sys.argv=old
            with (root/'priority_official_launches.jsonl').open('a') as stream:
                stream.write(json.dumps({'time':time.time(),'method':method,'job':name,
                    'evaluator_or_predictions_changed':False,'normal_jobs_stopped':0,
                    'source_modified':False,'available_RAM_bytes':available,'CPU_queue_priority_only':True})+'\n')
            todo.remove(method)
        if todo:time.sleep(5)


if __name__=='__main__':main()
