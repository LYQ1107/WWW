"""Wait only named existing project handles; never restart/kill a process."""
import argparse
import time
from common import *
from run_research_queue import process_identity


def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--gate',type=Path);a=p.parse_args()
    spec=json.loads(a.spec.read_text());start=time.monotonic()
    while True:
        if time.monotonic()-start>6*3600:raise ValueError('bounded CPU dependency wait expired; inspect original handles, never auto-restart')
        if a.gate:
            if a.gate.exists():
                gate=json.loads(a.gate.read_text())
                if gate['status']!='PASS':raise ValueError('actual runtime prefix gate failed')
                print(json.dumps(dict(status='PASS_PREFIX_DEPENDENCY',gate_sha256=sha256(a.gate))));return
            central=RUNTIME/'queues/trackvggt_live_baseline_v1_gpu0/manifest.json'
            if central.exists():
                controller=json.loads(central.read_text())
                if controller['status']!='RUNNING' or process_identity(controller['pid']) is None:raise ValueError('original prefix controller ended without a passing gate; do not run or restart')
            # A named controller may not yet have started; no observation
            # timeout is treated as terminal. The bounded six-hour wait remains.
        else:
            live=[]
            for row in spec['teacher_dependencies']:
                current=process_identity(row['pid'])
                if current is not None and current[0]==row['start_ticks']:
                    if row['command_match'] not in current[1]:raise ValueError('named handle identity changed')
                    live.append(row['pid'])
            if not live:
                uuids=subprocess.check_output(['nvidia-smi','-i','0,4,5,6','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()
                apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True).splitlines()
                if not any(line.split(',')[0].strip() in uuids for line in apps):
                    save_json(RUNTIME/'matrix_reports/trackvggt_runtime_teacher_dependency_v1.json',dict(status='PASS_NAMED_JOBS_ENDED_AND_ALLOCATED_DEVICES_FREE',
                        waited_seconds=time.monotonic()-start,spec_sha256=sha256(a.spec),all_original_handles_absent=True,project_devices_free=True))
                    return
        time.sleep(10)


if __name__=='__main__':main()
