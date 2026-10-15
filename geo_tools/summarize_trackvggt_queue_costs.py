"""Actual queue-stage costs, excluding explicit CPU waits from GPU wall time.

Not a total project ledger: separate manual probes/DEV12/export and inherited
work are disclosed. Incomplete stages have no invented duration or zero cost.
"""
import csv
from common import *


def main():
    runtime=Path('/data1/liuyeqiang/WWW_trackvggt_runtime');rows=[];pending=[];queues=[]
    for path in sorted((runtime/'queues').glob('trackvggt_*/manifest.json')):
        queue=json.loads(path.read_text());source=Path(queue['provenance']['source_worktree'])
        plan_path=Path(queue['plan'])
        if not plan_path.is_absolute():plan_path=source/plan_path
        if sha256(plan_path)!=queue['plan_sha256']:raise ValueError('executed queue plan changed')
        plan=json.loads(plan_path.read_text());specs={s['id']:s for s in plan['queues'][queue['queue']]}
        queues.append(dict(path=str(path),sha256=sha256(path),status=queue['status'],pid=queue['pid']))
        for stage in queue['stages']:
            if stage['command']!=specs[stage['id']]['command']:raise ValueError('stage command differs from locked plan')
            if stage['status']=='SKIPPED_FAILED_DEPENDENCY':continue
            if 'seconds' not in stage:
                pending.append(dict(queue=path.parent.name,stage=stage['id'],status=stage['status'],pid=stage.get('pid')))
                continue
            if sha256(stage['log'])!=stage['log_sha256']:raise ValueError('completed queue stage log changed')
            cpu=bool(specs[stage['id']].get('cpu_only',False))
            rows.append(dict(queue=path.parent.name,stage=stage['id'],status=stage['status'],
                allocation='CPU_ONLY' if cpu else 'GPU_VISIBLE_PROCESS_WALL',gpu='' if cpu else queue['gpu'],
                process_seconds=stage['seconds'],GPU_visible_process_wall_hours=0 if cpu else stage['seconds']/3600,
                log=stage['log'],log_sha256=stage['log_sha256']))
    totals={}
    for row in rows:
        totals[row['queue']]=totals.get(row['queue'],0)+row['GPU_visible_process_wall_hours']
    output=ROOT/'results/trackvggt_queue_stage_costs_v1.csv'
    with output.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    release=json.loads((ROOT/'results/trackvggt_dev12_native_release_v1.json').read_text())
    proof=dict(status='PARTIAL_ACTUAL_QUEUE_STAGE_COSTS' if pending or any(q['status']=='RUNNING' for q in queues) else 'COMPLETE_ACTUAL_QUEUE_STAGE_COSTS',
        completed_stage_rows=len(rows),completed_GPU_visible_process_wall_hours=sum(totals.values()),
        by_queue_GPU_visible_process_wall_hours=totals,pending_or_running=pending,queues=queues,csv_sha256=sha256(output),
        completed_CPU_only_stage_seconds_including_waits_not_GPU_wall=sum(r['process_seconds'] for r in rows if r['allocation']=='CPU_ONLY'),
        separate_DEV12_release_GPU_wall_hours=release['gpu_wall_hours_including_off_on_prechecks'],
        cost_scope='Assigned-GPU producer process wall from actual queue monotonic clocks, including setup/hash/export/validation even when GPU idle; not active kernel hours. Explicit cpu_only stages and controller dependency waits excluded. Failed executed stages remain charged. Running/missing durations remain pending, not zero.',
        exclusions='Not the total project cost. Manual phase0/numerical/gradient probes, original selected/native initial capture, CPU diagnostics and separate DEV12 release costs are outside these queue-stage rows. Inherited prior-project ledger stays separate. Overlapping passive monitor samples are not summed as experiments.',
        scientific_success=False,provenance=provenance())
    save_json(ROOT/'results/trackvggt_queue_stage_costs_v1.json',proof)
    print(json.dumps(dict(status=proof['status'],completed_stages=len(rows),completed_GPU_visible_process_wall_hours=proof['completed_GPU_visible_process_wall_hours'],pending=len(pending))))


if __name__=='__main__':main()
