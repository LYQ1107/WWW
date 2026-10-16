"""Lightweight read-only progress: actual /proc, heartbeats and trace source frame."""
import argparse
import gzip
import json
import re
import time
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);a=p.parse_args()
    ledger=json.loads((a.root/'results/CIR_GMT/RESOURCE_LEDGER.json').read_text())
    result=dict(measured_GPU_wall_hours=ledger['measured_GPU_wall_hours'],phases=json.loads((a.root/'results/CIR_GMT/RESEARCH_GOAL.json').read_text())['phases'],active=[])
    images=None
    for stage in ledger['stages']:
        if not stage['status'].startswith('RUNNING'):continue
        if stage['status']=='RUNNING_BATCH':
            jobs=stage['jobs'];running=[]
            for job in jobs:
                if job['status']!='RUNNING':continue
                pid=job['pid'];proc=Path(f'/proc/{pid}/cmdline');log=Path(job['log'])
                frame=None
                if log.exists():
                    with log.open('rb') as stream:
                        stream.seek(max(0,log.stat().st_size-8192));tail=stream.read().decode(errors='replace')
                    matches=re.findall(r'(\d+)/(\d+)\s*\[',tail)
                    if matches:frame=dict(completed_groups=int(matches[-1][0]),total_groups=int(matches[-1][1]))
                running.append(dict(job=job['name'],gpu=job['gpu'],pid=pid,alive=proc.exists(),
                    actual_command=proc.read_bytes().replace(b'\x00',b' ').decode() if proc.exists() else None,
                    progress=frame,log_last_update_seconds_ago=time.time()-log.stat().st_mtime if log.exists() else None,log=str(log)))
            result['active'].append(dict(stage=stage['stage'],elapsed_seconds=stage['elapsed_seconds'],
                GPU_wall_hours_including_live=stage['GPU_wall_hours_including_live'],
                completed_jobs=sum(j['status']=='COMPLETE' for j in jobs),total_jobs=len(jobs),running=running))
            continue
        pid=stage.get('child_pid');cmd=Path(f'/proc/{pid}/cmdline')
        actual=cmd.read_bytes().replace(b'\x00',b' ').decode() if cmd.exists() else None
        item=dict(stage=stage['stage'],pid=pid,actual_command=actual,elapsed_seconds=stage.get('elapsed_seconds'),log=stage['log'])
        command=stage['command']
        if '--out' in command:
            out=Path(command[command.index('--out')+1]);traces=[]
            for path in sorted(out.glob('**/actual_decisions.jsonl.gz')):
                count=0;last=None;complete=True
                try:
                    with gzip.open(path,'rt') as stream:
                        for line in stream:
                            row=json.loads(line);count+=1
                            if row.get('queries'):last=row['queries'][-1][0]
                except (EOFError,OSError):complete=False
                if last is not None:
                    if images is None:
                        data=json.loads(Path('/data/DATASETS/TRACKING/JDE/VisionTrack/annotations/train.json').read_text())
                        images={r['id']:r for r in data['images']}
                    metadata=images[last]
                    last=dict(image_id=last,frame=metadata['frame_id'],camera=metadata['view_id'])
                traces.append(dict(arm=str(path.parent.relative_to(out)),complete_records_read=count,last_observation=last,stream_closed=complete,bytes=path.stat().st_size))
            item['trace_progress_diagnostic_only']=traces
        result['active'].append(item)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
