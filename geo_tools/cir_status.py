"""Lightweight read-only progress: actual /proc, heartbeats and trace source frame."""
import argparse
import gzip
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);a=p.parse_args()
    ledger=json.loads((a.root/'results/CIR_GMT/RESOURCE_LEDGER.json').read_text())
    result=dict(measured_GPU_wall_hours=ledger['measured_GPU_wall_hours'],phases=json.loads((a.root/'results/CIR_GMT/RESEARCH_GOAL.json').read_text())['phases'],active=[])
    images=None
    for stage in ledger['stages']:
        if stage['status']!='RUNNING':continue
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
