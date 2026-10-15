"""Execute the unchanged same-checkpoint gate diagnostic at fixed epochs3/5."""
import argparse,json,time,datetime,os
from pathlib import Path
from pirga_indexed_store import sha
from run_pirga_uncertainty_gate_audit import execute


def main():
    p=argparse.ArgumentParser();p.add_argument('--runtime',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    code=Path(__file__).with_name('run_pirga_uncertainty_gate_audit.py');registration=json.loads((Path(__file__).resolve().parents[1]/'results/PIRGA_FULLTRAIN/UNCERTAINTY_GATE_DIAGNOSTIC_REGISTRATION.json').read_text());expected=registration['source_sha256']
    if sha(code)!=expected:raise ValueError('Locked gate audit source changed')
    a.out.mkdir(parents=True,exist_ok=False);status=dict(status='WAITING_FIXED_ENDPOINTS',pid=os.getpid(),completed=[],epochs=[3,5],GPU_wall_hours=0,source_sha256=expected)
    def save():
        status['observed_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();q=a.out/'STATUS.tmp';q.write_text(json.dumps(status,indent=2)+'\n');q.replace(a.out/'STATUS.json')
    try:
        for epoch in [3,5]:
            status.update(status='WAITING_FIXED_ENDPOINT',epoch=epoch);save();source=a.runtime/'phase2_fixed_native_replays_v1'/f'epoch{epoch}';start=time.monotonic()
            while not (source/'RESULTS.json').exists():
                s=json.loads((a.runtime/'phase2_fixed_native_replays_v1/STATUS.json').read_text())
                if s['status']=='FAILED_RETAIN_EVIDENCE':raise RuntimeError('Main fixed replay failed')
                if time.monotonic()-start>8*3600:raise RuntimeError('Wait timeout; inspect actual fitting, do not restart')
                time.sleep(3)
            if sha(code)!=expected:raise ValueError('Gate audit changed between endpoints')
            pins=dict(phase2_result=dict(pins_path=str(source/'INPUT_PINS.json'),pins_sha256=sha(source/'INPUT_PINS.json'),query_csv=str(source/'QUERY_RESULTS_OFFLINE_ONLY.csv'),query_csv_sha256=sha(source/'QUERY_RESULTS_OFFLINE_ONLY.csv')),
                diagnostic_registration='UNCHANGED_GATE_DIAGNOSTIC_LOCKED_BEFORE_FIXED_EPOCH3_5_RESULTS; no tuning or new training',source_sha256=expected,
                created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            path=a.out/f'epoch{epoch}_PINS.json';path.write_text(json.dumps(pins,indent=2)+'\n');status.update(status='RUNNING_CPU_DIAGNOSTIC',epoch=epoch);save();execute(path,a.out/f'epoch{epoch}');status['completed'].append(epoch);save()
        status.update(status='COMPLETE_FIXED_DIAGNOSTICS',scientific_success=False);save()
    except BaseException as e:
        status.update(status='FAILED_RETAIN_EVIDENCE',reason=repr(e));save();raise

if __name__=='__main__':main()
