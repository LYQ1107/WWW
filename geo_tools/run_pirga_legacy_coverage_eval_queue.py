"""Evaluate all paired-control endpoints without waiting on unrelated fits."""
import argparse,json,datetime,time,os
from pathlib import Path
from pirga_indexed_store import sha
from run_pirga_legacy_coverage_replay import execute


def main():
    p=argparse.ArgumentParser();p.add_argument('--runtime',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();repo=Path(__file__).resolve().parents[1]
    protocol=a.runtime/'LEGACY_CANONICAL_COVERAGE_PROTOCOL.json';expected='227456e663624c7170bc0464e0fe99bb0ec9aa025b377471f43aad976fbe7638'
    if sha(protocol)!=expected:raise ValueError('Unregistered coverage-control protocol')
    release=json.loads((repo/'results/cv_pil_v3_native_validation/P1_SOURCE_PINS.json').read_text())['release'];reference=repo/'results/cv_pil_v3_native_validation/P3_MATCHED300_FULL_DEV12/QUERY_RESULTS_OFFLINE_ONLY.csv';reference_sha='3db925d0e10d866dd2d761f061c01a8e25d0747ee21b47d95e6d9aca01a31926'
    if sha(reference)!=reference_sha:raise ValueError('Same-range reference changed')
    code={n:sha(repo/'geo_tools'/n) for n in ['run_pirga_legacy_coverage_eval_queue.py','run_pirga_legacy_coverage_replay.py','cv_pil_v3_call_scoring.py','run_pirga_phase1_replay.py','pirga_replay_reporting.py']}
    a.out.mkdir(parents=True,exist_ok=False);fit=a.runtime/'legacy_canonical_coverage_fits_v1';status=dict(status='WAITING_CONTROL',pid=os.getpid(),completed=[],source_sha256=code,GPU_wall_hours=0)
    def save():
        status['observed_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();q=a.out/'STATUS.tmp';q.write_text(json.dumps(status,indent=2)+'\n');q.replace(a.out/'STATUS.json')
    try:
        for mode,epoch in [('CANONICAL_MASKED300',None),('LEGACY_FULL_CANONICAL',1),('LEGACY_FULL_CANONICAL',3),('LEGACY_FULL_CANONICAL',5)]:
            name='canonical300' if epoch is None else f'epoch{epoch}';status.update(status='WAITING_CONTROL',endpoint=name);save();start=time.monotonic()
            while True:
                path=fit/('canonical300' if epoch is None else 'full')/'CHECKPOINTS.json';records=[] if not path.exists() else json.loads(path.read_text())['checkpoints']
                heads={v['arm']:{k:v[k] for k in ['path','sha256']} for v in records if epoch is None or v['epoch']==epoch}
                if set(heads)=={'C2','C3','C4'}:break
                s=json.loads((fit/'STATUS.json').read_text())
                if s['status'] in ['FAILED_RETAIN_EVIDENCE','RESOURCE_STOP_NOT_COMPLETE']:raise RuntimeError('Control fit stopped before registered endpoint')
                if time.monotonic()-start>6*3600:raise RuntimeError('Control wait timed out; inspect actual process')
                time.sleep(3)
            for n,d in code.items():
                if sha(repo/'geo_tools'/n)!=d:raise ValueError('Frozen coverage scorer changed')
            pins=dict(mode=mode,epoch=epoch,heads=heads,protocol=dict(path=str(protocol),sha256=expected),release=release,reference_csv=dict(path=str(reference),sha256=reference_sha),scorer_source_sha256=code)
            pin=a.out/(name+'_PINS.json');pin.write_text(json.dumps(pins,indent=2)+'\n');status.update(status='RUNNING_FULL_NATIVE_REPLAY',endpoint=name);save();execute(pins,a.out/name);status['completed'].append(name);save()
        status.update(status='COMPLETE_ALL_PAIRED_CONTROLS',scientific_success=False);save()
    except BaseException as e:
        status.update(status='FAILED_RETAIN_EVIDENCE',reason=repr(e));save();raise

if __name__=='__main__':main()
