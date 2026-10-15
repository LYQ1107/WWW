"""Evaluate every preregistered epoch, never choose an endpoint using DEV."""
import argparse,json,os,signal,subprocess,time,datetime
from pathlib import Path
from pirga_indexed_store import sha
from run_pirga_phase1_replay import execute


def main():
    p=argparse.ArgumentParser();p.add_argument('--training-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    repo=Path(__file__).resolve().parents[1];runtime=a.training_root.parent.parent;protocol=runtime/'PHASE1_MATCHED_PROTOCOL.json'
    expected='b9667b304852a6e4135f59104d618b2ed6108111ede3a26cd54dd9dc1d93d619'
    if sha(protocol)!=expected:raise ValueError('Wrong frozen training protocol')
    release=json.loads((repo/'results/cv_pil_v3_native_validation/P1_SOURCE_PINS.json').read_text())['release']
    reference=repo/'results/cv_pil_v3_native_validation/P3_MATCHED300_FULL_DEV12/QUERY_RESULTS_OFFLINE_ONLY.csv'
    if sha(reference)!='3db925d0e10d866dd2d761f061c01a8e25d0747ee21b47d95e6d9aca01a31926':raise ValueError('Frozen controls changed')
    a.out.mkdir(parents=True,exist_ok=False)
    status=dict(status='WAITING_REGISTERED_EPOCH',epochs=[1,3,5],completed=[],pid=os.getpid(),GPU_wall_hours=0,no_selection=True)
    def save():
        status['observed_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        q=a.out/'STATUS.tmp';q.write_text(json.dumps(status,indent=2)+'\n');q.replace(a.out/'STATUS.json')
    try:
        for epoch in [1,3,5]:
            status.update(status='WAITING_REGISTERED_EPOCH',current_epoch=epoch);save();start=time.monotonic()
            while True:
                index=a.training_root/'CHECKPOINTS.json'
                candidates=[] if not index.exists() else json.loads(index.read_text())['checkpoints']
                heads={v['arm']:{k:v[k] for k in ('path','sha256')} for v in candidates if v['kind']==f'epoch{epoch}'}
                if set(heads)=={'C2','C3','C4'}:break
                current=json.loads((a.training_root/'STATUS.json').read_text())
                if current['status']=='FAILED_RETAIN_EVIDENCE':raise RuntimeError('Training failed before fixed endpoint')
                if time.monotonic()-start>8*3600:raise RuntimeError('Registered epoch wait timeout; inspect real training, do not restart automatically')
                time.sleep(3)
            pins=dict(epoch=epoch,training_protocol=dict(path=str(protocol),sha256=expected),heads=heads,release=release,
                frozen_control_csv=dict(path=str(reference),sha256=sha(reference)))
            pinpath=a.out/f'epoch{epoch}_PINS.json';pinpath.write_text(json.dumps(pins,indent=2)+'\n')
            status.update(status='RUNNING_CPU_REPLAY',current_epoch=epoch,pins_sha256=sha(pinpath));save()
            execute(pins,a.out/f'epoch{epoch}')
            status['completed'].append(epoch);save()
        status.update(status='COMPLETE_ALL_FIXED_EPOCH_REPLAYS',scientific_success=False);save()
    except BaseException as e:
        status.update(status='FAILED_RETAIN_EVIDENCE',reason=repr(e));save();raise

if __name__=='__main__':main()
