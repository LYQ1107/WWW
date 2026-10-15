"""All fixed Phase2 endpoints plus their same-epoch Phase1 references."""
import argparse,datetime,json,os,time
from pathlib import Path
from pirga_indexed_store import sha
from run_pirga_phase2_replay import execute


def main():
    p=argparse.ArgumentParser();p.add_argument('--runtime',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    protocol=a.runtime/'PHASE2_MATCHED_PROTOCOL.json';expected='17ffe01425aa3da9e321c2b96d55eb30d82dccecfebdfadbff724f4563a552c6'
    if sha(protocol)!=expected:raise ValueError('Wrong preregistered protocol')
    repo=Path(__file__).resolve().parents[1];release=json.loads((repo/'results/cv_pil_v3_native_validation/P1_SOURCE_PINS.json').read_text())['release'];fit=a.runtime/'phase2_innovations_matched_v1'
    a.out.mkdir(parents=True,exist_ok=False)
    source_names=['run_pirga_phase2_eval_queue.py','run_pirga_phase2_replay.py','pirga_replay_reporting.py','run_pirga_phase1_replay.py','cv_pil_v3_call_scoring.py','cv_pil_dev12_shards.py']
    source_pins={name:sha(repo/'geo_tools'/name) for name in source_names}
    status=dict(status='WAITING_FIXED_EPOCH',pid=os.getpid(),completed=[],GPU_wall_hours=0,source_pins=source_pins)
    def save():
        status['observed_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();q=a.out/'STATUS.tmp';q.write_text(json.dumps(status,indent=2)+'\n');q.replace(a.out/'STATUS.json')
    try:
        for epoch in [1,3,5]:
            status.update(status='WAITING_FIXED_EPOCH',epoch=epoch);save();start=time.monotonic()
            while True:
                index=fit/'CHECKPOINTS.json';all_heads=[] if not index.exists() else json.loads(index.read_text())['checkpoints']
                heads={v['arm']:{k:v[k] for k in ('path','sha256')} for v in all_heads if v['epoch']==epoch}
                prior=a.runtime/'phase1_fixed_native_replays_v1'/f'epoch{epoch}'
                if len(heads)==7 and (prior/'RESULTS.json').exists():break
                s=json.loads((fit/'STATUS.json').read_text())
                if s['status'] in ('FAILED_RETAIN_EVIDENCE','RESOURCE_STOP_WITH_EXACT_CHECKPOINT_NOT_COMPLETE'):raise RuntimeError('Phase2 training stopped before endpoint')
                if time.monotonic()-start>8*3600:raise RuntimeError('Fixed endpoint wait timeout; inspect actual process')
                time.sleep(3)
            for name,digest in source_pins.items():
                if sha(repo/'geo_tools'/name)!=digest:raise ValueError('Replay source changed between endpoints')
            pins=dict(epoch=epoch,protocol=dict(path=str(protocol),sha256=expected),heads=heads,release=release,
                phase1_pins=dict(path=str(prior/'INPUT_PINS.json'),sha256=sha(prior/'INPUT_PINS.json')),
                phase1_reference_csv=dict(path=str(prior/'QUERY_RESULTS_OFFLINE_ONLY.csv'),sha256=sha(prior/'QUERY_RESULTS_OFFLINE_ONLY.csv')),replay_source_sha256=source_pins)
            path=a.out/f'epoch{epoch}_PINS.json';path.write_text(json.dumps(pins,indent=2)+'\n')
            status.update(status='RUNNING_CPU_REPLAY',epoch=epoch,pins_sha256=sha(path));save();execute(pins,a.out/f'epoch{epoch}')
            status['completed'].append(epoch);save()
        status.update(status='COMPLETE_ALL_FIXED_ENDPOINTS',scientific_success=False);save()
    except BaseException as e:
        status.update(status='FAILED_RETAIN_EVIDENCE',reason=repr(e));save();raise

if __name__=='__main__':main()
