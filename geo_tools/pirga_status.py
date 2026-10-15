"""Read-only server progress; waiting evaluators are distinct from fit failures."""
import argparse,datetime,json,shutil,time
from pathlib import Path


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def processes(runtime):
    found={}
    names=['train_pirga_grouped_phase1.py','train_pirga_innovation.py','train_pirga_decision.py','pirga_native_context.py','run_pirga_phase1_eval_queue.py','run_pirga_phase2_eval_queue.py','run_pirga_phase1_budget_correction.py','train_pirga_exhaustive_margin.py','run_pirga_decision_eval_queue.py','run_pirga_exhaustive_margin_eval_queue.py','run_pirga_uncertainty_gate_queue.py','run_pirga_uncertainty_gate_audit.py']
    names += ['train_pirga_legacy_coverage.py','run_pirga_legacy_coverage_eval_queue.py']
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            parts=(proc/'cmdline').read_bytes().split(b'\0');args=[x.decode(errors='replace') for x in parts if x]
            if not args or 'python' not in Path(args[0]).name or not any(str(runtime) in x for x in args):continue
            for name in names:
                if any(Path(x).name==name for x in args[1:]):found.setdefault(name,[]).append(int(proc.name))
        except (OSError,ProcessLookupError):continue
    return found


def main():
    p=argparse.ArgumentParser();p.add_argument('--runtime',type=Path,default=Path('/data1/liuyeqiang/WWW_pirga_runtime_20261016'));a=p.parse_args();root=a.runtime
    statusfiles=dict(phase1_training='phase1_grouped_matched_v1/fit/STATUS.json',phase1_resources='phase1_grouped_matched_v1/RESOURCE_LEDGER.json',
        phase1_corrective_resources='phase1_grouped_matched_v1/CORRECTIVE_RESOURCE_LEDGER.json',phase1_replay='phase1_fixed_native_replays_v1/STATUS.json',
        phase2_training='phase2_innovations_matched_v1/STATUS.json',phase2_replay='phase2_fixed_native_replays_v1/STATUS.json',ordinary_decision_training='ordinary_decision_matched_v1/STATUS.json',ordinary_replay='ordinary_fixed_native_replays_v1/STATUS.json',exhaustive_margin_training='exhaustive_margin_matched_v1/STATUS.json',exhaustive_margin_replay='exhaustive_margin_native_replays_v1/STATUS.json',uncertainty_gate_replay='uncertainty_gate_fixed_audits_v1/STATUS.json')
    statusfiles.update(legacy_coverage_training='legacy_canonical_coverage_fits_v1/STATUS.json',legacy_coverage_replay='legacy_canonical_coverage_replays_v1/STATUS.json')
    result=dict(observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),own_processes=processes(root),target_free_GiB=shutil.disk_usage(root).free/2**30,statuses={})
    for name,relative in statusfiles.items():
        path=root/relative;value=read(path)
        if value is not None:
            fields=['status','device','pid','CPU_pid','epoch','step','epoch_visit','endpoint','current_epoch','completed','GPU_wall_hours','CPU_wall_seconds','GPU_cap','reason']
            result['statuses'][name]={k:value[k] for k in fields if k in value};result['statuses'][name]['status_file_age_seconds']=time.time()-path.stat().st_mtime
    correction=read(root/'PHASE1_CORRECTIVE_RESOURCE_ALLOCATION.json')
    result['effective_phase1_GPU_residency_cap_hours']=correction['GPU_residency_maximum_hours'] if correction else None
    result['original_GPU_cap_is_historical_if_superseded']=bool(correction)
    result['joint_context_preparation_complete']=(root/'native_decision_context_v1/MANIFEST.json').exists()
    result['fixed_reliability_complete']=(root/'fixed_reliability_replay_v1/RESULTS.json').exists()
    result['qualification']='A live PID alone does not prove useful progress. Compare actual step/visit over time; waiting fixed-epoch evaluators are expected. See logs for lengthy admission/context construction. No online performance claim.'
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
