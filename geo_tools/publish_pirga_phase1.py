"""Publish completed endpoints without replacing raw evidence or choosing a best."""
import csv,datetime,json,shutil
from collections import Counter
from pathlib import Path
from pirga_indexed_store import sha
from audit_cv_pil_v3_outcomes import read_rows


def main():
    repo=Path(__file__).resolve().parents[1];runtime=Path('/data1/liuyeqiang/WWW_pirga_runtime_20261016');r=repo/'results/PIRGA_FULLTRAIN';fit=runtime/'phase1_grouped_matched_v1/fit';evaluation=runtime/'phase1_fixed_native_replays_v1'
    summary=[]
    for epoch in [1,3,5]:
        source=evaluation/f'epoch{epoch}'
        if not (source/'RESULTS.json').exists():continue
        destination=r/'PHASE1_REPLAY'/f'epoch{epoch}';destination.mkdir(parents=True,exist_ok=True)
        for name in ['RESULTS.json','INPUT_PINS.json','ORIGINAL_ERROR_MECHANISM.json','QUERY_RESULTS_OFFLINE_ONLY.csv']:
            target=destination/name
            if target.exists() and sha(target)!=sha(source/name):raise ValueError('Refuse to replace published endpoint')
            if not target.exists():shutil.copyfile(source/name,target)
        result=json.loads((source/'RESULTS.json').read_text());rows=read_rows(source/'QUERY_RESULTS_OFFLINE_ONLY.csv');mechanism=result['original_error_mechanism'];comparison=mechanism['comparisons']['C3_vs_C1']
        transitions={}
        for before in ('C1','C2','C4'):
            counts=Counter(str(v[before+'_primary'])+' -> '+str(v['C3_primary']) for v in rows)
            transitions['C3_vs_'+before]=dict(counts)
        adjudication=dict(status='NO_PREDICTION_SPECIFIC_GAIN' if comparison['net_original_error_repairs']<=0 else 'PHASE1_PREDICTIVE_SIGNAL_REQUIRES_PHASE2_STRONG_CONTROLS',
            epoch=epoch,comparison_to_C1= comparison,certification_transitions=transitions,
            old_helper_gate_is_not_PIRGA_admission=True,original_helper_label_preserved=mechanism['status'],
            no_online_admission=True,scientific_success=False,
            qualification='One exposed DEV scene. Raw inherited helper can call C3 vs C2/C4 positive; PIRGA additionally requires extra net native repairs over C1 and strong ordinary models. Never interpret inherited helper label as PIRGA success.')
        (destination/'PIRGA_ADJUDICATION.json').write_text(json.dumps(adjudication,indent=2)+'\n')
        for arm,metrics in result['methods'].items():
            m=mechanism['methods'][arm];summary.append(dict(epoch=epoch,method=arm,top1=metrics['top1'],mrr=metrics['mrr'],
                repaired_original_primary_errors=m['certified_reachable_original_primary_errors_repaired'],known_harms_to_original_correct=m['originally_correct_primary_known_harms'],
                original_correct_to_unknown=m['originally_correct_to_unknown'],repair_person_groups=m['repaired_video_person_groups'],
                all_primary_unknown=metrics['primary_unknown'],all_primary_known_wrong=metrics['primary_known_wrong'],
                stage_label='SAME_STATE_NATIVE_REPLAY_ONLY',scientific_success=False))
    if summary:
        with (r/'PHASE1_ENDPOINT_SUMMARY.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(summary[0]),lineterminator='\n');w.writeheader();w.writerows(summary)
    for name in ('TRAINING_EPOCH_COVERAGE.json','CHECKPOINTS.json'):
        if (fit/name).exists():shutil.copyfile(fit/name,r/('PHASE1_'+name))
    if (fit/'TRAINING_EPOCH_COVERAGE.json').exists():shutil.copyfile(fit/'TRAINING_EPOCH_COVERAGE.json',r/'TRAINING_EPOCH_COVERAGE.json')
    raw=(fit/'TRAINING_CURVES.csv').read_text();(r/'TRAINING_CURVES.csv').write_text(raw[:raw.rfind('\n')+1])
    now=datetime.datetime.now(datetime.timezone.utc).isoformat();live=json.loads((fit.parent/'RESOURCE_LEDGER.json').read_text());status=json.loads((fit/'STATUS.json').read_text())
    corrective=json.loads((fit.parent/'CORRECTIVE_RESOURCE_LEDGER.json').read_text()) if (fit.parent/'CORRECTIVE_RESOURCE_LEDGER.json').exists() else None
    (r/'PHASE1_TRAINING_STATUS_SNAPSHOT.json').write_text(json.dumps(dict(observed_utc=now,resource=live,corrective_resource=corrective,training=status,snapshot_not_live_monitor=True),indent=2)+'\n')
    master=json.loads((r/'MASTER_GOAL.json').read_text());master['phases']['1']='COMPLETE_TRAINING_PENDING_ALL_FIXED_REPLAYS' if status['status']=='COMPLETE' else 'RUNNING_MATCHED_FULL_EPOCH_C2_C3_C4';master['updated_utc']=now
    master['completed_Phase1_replay_epochs']=sorted({v['epoch'] for v in summary});master['latest_fixed_epoch_PIRGA_status']=None if not summary else json.loads((r/'PHASE1_REPLAY'/f'epoch{summary[-1]["epoch"]}'/'PIRGA_ADJUDICATION.json').read_text())['status']
    (r/'MASTER_GOAL.json').write_text(json.dumps(master,indent=2)+'\n')
    resource=json.loads((r/'RESOURCE_LEDGER.json').read_text());charged=10*64/3600
    resource.update(status='PHASE1_RUNNING_CONSERVATIVE_CUDA_CONTEXT_CORRECTION',updated_utc=now,
        gpu_wall_hours=1.3086310044572587+live['GPU_wall_hours']+charged,
        incidental_context_conservative_GPU_hours=charged,incidental_context_is_upper_bound_not_measured_utilization=True,
        phase1_fit_observed_GPU_wall_hours=live['GPU_wall_hours'],phase1_fit_GPU_cap=1.32,
        parent_aggregate_observed_plus_conservative_bound_GPU_hours=1.1772130608724223+1.3086310044572587+live['GPU_wall_hours']+charged,
        remaining_existing_parent_hours=4-1.1772130608724223-1.3086310044572587-live['GPU_wall_hours']-charged,
        current_free_target_bytes=shutil.disk_usage('/data1').free,corrective_allocation='PHASE1_CORRECTIVE_RESOURCE_ALLOCATION.json',resource_snapshot_is_not_terminal=status['status']!='COMPLETE')
    (r/'RESOURCE_LEDGER.json').write_text(json.dumps(resource,indent=2)+'\n')
    print(json.dumps(dict(published_epochs=master['completed_Phase1_replay_epochs'],training_status=status,latest_adjudication=master['latest_fixed_epoch_PIRGA_status'])))

if __name__=='__main__':main()
