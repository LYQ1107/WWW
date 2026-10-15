"""Publish all fixed historical-interface controls, keeping scope confounds explicit."""
import csv,datetime,json,shutil
from pathlib import Path
from pirga_indexed_store import sha
from pirga_replay_reporting import mechanism_audit
from publish_pirga_controls import rows,merge


def project(source,prefix):
    """Retain shared scope/labels; namespace every learned arm across fit families."""
    output={}
    for key,row in rows(source).items():
        output[key]={}
        for name,value in row.items():
            if name in ('key','person','positive_covered','source_primary_state') or name.startswith(('C0_','C1_')):
                output[key][name]=value
            elif name.startswith(('C2_','C3_','C4_')):
                output[key][prefix+'_'+name]=value
    return output


def main():
    repo=Path(__file__).resolve().parents[1];runtime=Path('/data1/liuyeqiang/WWW_pirga_runtime_20261016')
    root=repo/'results/PIRGA_FULLTRAIN';fit=runtime/'legacy_canonical_coverage_fits_v1';evaluation=runtime/'legacy_canonical_coverage_replays_v1'
    summaries=[];completed=[]
    for endpoint in ['canonical300','epoch1','epoch3','epoch5']:
        source=evaluation/endpoint
        if not (source/'RESULTS.json').exists():continue
        result=json.loads((source/'RESULTS.json').read_text());dest=root/'LEGACY_COVERAGE'/endpoint;dest.mkdir(parents=True,exist_ok=True)
        for name in ['RESULTS.json','INPUT_PINS.json','QUERY_RESULTS_OFFLINE_ONLY.csv']:
            target=dest/name
            if target.exists() and sha(target)!=sha(source/name):raise ValueError('Published control changed')
            if not target.exists():shutil.copyfile(source/name,target)
        completed.append(endpoint)
        for arm,metrics in result['methods'].items():
            native=result['mechanism']['methods'][arm]
            summaries.append(dict(endpoint=endpoint,family=result['family'],method=arm,step=result['step'],
                top1=metrics['top1'],mrr=metrics['mrr'],repairs=native['original_error_repairs'],
                harms=native['known_harms_to_original_correct'],correct_to_unknown=native['original_correct_to_unknown'],
                original_unknown_to_wrong=native['original_unknown_to_known_wrong'],repair_person_groups=native['repair_person_scene_groups'],
                primary_unknown=metrics['primary_unknown'],stage='SAME_STATE_NATIVE_REPLAY_ONLY',scientific_success=False))
    old=repo/'results/cv_pil_v3_native_validation/P3_MATCHED300_FULL_DEV12/QUERY_RESULTS_OFFLINE_ONLY.csv'
    canonical=evaluation/'canonical300/QUERY_RESULTS_OFFLINE_ONLY.csv'
    if canonical.exists():
        paired=project(canonical,'S1');merge(paired,project(old,'S0'))
        arms=['C0','C1']+[prefix+'_'+arm for prefix in ['S0','S1'] for arm in ['C2','C3','C4']]
        impact=mechanism_audit(list(paired.values()),arms,forecast='S1_C3',controls=('C1','S0_C3','S1_C2','S1_C4'))
        impact.update(control='SAME_OLD300_SCHEDULE_CANONICAL_MASK_VS_FROZEN_HISTORICAL_MASK',
            primary_action_changes_by_arm={arm:sum(row['S0_'+arm+'_accepted_id']!=row['S1_'+arm+'_accepted_id'] for row in paired.values()) for arm in ['C2','C3','C4']},
            original_csv_sha256=sha(old),canonical_csv_sha256=sha(canonical),online_admitted=False,scientific_success=False,
            qualification='Complete identical exposedDEV12; canonical mask changes onlyTRAIN supervision. Action equivalence does not mean equal learned tensors or prove masks irrelevant under full training. Historical original outputs remain immutable.')
        (root/'LEGACY_CANONICAL300_MASK_IMPACT.json').write_text(json.dumps(impact,indent=2)+'\n')
    for epoch in [1,3,5]:
        full=evaluation/f'epoch{epoch}/QUERY_RESULTS_OFFLINE_ONLY.csv'
        if not canonical.exists() or not full.exists():continue
        combined=project(full,'L1');merge(combined,project(canonical,'S1'));merge(combined,project(old,'S0'))
        native=runtime/'phase1_fixed_native_replays_v1'/f'epoch{epoch}/QUERY_RESULTS_OFFLINE_ONLY.csv'
        if native.exists():merge(combined,project(native,'L2'))
        arms=['C0','C1']+[name[:-8] for name in next(iter(combined.values())) if name.endswith('_primary') and not name.startswith(('C0_','C1_'))]
        audit=mechanism_audit(list(combined.values()),arms,forecast='L1_C3',controls=('C1','L1_C2','L1_C4','S1_C3','S0_C3'))
        audit.update(epoch=epoch,scientific_success=False,online_admitted=False,
            families=dict(S0='Old frozen300, historical negative masks',S1='Same old300 schedule refit under conservative canonical masks',
                L1='Legacy Top8/latest4 preprocessing, full106800-query epoch under canonical masks',L2='Full native candidates/history,113415-query common epoch'),
            causal_interpretation='S0 vs S1: canonical supervision policy change. S1 vs L1: preregistered coverage/order/epoch expansion under the same input and mask policies. L1 vs L2: training evidence/eligibility/update-count changes; not pure model effect.',
            qualifications=['All use complete identical exposed DEV12 native matrices at evaluation; no online mutated states.',
                'Never select the best epoch; all1/3/5 published. Single seed and exposed single scene do not establish generalization.',
                'Different update counts across S0/S1,L1,L2 are deliberate coverage controls, not equal-budget claims.'],
            input_csv_sha256={str(p):sha(p) for p in [old,canonical,full]+([native] if native.exists() else [])})
        (root/f'LEGACY_COVERAGE_COMPARISON_EPOCH{epoch}.json').write_text(json.dumps(audit,indent=2)+'\n')
    if summaries:
        with (root/'LEGACY_COVERAGE_ABLATIONS.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(summaries[0]),lineterminator='\n');w.writeheader();w.writerows(summaries)
    for family in ['canonical300','full']:
        for name in ['CHECKPOINTS.json','RESULTS.json','TRAINING_EPOCH_COVERAGE.json','TRAINING_CURVES.csv']:
            source=fit/family/name
            if not source.exists():continue
            target=root/('LEGACY_'+family.upper()+'_'+name)
            if name.endswith('.csv'):
                value=source.read_text();target.write_text(value[:value.rfind('\n')+1])
            else:shutil.copyfile(source,target)
    now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    (root/'LEGACY_TRAINING_STATUS_SNAPSHOT.json').write_text(json.dumps(dict(observed_utc=now,training=json.loads((fit/'STATUS.json').read_text()),snapshot_not_live_monitor=True),indent=2)+'\n')
    master=json.loads((root/'MASTER_GOAL.json').read_text());master['completed_legacy_coverage_replay_endpoints']=completed;master['updated_utc']=now
    master['legacy_preprocessing_control']='FIXED_CANONICAL300_AND_FULL106800_EPOCHS1_3_5_PAIRED_CONTROL'
    (root/'MASTER_GOAL.json').write_text(json.dumps(master,indent=2)+'\n')
    print(json.dumps(dict(completed_legacy_coverage_replay_endpoints=completed,observed_utc=now)))


if __name__=='__main__':main()
