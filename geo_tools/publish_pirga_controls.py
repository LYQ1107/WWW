"""Publish completed fixed control endpoints; preserve raw artifacts immutably."""
import csv,json,shutil,datetime
from pathlib import Path
from pirga_indexed_store import sha
from pirga_replay_reporting import mechanism_audit


def rows(path):
    with Path(path).open() as f:raw=list(csv.DictReader(f))
    result={}
    for r in raw:
        r['key']=json.loads(r['key']);r['person']=int(r['person']) if r['person'] else None
        for name,value in list(r.items()):
            if name.endswith(('_primary','_rank')) or name=='positive_covered':r[name]={'True':True,'False':False,'':None}[value]
            elif name.endswith(('_accepted_id','_rank_id')):r[name]=int(value) if value else None
            elif name.endswith('_rr'):r[name]=float(value) if value else None
        key=tuple(r['key'])
        if key in result:raise ValueError('Duplicate physical query')
        result[key]=r
    return result


def merge(base,other):
    if set(base)!=set(other):raise ValueError('Different query scopes cannot be combined')
    for key,target in base.items():
        for name,value in other[key].items():
            if name in target and target[name]!=value:raise ValueError('Shared control or supervision differs '+str((key,name)))
            target[name]=value


def main():
    repo=Path(__file__).resolve().parents[1];runtime=Path('/data1/liuyeqiang/WWW_pirga_runtime_20261016');root=repo/'results/PIRGA_FULLTRAIN';now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    families={'PHASE2':('phase2_fixed_native_replays_v1','PREDICTION_ABLATIONS.csv'),'ORDINARY_DECISION':('ordinary_fixed_native_replays_v1','NATIVE_DECISION_ABLATIONS.csv'),'EXHAUSTIVE_MARGIN':('exhaustive_margin_native_replays_v1','EXHAUSTIVE_MARGIN_ABLATIONS.csv')};completed={};group_rows=[]
    for family,(local,table) in families.items():
        summary=[];completed[family]=[]
        for epoch in [1,3,5]:
            source=runtime/local/f'epoch{epoch}'
            if not (source/'RESULTS.json').exists():continue
            dest=root/family/f'epoch{epoch}';dest.mkdir(parents=True,exist_ok=True)
            for name in ['RESULTS.json','INPUT_PINS.json','QUERY_RESULTS_OFFLINE_ONLY.csv']:
                target=dest/name
                if target.exists() and sha(target)!=sha(source/name):raise ValueError('Refuse to replace published control endpoint')
                if not target.exists():shutil.copyfile(source/name,target)
            result=json.loads((source/'RESULTS.json').read_text());completed[family].append(epoch)
            for arm,m in result['methods'].items():
                native=result['mechanism']['methods'][arm]
                summary.append(dict(family=family,epoch=epoch,training_queries_per_epoch=113453 if family=='EXHAUSTIVE_MARGIN' else 113415,matched_Phase2_update_budget=family!='EXHAUSTIVE_MARGIN',method=arm,top1=m['top1'],mrr=m['mrr'],original_error_repairs=native['original_error_repairs'],
                    harms_to_original_correct=native['known_harms_to_original_correct'],original_correct_to_unknown=native['original_correct_to_unknown'],
                    original_unknown_to_known_wrong=native['original_unknown_to_known_wrong'],repair_person_groups=native['repair_person_scene_groups'],
                    primary_unknown=m['primary_unknown'],stage='SAME_STATE_NATIVE_REPLAY_ONLY',scientific_success=False))
            for group,methods in result['by_person_scene'].items():
                for arm,m in methods.items():group_rows.append(dict(family=family,epoch=epoch,person_scene_group=group,method=arm,**m))
        if summary:
            with (root/table).open('w') as f:
                writer=csv.DictWriter(f,fieldnames=list(summary[0]),lineterminator='\n');writer.writeheader();writer.writerows(summary)
    for epoch in set(completed['PHASE2'])&set(completed['ORDINARY_DECISION']):
        combined=rows(runtime/'phase2_fixed_native_replays_v1'/f'epoch{epoch}'/'QUERY_RESULTS_OFFLINE_ONLY.csv')
        merge(combined,rows(runtime/'ordinary_fixed_native_replays_v1'/f'epoch{epoch}'/'QUERY_RESULTS_OFFLINE_ONLY.csv'))
        merge(combined,rows(runtime/'fixed_reliability_replay_v1/QUERY_RESULTS_OFFLINE_ONLY.csv'))
        arms=[name[:-len('_primary')] for name in next(iter(combined.values())) if name.endswith('_primary')]
        controls=('C1','A2_rank','AC_contrastive','C3','D1_MLP','D2_MLP_native_margin','R_mean','R_top4_recent','R_coordinate_median','R_trim10','R_latest','R_decay')
        gate=mechanism_audit(list(combined.values()),arms,controls=controls);gate.update(epoch=epoch,
            qualification='All fixed families on the same complete original-state DEV12 and same fitted epoch; no historical different sequence pooling. Simple-history controls are stronger unbounded aggregate replacements, separately qualified from bounded matched learned-head comparison.',
            phase2_result_sha256=sha(runtime/'phase2_fixed_native_replays_v1'/f'epoch{epoch}'/'RESULTS.json'),ordinary_result_sha256=sha(runtime/'ordinary_fixed_native_replays_v1'/f'epoch{epoch}'/'RESULTS.json'))
        gate['strong_ordinary_controls_complete_for_this_epoch']=True
        gate['online_gate_pending']='Native causal integration and online evaluation remain gated; strong ordinary controls are included in this same-epoch decision.'
        if gate['phase2_mechanism_gate_pass']:gate['status']='NATIVE_REPLAY_POSITIVE_PENDING_NATIVE_INTEGRATION'
        (root/f'MATCHED_ALL_CONTROL_GATE_EPOCH{epoch}.json').write_text(json.dumps(gate,indent=2)+'\n')
    if group_rows:
        with (root/'PER_PERSON_AND_SCENE_RESULTS.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(group_rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(group_rows)
    for family,local in [('PHASE2','phase2_innovations_matched_v1'),('ORDINARY_DECISION','ordinary_decision_matched_v1'),('EXHAUSTIVE_MARGIN','exhaustive_margin_matched_v1')]:
        fit=runtime/local
        if (fit/'STATUS.json').exists():(root/(family+'_TRAINING_STATUS_SNAPSHOT.json')).write_text(json.dumps(dict(observed_utc=now,training=json.loads((fit/'STATUS.json').read_text()),snapshot_not_live_monitor=True),indent=2)+'\n')
        for name in ['CHECKPOINTS.json','TRAINING_EPOCH_COVERAGE.json','RESULTS.json','TRAINING_CURVES.csv']:
            if (fit/name).exists():
                if name.endswith('.csv'):
                    content=(fit/name).read_text();(root/(family+'_'+name)).write_text(content[:content.rfind('\n')+1])
                else:shutil.copyfile(fit/name,root/(family+'_'+name))
    diagnostic=runtime/'bounded_action_diagnostic_v1'
    if (diagnostic/'RESULTS.json').exists():
        for name in ['RESULTS.json','INPUT_PINS.json']:shutil.copyfile(diagnostic/name,root/('BOUNDED_ACTION_DIAGNOSTIC_'+name))
    uncertainty_epochs=[]
    for epoch in [1,3,5]:
        source=runtime/('uncertainty_gate_audit_epoch1_v1' if epoch==1 else f'uncertainty_gate_fixed_audits_v1/epoch{epoch}')
        if not (source/'INPUT_PINS.json').exists() or not (source/'RESULTS.json').exists():continue
        result=json.loads((source/'RESULTS.json').read_text())
        if result['epoch']!=epoch or result['queries']!=14557 or not result['weights_unchanged'] or result['training_updates']!=0 or not result['original_ON_actions_exact_full_scope']:raise ValueError('Uncertainty audit scope/checkpoint drift')
        dest=root/'UNCERTAINTY_GATE'/f'epoch{epoch}';dest.mkdir(parents=True,exist_ok=True)
        for name in ['RESULTS.json','INPUT_PINS.json']:
            target=dest/name
            if target.exists() and sha(target)!=sha(source/name):raise ValueError('Refuse to replace published gate diagnostic')
            if not target.exists():shutil.copyfile(source/name,target)
        uncertainty_epochs.append(epoch)
    master=json.loads((root/'MASTER_GOAL.json').read_text());master['completed_new_control_replay_epochs']=completed;master['updated_utc']=now
    master['completed_uncertainty_gate_diagnostic_epochs']=uncertainty_epochs;(root/'MASTER_GOAL.json').write_text(json.dumps(master,indent=2)+'\n')
    print(json.dumps(dict(completed_control_replay_epochs=completed,observed_utc=now)))

if __name__=='__main__':main()
