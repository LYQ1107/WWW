"""Refresh derived registries from real published endpoints, never synthesize metrics."""
import csv,datetime,json
from pathlib import Path
from pirga_indexed_store import sha


def main():
    repo=Path(__file__).resolve().parents[1];root=repo/'results/PIRGA_FULLTRAIN'
    runtime=Path('/data1/liuyeqiang/WWW_pirga_runtime_20261016');now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    entries=[]
    for family in ['PHASE1_REPLAY','PHASE2','ORDINARY_DECISION','EXHAUSTIVE_MARGIN','FIXED_RELIABILITY','LEGACY_COVERAGE']:
        for path in sorted((root/family).rglob('RESULTS.json')):
            result=json.loads(path.read_text())
            if result.get('status')!='SAME_STATE_NATIVE_REPLAY_ONLY':raise ValueError('Unexpected evidence level '+str(path))
            entries.append(dict(family=family,path=str(path.relative_to(repo)),sha256=sha(path),label='SAME_STATE_NATIVE_REPLAY_ONLY'))
    registry=dict(status='ACTUAL_COMPLETED_PRIMARY_REPLAY_EVIDENCE_REGISTRY',entries=entries,full_native_online_identity_repairs=None,
        scientific_success=False,updated_utc=now,qualification='Original primary association repairs only; no final closed-loop identity repair count. Physical query and scene-person grouping in each endpoint; never pool different fit families as independent successes.')
    (root/'ORIGINAL_ERROR_REPAIRS.json').write_text(json.dumps(registry,indent=2)+'\n')
    fits=dict(P1='phase1_grouped_matched_v1/fit',P2='phase2_innovations_matched_v1',ordinary='ordinary_decision_matched_v1',
        exhaustive_margin='exhaustive_margin_matched_v1',legacy_coverage='legacy_canonical_coverage_fits_v1')
    status={name:json.loads((runtime/path/'STATUS.json').read_text()) for name,path in fits.items()}
    speed=json.loads((root/'SPEED_AND_MEMORY.json').read_text());speed.update(training_observed_utc=now,
        training_processes={name:{k:s[k] for k in ['status','device','step','epoch','CPU_wall_seconds','peak_RSS_bytes'] if k in s} for name,s in status.items()})
    (root/'SPEED_AND_MEMORY.json').write_text(json.dumps(speed,indent=2)+'\n')
    resource=json.loads((root/'RESOURCE_LEDGER.json').read_text());resource.update(CPU_fit_status_snapshot=speed['training_processes'],CPU_fit_status_observed_utc=now,
        CPU_snapshot_qualification='Per-process training wall times overlap in real time. No conversion to GPU hours; no old ledger reset. P1 resumed wall field covers only suffix.')
    (root/'RESOURCE_LEDGER.json').write_text(json.dumps(resource,indent=2)+'\n')
    master=json.loads((root/'MASTER_GOAL.json').read_text());master['updated_utc']=now
    control=master.get('completed_new_control_replay_epochs',{})
    p1complete=master.get('completed_Phase1_replay_epochs')==[1,3,5] and status['P1']['status']=='COMPLETE'
    p2complete=control.get('PHASE2')==[1,3,5] and status['P2']['status']=='COMPLETE_FIXED5EPOCHS'
    if p1complete:master['phases']['1']='COMPLETE_FIXED1_3_5_TRAINING_AND_NATIVE_REPLAYS'
    if p2complete:master['phases']['2']='COMPLETE_FIXED1_3_5_TRAINING_AND_NATIVE_REPLAYS'
    master['phases']['4']='ORDINARY_CONTROLS_FIXED_FITS_COMPLETE_REPLAY_PENDING_PREDICTIVE_COMBINATIONS_CONDITIONAL' if status['ordinary']['status']=='COMPLETE_FIXED5EPOCHS' else 'ORDINARY_CONTROLS_TRAINING_CPU_PREDICTIVE_COMBINATIONS_CONDITIONAL'
    if all(control.get(f)==[1,3,5] for f in ['ORDINARY_DECISION','EXHAUSTIVE_MARGIN']):
        master['phases']['4']='ORDINARY_FIXED1_3_5_REPLAYS_COMPLETE_PREDICTIVE_COMBINATIONS_CONDITIONAL'
    master['supplementary_exhaustive_margin']='FIXED5EPOCH_FIT_COMPLETE_PENDING_ALL_REPLAYS' if status['exhaustive_margin']['status']=='COMPLETE_FIXED5EPOCHS' else 'RUNNING_CPU113453_QUERIES_SEPARATE_FROM113415_COMMON_DOMAIN'
    master['goal_turn_audit'].update(this_turn='PROGRESS_CANONICAL_MASKED_COVERAGE_CONTROL_AND_ACTUAL_ENDPOINT_PUBLICATION',live_CPU_training=not (p1complete and p2complete),remaining_blocker=None)
    (root/'MASTER_GOAL.json').write_text(json.dumps(master,indent=2)+'\n')
    gate=json.loads((root/'GO_NO_GO.json').read_text());gate.update(updated_utc=now,
        fixed_endpoints_complete=dict(phase1=p1complete,phase2=p2complete,
            ordinary=control.get('ORDINARY_DECISION')==[1,3,5],exhaustive=control.get('EXHAUSTIVE_MARGIN')==[1,3,5],
            legacy=master.get('completed_legacy_coverage_replay_endpoints')==['canonical300','epoch1','epoch3','epoch5']),
        next_action='Complete all registered fixed endpoints including canonical300/full legacy coverage control; preserve all negatives. No extra model tuning or online admission from a single endpoint.')
    (root/'GO_NO_GO.json').write_text(json.dumps(gate,indent=2)+'\n')
    online=json.loads((root/'ONLINE_TRACKING_RESULTS.json').read_text())
    online.update(reason='Full11 training input contract passed. Full native online evaluation remains unadmitted because completed predictive replay endpoints have not passed the scientific gate; remaining fixed endpoints continue.',
        input_contract_status='FULLTRAIN_DATA_CONTRACT_PASS',input_admission_sha256=sha(root/'FULLTRAIN_ADMISSION.json'),updated_utc=now)
    (root/'ONLINE_TRACKING_RESULTS.json').write_text(json.dumps(online,indent=2)+'\n')
    objectives=[]
    armsets=dict(P1=['C2','C3','C4'],P2=['A2_rank','AC_contrastive','A4_uncertainty','A5_no_uncertainty','A6_no_camera','A7_no_time','AS_same_camera'],
        ordinary=['D1_MLP','D2_MLP_native_margin'],exhaustive_margin=['D1_MLP','D2_MLP_native_margin'],legacy_coverage=['C2','C3','C4'])
    for family,path in fits.items():
        coverage=runtime/path/('full/TRAINING_EPOCH_COVERAGE.json' if family=='legacy_coverage' else 'TRAINING_EPOCH_COVERAGE.json')
        if not coverage.exists():continue
        for epoch in json.loads(coverage.read_text())['epochs']:
            loss=epoch.get('per_arm_mean_total_loss',epoch.get('mean_total_loss_per_arm'))
            if len(loss)!=len(armsets[family]):raise ValueError('Loss arm axes differ')
            for i,arm in enumerate(armsets[family]):
                objectives.append(dict(family=family,epoch=epoch['epoch'],arm=arm,unique_query_visits=epoch['coverage']['unique_visited'],
                    mean_total_training_objective=loss[i],preupdate_train_top1=epoch['train_preupdate_top1'][i] if 'train_preupdate_top1' in epoch else None,
                    qualification='Mixed changing checkpoints during optimization; loss definitions differ across arms; not a heldout or cross-objective quality estimate'))
    if objectives:
        with (root/'TRAINING_OBJECTIVE_SUMMARY.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(objectives[0]),lineterminator='\n');w.writeheader();w.writerows(objectives)
    print(json.dumps(dict(published_evidence_entries=len(entries),fixed_endpoints_complete=gate['fixed_endpoints_complete'],observed_utc=now)))


if __name__=='__main__':main()
