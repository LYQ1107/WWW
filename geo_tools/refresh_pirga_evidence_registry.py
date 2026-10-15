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
    reference=repo/'results/cv_pil_v3_native_validation/P3_MATCHED300_FULL_DEV12/QUERY_RESULTS_OFFLINE_ONLY.csv'
    if sha(reference)!='3db925d0e10d866dd2d761f061c01a8e25d0747ee21b47d95e6d9aca01a31926':raise ValueError('Frozen camera/query metadata changed')
    with reference.open() as f:camera_by_key={tuple(json.loads(r['key'])):int(r['camera']) for r in csv.DictReader(f)}
    transitions=[];camera_rows=[]
    for entry in entries:
        source=repo/entry['path'];csv_path=source.with_name('QUERY_RESULTS_OFFLINE_ONLY.csv')
        with csv_path.open() as f:physical=list(csv.DictReader(f))
        if len(physical)!=14557 or {tuple(json.loads(r['key'])) for r in physical}!=set(camera_by_key):raise ValueError('Camera report scope differs')
        arms=[k[:-8] for k in physical[0] if k.endswith('_primary')]
        by_camera={}
        for row in physical:by_camera.setdefault(camera_by_key[tuple(json.loads(row['key']))],[]).append(row)
        for camera,group in sorted(by_camera.items()):
            for arm in arms:
                covered=[r for r in group if r[arm+'_rr']!=''];rr=[float(r[arm+'_rr']) for r in covered]
                repaired=[r for r in group if r['source_primary_state'] in ['wrong_link','missed_known_positive'] and r['positive_covered']=='True' and r[arm+'_primary']=='True' and r[arm+'_accepted_id']!='']
                correct=[r for r in group if r['source_primary_state'] in ['correct_link','correct_reject']]
                camera_rows.append(dict(family=entry['family'],endpoint=source.parent.name,camera=camera,method=arm,queries=len(group),positive_covered=len(covered),
                    top1=sum(v==1 for v in rr)/len(rr) if rr else None,mrr=sum(rr)/len(rr) if rr else None,
                    primary_known_correct=sum(r[arm+'_primary']=='True' for r in group),primary_known_wrong=sum(r[arm+'_primary']=='False' for r in group),primary_unknown=sum(r[arm+'_primary']=='' for r in group),
                    original_error_repairs=len(repaired),known_harms=sum(r[arm+'_primary']=='False' for r in correct),correct_to_unknown=sum(r[arm+'_primary']=='' for r in correct),
                    original_unknown_to_wrong=sum(r['C0_primary']=='' and r[arm+'_primary']=='False' for r in group),repair_person_groups=len({r['person'] for r in repaired}),
                    qualification='ExposedDEV12 physical camera groups; same people recur across cameras, not independent replications'))
        for kind in ['wrong_link','missed_known_positive']:
            errors=[r for r in physical if r['source_primary_state']==kind]
            for arm in arms:
                counters=dict(accepted_known_correct=0,accepted_known_wrong=0,accepted_unknown=0,rejected_with_pure_positive=0,rejected_without_pure_positive=0);repair_groups=set()
                for row in errors:
                    if row[arm+'_accepted_id']=='':key='rejected_with_pure_positive' if row['positive_covered']=='True' else 'rejected_without_pure_positive'
                    else:key={'True':'accepted_known_correct','False':'accepted_known_wrong','':'accepted_unknown'}[row[arm+'_primary']]
                    counters[key]+=1
                    if key=='accepted_known_correct':repair_groups.add((json.loads(row['key'])[0],row['person']))
                if sum(counters.values())!=len(errors):raise ValueError('Error action partition incomplete')
                transitions.append(dict(family=entry['family'],endpoint=source.parent.name,method=arm,original_error_type=kind,original_errors=len(errors),
                    **counters,identity_repair_person_groups=len(repair_groups),
                    qualification='Rejecting a wrong edge is separate from certified correct identity reuse; no final Bank/Gallery/birth counterfactual identity metric'))
    if transitions:
        with (root/'ORIGINAL_ERROR_ACTION_TRANSITIONS.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(transitions[0]),lineterminator='\n');w.writeheader();w.writerows(transitions)
    if camera_rows:
        with (root/'PER_CAMERA_RESULTS.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(camera_rows[0]),lineterminator='\n');w.writeheader();w.writerows(camera_rows)
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
    elif control.get('PHASE2'):master['phases']['2']='REPLAY_EPOCHS_'+'_'.join(map(str,control['PHASE2']))+'_COMPLETE_FIXED_REMAINING_ENDPOINTS_PENDING'
    for epoch in reversed([1,3,5]):
        path=root/f'MATCHED_ALL_CONTROL_GATE_EPOCH{epoch}.json'
        if path.exists():
            latest=json.loads(path.read_text());master['latest_matched_new_model_status']=dict(epoch=epoch,status=latest['status'],phase2_mechanism_gate_pass=latest['phase2_mechanism_gate_pass']);break
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
    if 'latest_matched_new_model_status' in master:gate['phase2']=master['latest_matched_new_model_status']
    if p1complete:gate['phase1']='FIXED1_3_5_COMPLETE_LATEST_'+master['latest_fixed_epoch_PIRGA_status']
    elif status['P1']['status']=='COMPLETE':gate['phase1']='TRAINING_COMPLETE_PENDING_ALL_FIXED_REPLAYS'
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
    bounded=root/'BOUNDED_ACTION_DIAGNOSTIC_RESULTS.json'
    if sha(bounded)!='cc8fa6a3c1c417f0c31b1474f402cfd9b3493d5c62c22596eaafab1c27028f8a':raise ValueError('Sealed interval diagnostic changed')
    diagnostic=json.loads(bounded.read_text());types={}
    for kind in ['wrong_link','missed_known_positive']:
        events=[e for e in diagnostic['events'] if e['original_state']==kind]
        pure=[e for e in events if e['pure_positive_candidates']>0];unfixed=[e for e in pure if not e['C1_original_error_repaired']]
        impossible=[e for e in unfixed if e['certified_impossible']];witness=[e for e in unfixed if e['interval_witness_found']]
        if any(e['certified_impossible'] and e['interval_witness_found'] for e in unfixed):raise ValueError('Interval proof contradicts witness')
        types[kind]=dict(original_errors=len(events),legal_pure_positive_available=len(pure),legal_pure_positive_absent=len(events)-len(pure),
            C1_repaired=sum(e['C1_original_error_repaired'] for e in events),C1_unrepaired_pure_positive=len(unfixed),
            bounded_interval_certified_impossible=len(impossible),arbitrary_interval_witnesses=len(witness),
            not_excluded_but_no_witness=len(unfixed)-len(impossible)-len(witness),
            interval_exclusion_keys=[e['key'] for e in impossible],witness_keys=[e['key'] for e in witness])
    (root/'BOUNDED_OPPORTUNITY_BY_ORIGINAL_ERROR_TYPE.json').write_text(json.dumps(dict(
        status='OFFLINE_ORIGINAL_STATE_ACTION_SCOPE_DIAGNOSTIC',source_sha256=sha(bounded),by_original_error_type=types,
        models_in_scope=diagnostic['models_in_scope'],scientific_success=False,online_admitted=False,
        qualification='Missing pure candidate is not score-repairable under the certified repair definition. Interval exclusions concern bounded new-head primary assignment at these original states. They do not exclude earlier online interventions changing later states, nor unbounded old heads. GT-assisted witnesses are not learned gains; no-witness is not proof of impossibility.'),indent=2)+'\n')
    print(json.dumps(dict(published_evidence_entries=len(entries),fixed_endpoints_complete=gate['fixed_endpoints_complete'],observed_utc=now)))


if __name__=='__main__':main()
