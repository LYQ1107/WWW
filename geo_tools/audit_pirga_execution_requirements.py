"""Evidence audit of fixed scientific obligations; never marks the Goal achieved."""
import csv,datetime,json
from pathlib import Path
import torch
from pirga_indexed_store import sha
from cv_pil_native_handoff import safe_torch_load


def main():
    repo=Path(__file__).resolve().parents[1];root=repo/'results/PIRGA_FULLTRAIN';runtime=Path('/data1/liuyeqiang/WWW_pirga_runtime_20261016')
    specs=[('PHASE1','PHASE1_MATCHED_PROTOCOL.json','phase1_grouped_matched_v1/fit','phase1_fixed_native_replays_v1',113415,('C2','C3','C4')),
        ('PHASE2','PHASE2_MATCHED_PROTOCOL.json','phase2_innovations_matched_v1','phase2_fixed_native_replays_v1',113415,('A2_rank','AC_contrastive','A4_uncertainty','A5_no_uncertainty','A6_no_camera','A7_no_time','AS_same_camera')),
        ('ORDINARY','ORDINARY_DECISION_MATCHED_PROTOCOL.json','ordinary_decision_matched_v1','ordinary_fixed_native_replays_v1',113415,('D1_MLP','D2_MLP_native_margin')),
        ('EXHAUSTIVE','EXHAUSTIVE_MARGIN_PROTOCOL.json','exhaustive_margin_matched_v1','exhaustive_margin_native_replays_v1',113453,('D1_MLP','D2_MLP_native_margin')),
        ('LEGACY','LEGACY_CANONICAL_COVERAGE_PROTOCOL.json','legacy_canonical_coverage_fits_v1/full','legacy_canonical_coverage_replays_v1',106800,('C2','C3','C4'))]
    output=dict(observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='INCOMPLETE_UNTIL_ALL_REQUIRED_EVIDENCE',
        sufficient_to_mark_goal_complete=False,scientific_success=False,fixed_families={},deliverables={},conditional_obligations={})
    reference=repo/'results/cv_pil_v3_native_validation/P3_MATCHED300_FULL_DEV12/QUERY_RESULTS_OFFLINE_ONLY.csv'
    if sha(reference)!='3db925d0e10d866dd2d761f061c01a8e25d0747ee21b47d95e6d9aca01a31926':raise ValueError('Frozen same-range reference changed')
    with reference.open() as f:old={tuple(json.loads(row['key'])):row for row in csv.DictReader(f)}
    for name,protocol,fit,replay,n,arms in specs:
        public=root/protocol;live=runtime/protocol;p=json.loads(public.read_text());digest=sha(public)
        source_checks={path:sha(repo/path)==expected for path,expected in p['source_code_sha256'].items()}
        if sha(live)!=digest or not all(source_checks.values()):raise ValueError('Frozen protocol/source mutated '+name)
        coverage=runtime/fit/'TRAINING_EPOCH_COVERAGE.json';epochs=[]
        if coverage.exists():
            for epoch in json.loads(coverage.read_text())['epochs']:
                c=epoch['coverage']
                if not c['full_epoch_pass'] or c['unique_visited']!=n or c['visits']!=n or c['missing'] or c['duplicate_visits'] or c['unexpected']:raise ValueError('Coverage breach '+name)
                epochs.append(epoch['epoch'])
        records=runtime/fit/'CHECKPOINTS.json';checked=[];diagnostics=[]
        if records.exists():
            for spec in json.loads(records.read_text())['checkpoints']:
                if sha(spec['path'])!=spec['sha256']:raise ValueError('Checkpoint mutated '+name)
                state=safe_torch_load(spec['path']);arm=spec['arm'];epoch=spec['epoch']
                if not all(torch.isfinite(value).all() for value in state['state_dict'].values()):raise ValueError('Nonfinite checkpoint weights '+name)
                if name=='PHASE1' and spec.get('kind')=='step300_diagnostic':
                    if epoch is not None or state['epoch'] is not None or arm not in arms or state['arm']!=arm or state['step']!=300 or state['seed']!=20261015 or state['protocol_sha256']!=digest:raise ValueError('Wrong diagnostic checkpoint')
                    diagnostics.append(arm);continue
                if arm not in arms or epoch not in [1,3,5] or state['epoch']!=epoch or state['arm']!=arm or state['seed']!=20261015 or state['step']!=n*epoch or state['protocol_sha256']!=digest:raise ValueError('Wrong checkpoint source/endpoint '+name)
                checked.append([arm,epoch])
        if len({tuple(x) for x in checked})!=len(checked):raise ValueError('Duplicate checkpoint evidence')
        replays=[]
        for epoch in [1,3,5]:
            directory=runtime/replay/f'epoch{epoch}'
            if not (directory/'RESULTS.json').exists():continue
            result=json.loads((directory/'RESULTS.json').read_text())
            if result['status']!='SAME_STATE_NATIVE_REPLAY_ONLY' or result['epoch']!=epoch or result.get('scientific_success') is not False:raise ValueError('Wrong replay level '+name)
            with (directory/'QUERY_RESULTS_OFFLINE_ONLY.csv').open() as f:replayed=list(csv.DictReader(f))
            keys=[tuple(json.loads(row['key'])) for row in replayed]
            if len(keys)!=14557 or set(keys)!=set(old):raise ValueError('DEV scope breach '+name)
            for key,row in zip(keys,replayed):
                for field in ['person','positive_covered','source_primary_state']+[arm+'_'+suffix for arm in ['C0','C1'] for suffix in ['rank_id','rr','rank','accepted_id','primary']]:
                    if row[field]!=old[key][field]:raise ValueError('Frozen control/supervision drift '+str((name,epoch,key,field)))
            replays.append(epoch)
        terminal=runtime/fit/'RESULTS.json';health=dict(status='PENDING_FINAL_FIT_RESULT')
        if terminal.exists():
            final=json.loads(terminal.read_text());steps=final['per_arm_optimizer_steps'];zeros=final['zero_gradient_updates'];nonzeros=final['nonzero_gradient_updates']
            if steps!=n*5 or len(zeros)!=len(arms) or len(nonzeros)!=len(arms) or any(z<0 or g<=0 or z+g!=steps for z,g in zip(zeros,nonzeros)):raise ValueError('Optimizer/gradient coverage breach '+name)
            health=dict(status='PASS_REAL_FINAL_COUNTERS',zero_gradient_updates=zeros,nonzero_gradient_updates=nonzeros,
                actual_per_arm_optimizer_steps=steps,qualification='Zero-gradient visits are preserved explicitly; visits are not independent effective gradients. No claim of global optimization convergence.')
        full=epochs==[1,2,3,4,5] and len(checked)==len(arms)*3 and replays==[1,3,5] and health['status']=='PASS_REAL_FINAL_COUNTERS'
        output['fixed_families'][name]=dict(status='PASS_FIXED_FIT_AND_REPLAYS' if full else 'PENDING_REGISTERED_FIXED_ENDPOINTS',protocol_sha256=digest,
            source_checks=source_checks,completed_full_epochs=epochs,expected_queries_per_epoch=n,checked_checkpoint_endpoints=checked,
            completed_full_DEV12_replays=replays,expected_per_arm_total_updates=n*5,actual_checkpoint_bytes_verified=True,engineering_step300_checkpoints_excluded=diagnostics,
            completed_replay_C0_C1_and_supervision_exact_frozen_reference=True,gradient_health=health,checkpoint_weights_finite=True)
    records=runtime/'legacy_canonical_coverage_fits_v1/canonical300/CHECKPOINTS.json';small=[]
    if records.exists():
        digest=sha(runtime/'LEGACY_CANONICAL_COVERAGE_PROTOCOL.json')
        for spec in json.loads(records.read_text())['checkpoints']:
            if sha(spec['path'])!=spec['sha256']:raise ValueError('Canonical300 checkpoint changed')
            state=safe_torch_load(spec['path'])
            if state['arm']!=spec['arm'] or state['epoch'] is not None or state['step']!=300 or state['seed']!=20261015 or state['protocol_sha256']!=digest or state['mode']!='CANONICAL_MASKED300':raise ValueError('Wrong canonical300 control')
            small.append(spec['arm'])
    output['canonical300_control']=dict(verified_checkpoint_arms=small,full_DEV_replay_complete=(runtime/'legacy_canonical_coverage_replays_v1/canonical300/RESULTS.json').exists())
    output['variance_gate_same_checkpoint_diagnostics']={str(epoch):dict(complete=path.exists(),path=str(path),sha256=sha(path) if path.exists() else None) for epoch,path in [
        (1,runtime/'uncertainty_gate_audit_epoch1_v1/RESULTS.json'),(3,runtime/'uncertainty_gate_fixed_audits_v1/epoch3/RESULTS.json'),(5,runtime/'uncertainty_gate_fixed_audits_v1/epoch5/RESULTS.json')]}
    names=['MASTER_GOAL.json','DATA_SCOPE_AUDIT.json','FULLTRAIN_ASSET_MANIFEST.json','SUPERVISION_COVERAGE.json','TRAINING_PROTOCOL.json',
        'TRAINING_CURVES.csv','TRAINING_EPOCH_COVERAGE.json','PREDICTION_ABLATIONS.csv','RELIABILITY_ABLATIONS.csv','NATIVE_DECISION_ABLATIONS.csv',
        'ORIGINAL_ERROR_REPAIRS.json','PER_PERSON_AND_SCENE_RESULTS.csv','ONLINE_TRACKING_RESULTS.json','SPEED_AND_MEMORY.json','RESOURCE_LEDGER.json','GO_NO_GO.json']
    for name in names:
        path=root/name
        output['deliverables'][name]=dict(exists=path.exists(),sha256=sha(path) if path.exists() else None,existence_alone_proves_content_complete=False)
    for name in ['FINAL_RESEARCH_REPORT_ZH.md','REVIEWER_QUESTIONS.md']:
        path=repo/'docs/PIRGA_FULLTRAIN'/name;output['deliverables'][name]=dict(exists=path.exists(),sha256=sha(path) if path.exists() else None,existence_alone_proves_content_complete=False)
    admission=root/'FULLTRAIN_ADMISSION.json';data=json.loads(admission.read_text())
    output['asset_admission']=dict(status=data['status'],actual_queries=data['actual_queries'],fit_scenes=data['fit_scenes'],sha256=sha(admission),official_TEST_in_scope=False)
    output['conditional_obligations']=dict(learned_reliability='PENDING_FINAL_PHASE2_GATE',predictive_margin_and_full_PIRGA='PENDING_FINAL_PHASE2_GATE',
        full_native_online='NOT_ADMITTED_AT_COMPLETED_ENDPOINTS',full24_OOF='NOT_RUN_SCOPE_AND_SEPARATE_RESOURCE_REVIEW_REQUIRED',
        full24_final_fit='NOT_RUN_REQUIRES_LOCKED_METHOD_AND_VALIDATION_PLAN',second_seed='NOT_RUN_NO_INDEPENDENT_GENERALIZATION_CLAIM')
    output['limitations']=['Checkpoint/coverage integrity and complete replay scope do not themselves certify scientific value.',
        'Final audit must inspect objectives, supervision contract, old evidence preservation, GT separation, gate and report content.',
        'Conditional phases require explicit gate decisions after all fixed endpoints. No goal completion or online metrics inferred here.']
    path=root/'EXECUTION_REQUIREMENTS_AUDIT.json';path.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(dict(families={k:v['status'] for k,v in output['fixed_families'].items()},goal_completion_proven=False)))


if __name__=='__main__':main()
