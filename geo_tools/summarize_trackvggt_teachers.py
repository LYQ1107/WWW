"""Snapshot only actual completed teacher reports; never impute missing scores."""
import csv
import collections
import argparse
import torch
from common import *
from run_research_queue import process_identity
from audit_trackvggt_ag_results import changes


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--final-negative-report',action='store_true',help='close the report index only after actual full-system results and final documents are complete; never admits a student or changes the goal state')
    args=parser.parse_args()
    runtime=Path('/data1/liuyeqiang/WWW_trackvggt_runtime');rows=[]
    gate_path=ROOT/'results/trackvggt_teacher_gate_v1.json'
    gate=json.loads(gate_path.read_text()) if gate_path.exists() else {}
    closed_negative=gate.get('status')=='NO_UNIQUE_VGGT_TEACHER_EVIDENCE'
    if args.final_negative_report:
        if not closed_negative:raise ValueError('a completed negative teacher gate is required')
        for name,status in [('trackvggt_system_execution_snapshot_v1.json','COMPLETE_FRESH_NATIVE_SYSTEM_EVALUATION'),
            ('trackvggt_system_query_pairs_v1.json','COMPLETE_ACTUAL_NATIVE_QUERY_PAIRS'),
            ('trackvggt_queue_stage_costs_v1.json','COMPLETE_ACTUAL_QUEUE_STAGE_COSTS')]:
            if json.loads((ROOT/'results'/name).read_text())['status']!=status:
                raise ValueError('final report cannot close before actual complete evidence: '+name)
        for name,marker in [('TRACKVGGT_FINAL_REPORT.md','状态：NEGATIVE_RESULT_COMPLETE'),
            ('TRACKVGGT_REVIEWER_AUDIT.md','FINAL_REVIEWER_AUDIT_NEGATIVE_SCOPE'),
            ('TRACKVGGT_REQUIREMENT_AUDIT.md','最终需求验收（完成）')]:
            if marker not in (ROOT/'docs'/name).read_text():raise ValueError('final document closure is still missing: '+name)
    student_status='CONDITIONAL_NOT_EXECUTED_RELATION_GATE_FAILED' if closed_negative else 'CONDITIONAL_NOT_EXECUTED_RELATION_GATE_NOT_PASSED'
    for path in sorted((runtime/'models').glob('*/manifest.json')):
        value=json.loads(path.read_text())
        temporal=value.get('status')=='COMPLETE_MATCHED_REAL_CYCLE_CONTINUATION'
        if value.get('status')!='COMPLETE_FINITE_TEACHER_FIT_AND_FIXED_VALIDATION' and not temporal:continue
        for name in ('checkpoint.pt','validation.json','validation_query_shuffle.json','loss_curve.json'):
            if sha256(path.parent/name)!=value['artifacts'][name]:raise ValueError('completed teacher artifact changed')
        if temporal:
            state=torch.load(path.parent/'checkpoint.pt',map_location='cpu');source=json.loads((Path(value['source_checkpoint']).parent/'manifest.json').read_text())
            value=dict(value,settings=state['settings'],all_fit_queries=source['all_fit_queries'],validation_queries=value['validation_metrics']['requests'],
                completed_steps=value['total_updates'],trainable_parameters=source['trainable_parameters'])
        settings=value['settings'];metrics=value['validation_metrics'];shuffle=value['query_shuffle_metrics']
        rows.append(dict(tag=path.parent.name,status=value['status'],fit_queries=value['all_fit_queries'],supervised_fit_queries=value['supervised_fit_queries'],
            validation_queries=value['validation_queries'],training_seed=settings['training_seed'],native_inference_seed=settings['actual_native_inference_seed'],
            raw_generic_provenance_seed=value['provenance']['seed'],seed_reporting_correction_required=value['provenance']['seed']!=settings['training_seed'],
            completed_updates=value['completed_steps'],backbone=settings['backbone'],training_phase='temporal_continuation' if temporal else 'no_cycle_fit',lambda_cycle=value.get('lambda_cycle',''),
            architecture=settings['architecture'],AG=settings['ag_mode'],QD=settings['qd_mode'] if settings['architecture']=='qd' else '',history_summary=settings['summary'],
            K=settings['k'],feature_variant=settings.get('feature_variant','full'),feature_plan_sha256=settings.get('feature_plan_sha256') or '',
            parameters=value['trainable_parameters'],Recall1=metrics['Recall1'],Recall5=metrics['Recall5'],MRR=metrics['MRR'],pair_AUC=metrics['pair_auc'],pair_AP=metrics['pair_ap'],
            matched_positive_denominator=metrics['matched_eligible'],known_queries=metrics['known_queries'],neutral_queries=metrics['neutral_small_target_queries'],
            shuffle_Recall1=shuffle['Recall1'],neutral_fit_forwards=value.get('neutral_fit_query_forwards',0),
            zero_gradient_batches=value.get('zero_gradient_batches',0),checkpoint_sha256=value['checkpoint_sha256'],manifest_sha256=sha256(path),manifest=str(path)))
    target=ROOT/'results/teacher_ablation.csv'
    with target.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    queues=[]
    for pattern in ['trackvggt_qd_fit_v1_gpu*','trackvggt_crop_fit_v1_gpu*','trackvggt_cycle_fit_v1_gpu*','trackvggt_qd_native_control_v1_gpu*','trackvggt_live_baseline_v1_gpu*']:
        for path in sorted((runtime/'queues').glob(pattern+'/manifest.json')):
            value=json.loads(path.read_text());live=process_identity(value['pid'])
            if live and (f"--plan {value['plan']}" not in live[1] or f"--queue {value['queue']}" not in live[1]):live=None
            queues.append(dict(path=str(path),status=value['status'],pid=value['pid'],live_controller=live is not None,
                actual_command=live[1] if live else None,last_stage=value['stages'][-1] if value['stages'] else None))
    if closed_negative and (len(rows)!=56 or gate['completed_registered_runs']!=56):raise ValueError('closed gate does not match actual teacher inventory')
    snapshot=dict(status='COMPLETE_REGISTERED_TEACHER_SCREENING' if closed_negative else 'ACTUAL_TEACHER_PROGRESS_SNAPSHOT',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),completed_reports=len(rows),
        csv_sha256=sha256(target),rows=rows,queues=queues,GT_unknown_not_known_wrong=True,
        teacher_gate_status=gate.get('status','NOT_AUDITED'),student_disposition=student_status,
        scope='Every number from an actual completed source-bound run.150/1000/1200 rows are distinct budgets. Temporal rows resolve seed15 from actual code/source checkpoint; raw generic default14 is retained and flagged. Pair results do not prove actual Hungarian/final tracking gain.',provenance=provenance())
    save_json(ROOT/'results/trackvggt_teacher_execution_snapshot_v1.json',snapshot)
    paired=[];native_states=None;primary_states=None;reference=None
    for row in rows:
        if row['completed_updates']<1000:continue
        validation_path=Path(row['manifest']).parent/'validation.json'
        values=json.loads(validation_path.read_text())['rows']
        signature=[[v[k] for k in ('key','query_gt_known','top8_positive','known_unmatched','original_native_top1_state','actual_frozen_gmt_primary_state')] for v in values]
        if reference is None:
            reference=signature
            native_states=dict(collections.Counter(v['original_native_top1_state'] for v in values))
            primary_states=dict(collections.Counter(v['actual_frozen_gmt_primary_state'] for v in values))
        if signature!=reference or len(values)!=300:raise ValueError('teacher correction comparison uses different validation keys/labels')
        paired.append(dict(tag=row['tag'],completed_updates=row['completed_updates'],backbone=row['backbone'],
            training_phase=row['training_phase'],feature_variant=row['feature_variant'],
            ranking_changes_vs_native=changes(values),validation_sha256=sha256(validation_path),
            actual_learned_solver_or_final_identity_repairs=None))
    corrections=dict(status='ACTUAL_COMPLETED_TEACHER_RANKING_AUDIT_NOT_SOLVER_REPAIRS',
        original_validation_queries=300,native_row_top1_states=native_states,actual_native_primary_states=primary_states,
        original_validation_actual_primary_known_error_queries=sum(primary_states.get(k,0) for k in ('wrong_link','missed_known_positive')),
        rows=paired,teacher_results_not_additive=True,
        actual_learned_primary_or_final_ID_audit='NOT_EXECUTED_NO_VALID_TEACHER_OR_STUDENT_ADMISSION',
        interpretation='Per-query rank changes on fixed actual native candidates. Unknown/ambiguous transitions remain uncertified. The original300 cohort has native row-ranking errors but zero certified actual primary errors. Do not turn these ranking fixes into Hungarian/Bank/final-ID repairs or sum overlapping fixes across teachers.',
        DEV12_engineering_release='Separate complete14557-query asset covers26 actual native errors; neither its expanded inputs nor its labels are used here for model selection, fitting or a new VGGT repair claim.',
        provenance=provenance())
    save_json(ROOT/'results/identity_correction_audit.json',corrections)
    index_path=ROOT/'results/experiment_manifest.json'
    index=json.loads(index_path.read_text())
    index.setdefault('inherited_geo_stream_status',index['status'])
    index['status']='TRACKVGGT_BOUNDED_NEGATIVE_REPORTS_COMPLETE' if args.final_negative_report else 'TRACKVGGT_RESEARCH_ACTIVE_SNAPSHOT'
    index['inherited_experiments_scope']='Original experiments array/raw_gt_policy/split retained from the completed predecessor project; not newly executed TrackVGGT results.'
    index['trackvggt']=dict(branch='research/trackvggt-identity-query-www27-20261015',
        snapshot_utc=snapshot['utc'],teacher_reports=[dict(tag=r['tag'],manifest=r['manifest'],manifest_sha256=r['manifest_sha256'],
            checkpoint_sha256=r['checkpoint_sha256'],actual_training_seed=r['training_seed'],native_inference_seed=r['native_inference_seed'],
            raw_generic_provenance_seed=r['raw_generic_provenance_seed'],seed_reporting_correction_required=r['seed_reporting_correction_required'],
            completed_updates=r['completed_updates'],training_phase=r['training_phase']) for r in rows],
        live_queue_snapshot=queues,
        current_evidence={str(path.relative_to(ROOT)):sha256(path) for name in [
            'trackvggt_dev12_native_release_v1.json','trackvggt_ag_v2_v3_primary_results_v1.json',
            'trackvggt_crop_primary_results_v1.json','trackvggt_qd_primary_results_v1.json',
            'trackvggt_cycle_vggt_paired_v1.json','trackvggt_cycle_dinov2_paired_v1.json',
            'trackvggt_system_execution_snapshot_v1.json','identity_correction_audit.json',
            'trackvggt_fit_endpoint_diagnostics_v1.json','trackvggt_queue_stage_costs_v1.json',
            'trackvggt_inherited_native_reference_v1.json','trackvggt_teacher_gate_v1.json',
            'trackvggt_frozen_direct_input_hash_recheck_v1.json','trackvggt_system_query_pairs_v1.json',
            'trackvggt_fit_population_disclosure_v1.json']
            if (path:=ROOT/'results'/name).exists()},
        student_status=student_status,
        overall_goal_complete=False,scientific_success=False)
    if args.final_negative_report:
        index['trackvggt'].pop('overall_goal_complete')
        index['trackvggt'].update(negative_report_scope_complete=True,
            goal_completion_scope='Final goal completion additionally requires manual requirement review and verification that the final revision is committed/pushed. This reporter does not change the goal state.')
    save_json(index_path,index)
    student_rows=[dict(variant=variant,planned_training_seed=seed,status=student_status,
        actual_trained_seed='',checkpoint_sha256='',HOTA='',CVIDF1='',IDSW='',complete_process_FPS='',
        reason=('All56 registered teacher runs completed; independent relation gate failed. Planned seeds are not trained seeds.' if closed_negative else
            'No valid independent teacher/student admission. Remaining teacher hypotheses must finish before final gate disposition; planned seeds are not trained seeds.'))
        for variant in ['GT_only','KD_only','GT_plus_KD','DINO_KD','GMT_teacher'] for seed in [20261015,20261016,20261017]]
    student_path=ROOT/'results/student_ablation.csv'
    if student_path.exists():
        with student_path.open(newline='') as stream:
            if any(r.get('checkpoint_sha256') or r.get('actual_trained_seed') for r in csv.DictReader(stream)):
                raise ValueError('actual student results need their own reporter; do not replace them with conditional rows')
    with student_path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,list(student_rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(student_rows)
    print(json.dumps(dict(status=snapshot['status'],completed_reports=len(rows),csv_sha256=sha256(target))))


if __name__=='__main__':main()
