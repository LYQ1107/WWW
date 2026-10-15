"""Current requirement receipts; readiness still needs a manual scope review.

Never converts conditional/unmeasured evidence into scientific success and never
changes the active goal. System runs, raw hashes and document closure are checked
separately from mere named-artifact presence.
"""
import csv
from common import *


def main():
    rows=[];bound={}
    def receipt(name,expected,identity,scope,check=lambda value:True):
        path=ROOT/'results'/name
        if not path.exists():
            rows.append(dict(id=identity,status='INCOMPLETE',reason='Missing evidence',path=str(path)));return None
        value=json.loads(path.read_text());bound[str(path)]=sha256(path)
        ok=value.get('status')==expected and check(value)
        rows.append(dict(id=identity,status='VERIFIED_AT_STATED_SCOPE' if ok else 'INCOMPLETE',
            path=str(path),sha256=sha256(path),observed_status=value.get('status'),scope=scope))
        return value
    design=json.loads((ROOT/'manifests/trackvggt_design_v1.json').read_text())
    request=ROOT/'docs/TRACKVGGT_FINAL_GOAL.md'
    if sha256(request)!=design['request_sha256']:raise ValueError('original research request changed')
    if subprocess.check_output(['git','symbolic-ref','--short','HEAD'],cwd=ROOT,text=True).strip()!=design['branch']:
        raise ValueError('not the authorized isolated research branch')
    phase0=receipt('trackvggt_phase0_audit_v1.json','PASS_PHASE0_SOURCE_INPUT_INTERFACE_GATES','source_and_actual_interfaces',
        'Actual source/weight/layer/attention interface checks, not identity utility.')
    quality=receipt('trackvggt_gt_quality_summary_v1.json','COMPLETE_ALL24_CPU_CENSUS','all24_metadata_GT_quality',
        'All24 TRAIN metadata/GT census; reused raw uniqueness audit, no repeat tracking.')
    if phase0 and quality:
        signatures={str(CHECKPOINT):phase0['provenance']['checkpoint_sha256'],
            str(ROOT/'configs/VISION_test.yaml'):phase0['provenance']['config_sha256'],
            str(DATA/'annotations/train.json'):phase0['provenance']['annotation_sha256'],**quality['raw_gt_sha256']}
        for path,digest in signatures.items():
            if sha256(path)!=digest:raise ValueError('frozen original checkpoint/config/annotation/raw GT changed: '+path)
        rows.append(dict(id='unchanged_original_GMT_config_annotations_raw_GT',status='VERIFIED_AT_STATED_SCOPE',
            checked=len(signatures),scope='Current file hashes against original locks; not a repeated GT uniqueness audit or a hash of every video pixel.'))
    with (ROOT/'results/trackvggt_all24_dataset_census_v1.csv').open(newline='') as stream:
        if len(list(csv.DictReader(stream)))!=24:raise ValueError('all24 census membership is incomplete')
    receipt('trackvggt_full_train_completion_audit_v1.json','PASS_COMPLETE_REAL_AUDIT_CONTRACTS','full12_causal_error_audit',
        'Twelve complete association-TRAIN sequences, actual native states and post-run labels.',
        lambda v:len(v['rows'])==12 and v['total_actual_primary_queries']==169468)
    receipt('trackvggt_v3_native_capture_complete_v1.json','PASS_ALL11_SELECTED_INPUT_CONSTRUCTION_AND_FULL_PARITY','native_V3_inputs',
        'All11 fit scenes, continuous native capture with selected inputs; original validation reused.',
        lambda v:len(v['rows'])==11 and v['original_validation300_reused'])
    receipt('trackvggt_v3_tokens_complete_v1.json','PASS_V3_FROZEN_FEATURES_COMPLETE','frozen_backbone_inputs',
        'Actual completed frozen feature inputs and exact old validation control.',lambda v:len(v['rows'])==2)
    crop=receipt('trackvggt_crop_inputs_complete_v1.json','PASS_ALL_FIXED_CROP_INPUTS','matched_crop_inputs',
        'Same256 fit/300 validation with native/history/label equality across two encoders.',
        lambda v:len(v['rows'])==2 and all(r['fit']==256 and r['validation']==300 and r['matched_supervised_fit_queries']==234 for r in v['rows']))
    cap=design['max_teacher_cache_bytes']
    token_bytes=sum(p.stat().st_size for p in Path(design['runtime']).joinpath('tokens').rglob('*.npz'))
    rows.append(dict(id='shared_compact_NPZ_budget',status='VERIFIED_AT_STATED_SCOPE' if token_bytes<=cap else 'INCOMPLETE',
        actual_bytes=token_bytes,maximum_bytes=cap,scope='All current shared token NPZ files, including probes; not total server disk.'))
    receipt('trackvggt_ag_v2_v3_primary_results_v1.json','COMPLETE_PRIMARY_V2_V3_AG_COMPARISON','AG_V2_V3_controls',
        'All12 full primary AG/ROI/DINO/native settings; same original300 validation.',lambda v:v['completed_full_settings']==12)
    receipt('trackvggt_ag_K_ablation_v1.json','COMPLETE_FITTED_K_INFERENCE_ABLATION','K16_24_32',
        'Same fitted weights, bounded inference-K ablation, not three K training runs.',lambda v:sorted(v['evaluation_K'])==[16,24,32])
    receipt('trackvggt_crop_primary_results_v1.json','COMPLETE_FIXED224_FULL_LOCAL_COMBINED_PRIMARY_COMPARISON','crop_resolution_controls',
        'Six full/local/combined1000 fits, known/unknown/size-bin/paired DINO audit.',lambda v:len(v['rows'])==6)
    receipt('trackvggt_crop_matched_randomness_v1.json','PASS_MATCHED_CROP_QUERY_DRAWS','crop_draw_matching',
        'Frozen index/order/seed/code and terminal RNG; direct per-step keys were not logged.')
    receipt('trackvggt_qd_primary_results_v1.json','COMPLETE_MATCHED_QD_PRIMARY_COMPARISON','QD_capacity_shuffle',
        'Five VGGT and two DINO QD fits plus native capacity control, per-candidate shuffle and known fixes.',
        lambda v:len(v['rows'])==7 and not v['pending'] and v['native_QD_capacity_control'] is not None)
    receipt('trackvggt_real_cycle_features_v1.json','PASS_REAL_V3_CYCLE_FEATURES_AND_FITTED_AG_GRADIENT','real_causal_cycle_features',
        'Actual A(t-1)-B(t)-A(t) bounded native features, legal NULL/visibility and real gradients.',
        lambda v:v['cycles']==128 and v['visible_cycles']==63)
    for backbone in ['vggt','dinov2']:
        receipt(f'trackvggt_cycle_{backbone}_paired_v1.json','PASS_IDENTICAL_SOURCE_AND_MATCHED_TEMPORAL_CONTROLS','cycle_pair_'+backbone,
            'Equal source/AdamW/schedules/RNG,200 added updates, known versus unknown ranking transitions.',
            lambda v:v['additional_updates']==200 and v['total_updates']==1200 and v['final_numpy_and_Torch_dropout_RNG_exact'])
    receipt('trackvggt_fit_endpoint_diagnostics_v1.json','SAVED_ENDPOINT_DIAGNOSTICS_COMPLETE','retained_fit_endpoints',
        'All52 no-cycle endpoints on up to256 fixed fit queries, zero updates; no reconstructed historical periodic Recall curve.',
        lambda v:len(v['completed'])==52 and not v['pending'] and v['model_updates']==0)
    receipt('trackvggt_fit_population_disclosure_v1.json','COMPLETE_ACTUAL_FIT_POPULATION_DISCLOSURE','actual_training_population_and_upstream_budget',
        'Visual2201 versus native/ROI2733 gradient queries disclosed; exact VGGT/DINO visual populations and matched234-query crops. No claim of identical native/ROI gradient populations or total upstream training.',
        lambda v:len(v['profiles'])==4 and v['model_updates']==0 and v['matched_crop_supervised']==234)
    gate=receipt('trackvggt_teacher_gate_v1.json','NO_UNIQUE_VGGT_TEACHER_EVIDENCE','teacher_relation_gate',
        'Bounded negative screening after every registered independent hypothesis; no positive scientific claim.',
        lambda v:v['completed_registered_runs']==56 and not v['pending'] and not v['teacher_admitted'])
    if gate and gate.get('status')=='NO_UNIQUE_VGGT_TEACHER_EVIDENCE':
        for run in gate['runs'].values():
            if sha256(run['path'])!=run['sha256']:raise ValueError('fitted teacher receipt changed since completed gate')
        with (ROOT/'results/student_ablation.csv').open(newline='') as stream:students=list(csv.DictReader(stream))
        if len(students)!=15 or any(r['actual_trained_seed'] or r['checkpoint_sha256'] or r['status']!='CONDITIONAL_NOT_EXECUTED_RELATION_GATE_FAILED' for r in students):
            raise ValueError('conditional student table falsely implies trained results or omits prespecified cases')
        rows.append(dict(id='IRGD_student_and_three_trained_seeds',status='CONDITIONAL_NOT_EXECUTED_RELATION_GATE_FAILED',
            scope='Explicit locked relation gate prevents conditional distillation/deployment. Fifteen planned variant/seed rows are unexecuted, not actual training seeds or failed students.'))
    receipt('trackvggt_relation_zero_prefix40_v1.json','PASS_ACTUAL_ZERO_RESIDUAL_RAW_BYTE_PARITY','student_prototype_zero_contract',
        'Actual40-group zero adapter equality only; no fitted student efficacy.')
    receipt('trackvggt_inherited_native_reference_v1.json','PASS_SIX_INHERITED_NATIVE_REFERENCES','native_inference_seed_controls',
        'Six old full DEV5 native/Prior references reverified; not newly trained students or new timings.',lambda v:len(v['rows'])==6)
    release=receipt('trackvggt_dev12_native_release_v1.json','DEV12_NATIVE_ASSET_HANDOFF_COMPLETE','DEV12_independent_handoff',
        'Complete GT-independent1200-group expanded development release with actual full-input consumer/parity evidence.',
        lambda v:v['groups']==1200 and v['queries']==14557 and v['coverage_audit']['errors_with_full_candidate_and_history_inputs']==26)
    if release:
        root=Path(release['release_directory'])
        if sha256(root/'artifact_sha256.json')!=release['artifact_sha256_index_sha256'] or sha256(root/'manifest.json')!=release['sealed_manifest_sha256']:
            raise ValueError('DEV12 release trust anchors changed')
    receipt('trackvggt_frozen_direct_input_hash_recheck_v1.json','PASS_DIRECT_FROZEN_INPUT_HASH_RECHECK','direct_frozen_reference_recheck',
        '4006 direct input references, not recursively every internal file or new tracker inference.',lambda v:v['direct_inputs']==4006 and not v['mismatches'])
    system=receipt('trackvggt_system_execution_snapshot_v1.json','COMPLETE_FRESH_NATIVE_SYSTEM_EVALUATION','fresh_full_system_retrieval_DEV5',
        'Six actual full profiled native/Prior2/4/10 cases plus two new complete DEV5 TrackEval results.',
        lambda v:len(v['rows'])==6 and not v['pending'] and not v['pending_tracking_metrics'])
    receipt('trackvggt_system_query_pairs_v1.json','COMPLETE_ACTUAL_NATIVE_QUERY_PAIRS','full_detector_and_retrieval_pairing',
        'Actual OFF/Prior full detector boxes/scores/order and query GT labels paired; eligibility drift and common-query fixes/harms separate.',
        lambda v:len(v['rows'])==3 and not v['pending'] and v['verified_complete_outcome_runs']==6 and len(v['all_detector_request_retention'])==6)
    receipt('trackvggt_queue_stage_costs_v1.json','COMPLETE_ACTUAL_QUEUE_STAGE_COSTS','actual_queue_costs',
        'Complete actual GPU-visible process wall; CPU waits excluded, separate manual/DEV12/inherited exclusions.',
        lambda v:not v['pending_or_running'] and v['native_system_completed_producer_processes']==8
            and v['native_system_completed_GPU_visible_process_wall_hours']<=v['native_system_GPU_wall_hour_cap']
            and v['native_system_current_logical_output_bytes']<=v['native_system_output_byte_cap'])
    receipt('trackvggt_native_system_sampled_resources_v1.json','FINISHED_PROBE_SAMPLED_RESOURCES','sealed_passive_native_resources',
        'Finished3650-group/31-observed-process passive probe. Whole-device samples include waits/unattributed activity; not per-method kernel utilization or lifetime allocation peaks.',
        lambda v:v['sample_groups']==3650 and len(v['process_rows'])==31)
    rows.append(dict(id='historical_monitoring_limits',status='NOT_CAPTURED_DISCLOSED_LIMITATION',
        scope='Periodic training Recall/margin curves and direct per-step crop draw keys were not recorded. Retained endpoints, loss/gradient logs and terminal RNG do not recreate them. This prevents claims of full historical monitoring and does not establish positive efficacy.'))
    docs=['TRACKVGGT_FINAL_GOAL','TRACKVGGT_SOURCE_AUDIT','TRACKVGGT_ARCHITECTURE','TRACKVGGT_EXPERIMENT_PLAN',
        'TRACKVGGT_HARD_EVENTS','TRACKVGGT_TEACHER_RESULTS','TRACKVGGT_STUDENT_RESULTS','TRACKVGGT_REVIEWER_AUDIT','TRACKVGGT_FINAL_REPORT',
        'PHASE0_SOURCE_AND_INTERFACE_AUDIT','TRACKVGGT_FULL_TRAIN_AUDIT','TRACKVGGT_DATASET_DIFFICULTY_AUDIT',
        'TRACKVGGT_CORRECTABLE_ERROR_ANALYSIS','TRACKVGGT_FIT_DATA_EXPANSION_DECISION','TRACKVGGT_DEV12_NATIVE_ASSET_HANDOFF',
        'TRACKVGGT_DEV12_NATIVE_ASSET_SCHEMA','TRACKVGGT_REQUIREMENT_AUDIT','TRACKVGGT_RUNTIME_PROTOCOL','TRACKVGGT_SYSTEM_RESULTS','TRACKVGGT_CAPACITY_CONTROL']
    results=['teacher_ablation.csv','student_ablation.csv','full_tracking_metrics.csv','runtime_comparison.csv','identity_correction_audit.json','experiment_manifest.json',
        'trackvggt_all24_dataset_census_v1.csv','trackvggt_full_train_error_census_v1.csv','trackvggt_candidate_headroom_v1.json',
        'trackvggt_event_type_distribution_v1.json','trackvggt_scene_difficulty_v1.csv','trackvggt_dev5_retrospective_analysis_v1.json','trackvggt_cohort_v2_v3_comparison.json']
    manifests=['trackvggt_full_audit_plan_v1.json','trackvggt_expanded_training_v3.json','trackvggt_dev12_native_asset_release_v1.json']
    artifacts={}
    for path in [*(ROOT/'docs'/(n+'.md') for n in docs),*(ROOT/'results'/n for n in results),*(ROOT/'manifests'/n for n in manifests)]:
        artifacts[str(path)]=sha256(path) if path.exists() else None
    missing=[p for p,digest in artifacts.items() if digest is None]
    if missing:rows.append(dict(id='all_named_artifacts_present',status='INCOMPLETE',missing=missing))
    else:rows.append(dict(id='all_named_artifacts_present',status='VERIFIED_PRESENCE_AND_HASH_ONLY',scope='Content acceptance is separate from existence.'))
    draft_markers={str(ROOT/'docs'/name):marker for name,marker in [
        ('TRACKVGGT_FINAL_REPORT.md','状态：IN_PROGRESS'),('TRACKVGGT_REVIEWER_AUDIT.md','INTERIM_REVIEWER_AUDIT'),
        ('TRACKVGGT_RUNTIME_PROTOCOL.md','ACTUAL_PREFIX_GATE_PASSED_FULL_REPLAYS_RUNNING'),
        ('TRACKVGGT_SYSTEM_RESULTS.md','ACTUAL_NATIVE_SYSTEM_PROGRESS_SNAPSHOT'),
        ('TRACKVGGT_REQUIREMENT_AUDIT.md','最终需求验收（进行中）')]
        if marker in (ROOT/'docs'/name).read_text()}
    if draft_markers:rows.append(dict(id='final_document_closure',status='INCOMPLETE',markers=draft_markers))
    pending=[r['id'] for r in rows if r['status']=='INCOMPLETE']
    proof=dict(status='INCOMPLETE_REQUIREMENT_AUDIT' if pending else 'READY_FOR_FINAL_MANUAL_NEGATIVE_SCOPE_REVIEW',
        rows=rows,pending_requirements=pending,original_request_sha256=sha256(request),named_artifact_receipts=artifacts,evidence_receipts=bound,
        conditional_nonexecution_is_not_a_trained_result=True,overall_goal_complete=False,
        remaining_review='A human-readable scope audit must still confirm actual test coverage, raw sources, limitations, all full-run evaluation contracts and publication. This script does not achieve the goal or imply scientific success.',
        scientific_success=False,teacher_success=False,student_success=False,paper_ready=False,provenance=provenance())
    save_json(ROOT/'results/trackvggt_requirement_completion_audit_v1.json',proof)
    print(json.dumps(dict(status=proof['status'],requirements=len(rows),pending=pending)))


if __name__=='__main__':main()
