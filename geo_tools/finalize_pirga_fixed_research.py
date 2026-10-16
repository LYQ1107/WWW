"""Close the finite protocol only after real fixed endpoints and negative gates."""
import datetime,json,shutil
from pathlib import Path
from pirga_indexed_store import sha
from pirga_status import processes


def save(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


def main():
    repo=Path(__file__).resolve().parents[1];root=repo/'results/PIRGA_FULLTRAIN';runtime=Path('/data1/liuyeqiang/WWW_pirga_runtime_20261016')
    audit=json.loads((root/'EXECUTION_REQUIREMENTS_AUDIT.json').read_text())
    if not all(v['status']=='PASS_FIXED_FIT_AND_REPLAYS' for v in audit['fixed_families'].values()):raise ValueError('Fixed endpoints incomplete')
    if audit['current_fit_input_integrity']['status']!='PASS_FRESH_ALL11_STORES_AND_ACTOR_CACHE_HASHES':raise ValueError('Full inputs not reverified')
    if set(audit['canonical300_control']['verified_checkpoint_arms'])!={'C2','C3','C4'} or not audit['canonical300_control']['full_DEV_replay_complete']:raise ValueError('Paired canonical300 control incomplete')
    if processes(runtime):raise ValueError('Own registered worker processes still live; verify instead of terminating')
    gates=[]
    for epoch in [1,3,5]:
        path=root/f'MATCHED_ALL_CONTROL_GATE_EPOCH{epoch}.json';g=json.loads(path.read_text())
        if g['status']!='NO_PREDICTION_SPECIFIC_GAIN' or g['phase2_mechanism_gate_pass'] or not g['strong_ordinary_controls_complete_for_this_epoch']:raise ValueError('Negative stop not established; continue necessary research')
        legacy=json.loads((root/f'LEGACY_COVERAGE_COMPARISON_EPOCH{epoch}.json').read_text())
        if legacy['comparisons']['L1_C3_vs_C1']['net_original_repairs_minus_additional_harms']>0:raise ValueError('Legacy positive needs independent evaluation, not negative closure')
        diagnostic=root/'UNCERTAINTY_GATE'/f'epoch{epoch}/RESULTS.json';d=json.loads(diagnostic.read_text())
        if not d['weights_unchanged'] or d['training_updates']!=0 or not d['original_ON_actions_exact_full_scope'] or d['queries']!=14557:raise ValueError('Same-checkpoint diagnostic incomplete')
        gates.append(dict(epoch=epoch,status=g['status'],path=str(path.relative_to(repo)),sha256=sha(path),
            uncertainty_gate_score_changes=d['changed_score_edges'],uncertainty_gate_action_changes=d['primary_action_changes'],diagnostic_sha256=sha(diagnostic)))
    preserved=json.loads((root/'OLD_EVIDENCE_PRESERVATION_RECHECK.json').read_text())
    if preserved['status']!='PASS_PRE_PIRGA_PRESERVATION' or preserved['issues']:raise ValueError('Old artifacts not preserved')
    log=runtime/'logs/pirga_all_contracts_v17.log'
    if 'Ran 35 tests' not in log.read_text() or not log.read_text().rstrip().endswith('OK'):raise ValueError('Current contracts not passed')
    resource=json.loads((root/'RESOURCE_LEDGER.json').read_text());free=shutil.disk_usage(runtime).free
    if resource['old_ledgers_reset'] or resource['parent_aggregate_observed_plus_conservative_bound_GPU_hours']>resource['inherited_parent_cap'] or free<30*2**30:raise ValueError('Resource invariants not satisfied')
    now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    conditional=dict(learned_history_reliability='NOT_RUN_CONDITIONAL_NO_PREDICTIVE_GATE',
        D4_prediction_reliability='NOT_RUN_CONDITIONAL_NO_PREDICTIVE_GATE',D5_prediction_margin='NOT_RUN_CONDITIONAL_NO_PREDICTIVE_GATE',D6_full_PIRGA='NOT_RUN_CONDITIONAL_NO_PREDICTIVE_GATE',
        full_native_online='NOT_RUN_FAILED_SCIENTIFIC_ADMISSION',full24_OOF='NOT_RUN_CORE_NO_GO_SEPARATE_STAGE_NOT_ACTIVATED',full24_final_fit='NOT_RUN_NO_QUALIFIED_LOCKED_METHOD',
        second_seed_and_independent_validation='NOT_RUN_NO_QUALIFIED_CANDIDATE_NO_GENERALIZATION_CLAIM')
    decision=dict(status='STOP_PIRGA_NO_PREDICTION_SPECIFIC_GAIN',execution='ALL_REGISTERED_FIXED_EXPERIMENTS_COMPLETE',scientific_status='NO_PREDICTION_SPECIFIC_GAIN',
        observed_utc=now,scientific_success=False,paper_ready=False,online_admitted=False,matched_fixed_gates=gates,conditional_not_run=conditional,
        user_stop_instruction='如果新预测状态仍不能超越C1，保留负结果，不进入更复杂模块组合。',
        user_finite_stop_instruction='如果多个预注册机制都无法带来超出普通余弦和普通MLP的收益，就明确停止PIRGA，保留最强的有效基础方法。',
        retained_role='Original GMT and C1 remain references; all learned weights/results retained for audit. No newly validated superior online method or deployment.',
        local_signal_qualification='Legacy epoch5 C3 has one C1-extra local repair but loses four C1 repairs and adds uncertainty/error transitions; one person, no independent net value. Do not erase this event or promote it to success.',
        provenance=dict(instruction_sha256='8f639d3688a24a27a56523d9726985c2237194523bfce7a5e24a1fdf1a380187',old_preservation_sha256=sha(root/'OLD_EVIDENCE_PRESERVATION_RECHECK.json')),
        completion_scope='Finite preregistered execution plus explicit failed-gate stop. Quality improvement, full online metrics, cross-scene/seed generalization and WWW contribution are not established.')
    save(root/'FINAL_STAGE_GATE_DECISION.json',decision)
    master=json.loads((root/'MASTER_GOAL.json').read_text());master.update(status='COMPLETE_REGISTERED_EXECUTION_NEGATIVE_RESULT',scientific_status='NO_PREDICTION_SPECIFIC_GAIN',scientific_success=False,updated_utc=now)
    master['phases'].update({'3':'SIX_FIXED_RULES_COMPLETE_LEARNED_NOT_RUN_NO_PREDICTIVE_GATE','4':'ORDINARY_1_3_5_COMPLETE_D4_D5_D6_NOT_RUN_NO_PREDICTIVE_GATE',
        'online':conditional['full_native_online'],'24_scene_CV':conditional['full24_OOF'],'24_scene_final_fit':conditional['full24_final_fit']})
    master['goal_turn_audit'].update(this_turn='COMPLETE_FIXED_EXECUTION_NEGATIVE_STOP_PENDING_FINAL_MANUAL_AND_GITHUB_REVIEW',live_CPU_training=False,live_CPU_evaluation_or_waiting_queue=False,remaining_blocker=None)
    save(root/'MASTER_GOAL.json',master)
    g=json.loads((root/'GO_NO_GO.json').read_text());g.update(status='NO_PREDICTION_SPECIFIC_GAIN',local_scientific_signal='NO_PREDICTION_SPECIFIC_NET_GAIN_IN_FIXED_PROTOCOL',
        online='NOT_RUN_FAILED_SCIENTIFIC_ADMISSION',paper_ready=False,scientific_success=False,goal_remains_active=False,
        fixed_execution_complete=True,final_stage_decision_sha256=sha(root/'FINAL_STAGE_GATE_DECISION.json'),conditional_not_run=conditional,updated_utc=now,
        next_action='Stop this PIRGA expansion after all registered endpoints. Preserve negative and local positive evidence; no further tuning, online run or24-scene extension under this failed gate.')
    save(root/'GO_NO_GO.json',g)
    online=json.loads((root/'ONLINE_TRACKING_RESULTS.json').read_text());online.update(status='NOT_RUN_FAILED_SCIENTIFIC_ADMISSION',admission=False,updated_utc=now,
        reason='All fixed1/3/5 matched predictive gates failed. No new full native online run; no historical other-scene metrics pooled.',final_stage_decision_sha256=sha(root/'FINAL_STAGE_GATE_DECISION.json'));save(root/'ONLINE_TRACKING_RESULTS.json',online)
    resource.update(status='COMPLETE_REGISTERED_EXECUTION_WITHIN_EXISTING_GPU_CAP',updated_utc=now,current_free_target_bytes=free,own_registered_workers_complete=True,
        qualification_final='New observedGPU2.619336819870191h plus incident upper0.17777777777777778h; with inherited1.1772130608724223h totals3.9743276585203904h upper under4h. CPU fit walls overlap; no old budget reset.')
    save(root/'RESOURCE_LEDGER.json',resource)
    save(root/'CONDITIONAL_PHASE_DISPOSITION.json',dict(observed_utc=now,conditional_not_run=conditional,decision_sha256=sha(root/'FINAL_STAGE_GATE_DECISION.json'),untrained_is_not_ineffective=True))
    print(json.dumps(dict(status=decision['status'],registered_execution_complete=True,scientific_success=False,external_goal_completion_requires_manual_audit_and_verified_push=True)))


if __name__=='__main__':main()
