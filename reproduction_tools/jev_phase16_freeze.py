"""Freeze Phase XV facts and sequential Phase XVI budgets before experiments."""
from jev_phase16_common import *

def main():
    protect();OUT.mkdir(parents=True,exist_ok=True);REPORTS.mkdir(parents=True,exist_ok=True);storage_guard()
    assert not (REPORTS/'PREREGISTRATION.json').exists()
    original=XVROOT/'reports/JEV_PHASE15'
    final=read(original/'FINAL_GO_NO_GO.json')
    assert final['status']=='SCIENTIFIC_NO_GO'
    assert not any(final[k] for k in ['GO_TRACKING','GO_JEV_INDEPENDENT_VALUE','GO_DEPLOYMENT'])
    pilots=read(original/'PILOT_RESULTS.json');assert len(pilots['versions'])==3 and not pilots['formal_qualified']
    for report in ['FORMAL_TRAINING_RESULTS','FAIR_BASELINE_RESULTS']:
        assert read(original/(report+'.json'))['metrics'] is None
    preserved={str(XVROOT/p):h for p,h in read(original/'PHASE14_FROZEN_EVIDENCE.json')['protected_report_SHA256'].items()}
    preserved.update({str(p):sha(p) for p in original.glob('*.json')})
    checkpoints={}
    def collect(value):
        if isinstance(value,dict):
            p=value.get('path');digest=value.get('SHA256')
            if p and digest and p.endswith(('.pth','.pt')) and '/training_full_payload_' in p:
                checkpoints[p]=digest
            for v in value.values():collect(v)
        elif isinstance(value,list):
            for v in value:collect(v)
    for name in ['TINY_RESULTS','PILOT_RESULTS']:collect(read(original/(name+'.json')))
    for path in XV.glob('training*/**/*FROZEN.pth'):checkpoints[str(path)]=sha(path)
    for p,h in checkpoints.items():assert sha(p)==h
    save(REPORTS/'FROZEN_PRIOR_EVIDENCE.json',dict(status='PASS',starting_commit=BASE,
        prior_report_SHA256=preserved,phase15_model_checkpoint_SHA256=checkpoints,
        previous_full_332_checkpoint_hash_audit=ref(original/'FINAL_PRIOR_INTEGRITY.json'),
        previous_checkpoint_manifest=ref(original/'SOURCE_AND_CHECKPOINT_MANIFEST.json'),
        phase15_full_video_diagnostic=final['diagnostic_metrics'],
        CVIDF1=final['diagnostic_CVIDF1'],CVMA=final['diagnostic_CVMA'],
        pilot_correction=[v['assessment']['necessary_correction_accuracy'] for v in pilots['versions']],
        pilot_adverse_horizon_events=[len(v['adverse_native_cases']) for v in pilots['versions']],
        all_existing_results_remain_read_only=True,formal20k_and_matched4k_were_NOT_RUN=True))
    protocol=dict(status='FROZEN_BEFORE_PHASE16_RESULTS',starting_commit=BASE,
        order=['P0_actual_history_and_candidate_audit','P1_if_summary_loss_supported','P2_rule_memory_admission',
            'P3_frozen_model_native_factorial','P4_and_training_only_if_safely_identifiable'],
        frozen_primary_policy='Phase XV v3 F_full LAST1500 seed20261009',
        P0=dict(full_replays=[dict(video=v,policy=p) for p in ['original','v1','v2','v3'] for v in TRAIN]+
            [dict(video=v,policy='v3') for v in DEV],
            native_window_replays='all24 frozen TRAIN prefixes under each of3 XV checkpoints;72 H32 branches, all adverse windows retained',
            labels='post-decision offline IoU>=0.5; duplicate identity GT=>UNKNOWN; no GT inputs to tracker',
            high_risk='selected unanchored/mixed/wrong, DEFER, absent pure compatible support; all queries denominators retained',
            taxonomy='A-G multi-label mechanisms plus declared primary precedence; ambiguity cannot become wrong',
            identity_content_is_not_correct_Global_ID=True,
            correct_owner_certificate='first past moment with>=3 known observations,>=0.8 coverage and one GT; immutable afterwards; mixtures before certificate stay unanchored',
            summary_loss='actual target-containing ID has raw cosine>=0.75, four-summary cosine<0.75 or raw target-vs-foreign margin improves>=0.02',
            local_support='at least3 target observations; continuous pure-segment subcertificate separately disclosed',
            causal_summary_loss_claim='paired retrieval plus native intervention required; offline cosine gap alone is not causal proof',
            P1_eligibility='>=8 distinct TRAIN owner-anchored identity/camera/64-frame clusters with raw support and summary loss; report false activation and all F/G'),
        history=dict(no_new_network_capacity=True,slots=4,max_stored_tokens_per_ID=16,
            variants=['A_original_four','B_recent_four','C_temporal_segments','D_camera_segments','E_bounded_combination'],
            bounded_view_never_drops_ID_candidates=True,
            P1_retrieval_gate='>=2 percentage-point positive support improvement on eligible TRAIN queries and no>1pp certified-wrong activation increase;>=8 clusters'),
        memory=dict(policies=['always_write','recent_only','provisional','delayed_confirmation','bounded_quarantine'],
            raw_Gallery_hits_native_Bank_unchanged=True,confirmed_is_not_GT_purity=True,
            matched_raw_commit_log_and_trusted_evidence_view_separate=True,no_learned_WRITE_head=True),
        factorial=dict(arms=['A_original_decision_original_memory','B_original_decision_safe_evidence',
            'C_evidence_decision_original_memory','D_evidence_decision_safe_evidence'],
            horizons=[8,16,32],H64_max_prefixes=8,primary_TRAIN_prefixes=24,
            same_native_prefix_RNG_and_detector=True,no_future_history_synchronization=True,
            cluster='video+offline targetGT+camera+64-frame block; overlapping horizons/prefixes not independent samples',
            baseline_parity='A exact XV V3 native IDs before intervention; state metadata does not alter original input tensors',
            safety_gate='at least8 distinct strict TRAIN clusters benefit in certified recovery, owner errors or mixing; no strict prefix at any horizon worsens new mixing, wrong-owner duration or false_split_birth',
            pure_false_birth_proxy_and_recovery_birth_separate=True,
            TRAIN14_duplicate_GT='preserve strict failure, permissive isolated diagnostic only; not qualifying'),
        learning=dict(max_major_versions=3,Tiny_updates=128,Pilot_updates=1500,Pilot_seed=20261009,
            formal_updates=20000,seeds=SEEDS,matched_onpolicy_updates=4000,batch=4,optimizer='AdamW',lr=.0003,wd=.01,
            reserved_TRAIN_blocks='(frame//64)%5==4; no gradients or label tuning on DEV',
            label_gate='>=8 independent certified safe-recovery clusters and>=8 unsafe-recovery clusters, both train/reserved support; necessary-correction reserved>=8 clusters; normal continuation represented',
            Q5_without_qualified_supervision='frozen native rule only, never untrained action logits',
            Pilot_gate=dict(safe_continuation_min=.9,necessary_correction_recall_min=.75,
                recovery_precision_min=.9,both_action_label_classes_nonzero=True,native_factorial_safety='PASS'),
            fair_arms=['Frozen_XV','Frozen_evidence','Frozen_safe_memory','Frozen_both',
                'ordinary_Set_same_evidence_supervision','Fixed_same_evidence_supervision','new_evidence_JEV'],
            formal_and_controls_require_safe_Pilot=True,onpolicy4k_requires_safe_formal=True),
        gates=dict(GO_TRACKING=dict(IDSW_each_seed_max=250,IDSW_mean_max=195.8335,
            HOTA_AssA_vs_strongest_same_evidence='no worse',CVIDF1_vs_XV_tolerance_pp=-.5,
            wrong_owner_duration_mix_false_birth='no worse than matched controls',three_seeds_required=True),
            GO_JEV_INDEPENDENT_VALUE=dict(requires='GO_TRACKING',paired_HOTA_mean_gain_vs_Fixed_and_Set_pp=.5,
                each_seed_HOTA_gain_min=0,IDSW_no_seed_worse=True,
                controlled_Q4_option_reader_retraining='positive contribution without safety harm',
                scope='fixed exploratory effects, not population significance'),
            GO_DEPLOYMENT=dict(Stage2_p95_ms_max=10,two_camera_scene_FPS_min=25,actual_images=True)),
        data=dict(TRAIN=TRAIN,development=DEV,sealed=[20,21,22],official_TEST=False,Full24=False,
            Stage1_exposure='known all24; no absolute unseen full-system claim'),
        budget=dict(home_reserve_GiB=30,new_runtime_max_GiB=32,max_GPU_workers=6,
            reuse_all_prior_caches_weights_and_prefixes=True,new_large_artifacts_server_only=True),
        stop='three major versions unsafe or unobservable recovery; retain failures, explicit blocker, unrun metrics null')
    save(REPORTS/'PREREGISTRATION.json',protocol)
    goal=dict(status='ACTIVE_P0',scientific_status='NOT_YET_EVALUATED',binding=binding(evaluator='frozen source/report/hash inspection',
        scope='protocol registration, no new inference or training'),branch=BRANCH,versions_completed=[],
        protected_prior=ref(REPORTS/'FROZEN_PRIOR_EVIDENCE.json'))
    save(REPORTS/'FINAL_GOAL.json',goal)
    source=Path('/home/liuyeqiang/.codex/attachments/d1f058a5-24f8-4e2d-8340-cccb5e24c91d/Pasted text.txt')
    (ROOT/'docs/JEV_PHASE16_RESEARCH_GOAL.md').write_text(source.read_text())
    print('PHASE16_FROZEN',len(preserved),len(checkpoints),flush=True)

if __name__=='__main__':main()
