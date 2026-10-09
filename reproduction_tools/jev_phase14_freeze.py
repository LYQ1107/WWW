"""Freeze facts, labels, budgets and scientific gates before new outcomes."""
from jev_phase14_common import *

def main():
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() == BASE
    assert not (REPORTS / 'PREREGISTRATION.json').exists()
    old = ROOT / 'reports/JEV_PHASE13'
    names = ['docs/JEV_PHASE13_FINAL_RESEARCH_REPORT.md'] + [
        'reports/JEV_PHASE13/' + n + '.json' for n in ['FINAL_GO_NO_GO', 'TRAINING_PROTOCOL',
        'DENSE_DATASET_MANIFEST', 'CANDIDATE_RECALL', 'LABEL_AUDIT', 'ARCHITECTURE_ABLATION',
        'ON_POLICY_TRAINING', 'ONPOLICY_COMPARISON', 'OFFICIAL_MATLAB_PRIMARY', 'EFFICIENCY']]
    original = read(old / 'BASELINE_COMPARISON.json')
    formal = original['strict_seed_summary']
    own = read(old / 'ONPOLICY_COMPARISON.json')['strict_seed_summary']
    evidence = dict(status='FROZEN', starting_commit=BASE,
        protected_report_SHA256={p: sha(ROOT / p) for p in names},
        original_GMT_HOTA=original['GMT_Original_separate_full_system']['GMT_OFF_metrics']['strict_online']['HOTA'],
        cosine_HOTA=69.0852428815201, cosine_IDSW=82,
        Full={m: formal['full'][m]['mean'] for m in ['HOTA', 'AssA', 'IDF1', 'IDSW']},
        Fixed_Question={m: formal['fixed_question'][m]['mean'] for m in ['HOTA', 'IDSW']},
        Set_Transformer_HOTA=formal['set_transformer']['HOTA']['mean'],
        Full_onpolicy24k={m: own['full'][m]['mean'] for m in ['HOTA', 'IDSW']},
        updates={'formal': 20000, 'onpolicy_total': 24000},
        native_GTA_free='PASS', full_lifecycle_supervision='NOT_RUN', heldout='SEALED',
        Stage1_pretraining_exposed_all24_TRAIN=True,
        historical_failed_gates=['G5', 'G7', 'G8', 'G9'],
        checkpoint_audit='PENDING_ACTUAL_HASH_AUDIT',
        storage='Phase XIV sparse worktree excludes old large evidence copies; Git objects/original runtime retained')
    save(REPORTS / 'PHASE13_FROZEN_EVIDENCE.json', evidence)
    protocol = dict(status='FROZEN_BEFORE_PHASE14_DIAGNOSTICS', starting_commit=BASE,
        frozen_evidence_SHA256=sha(REPORTS / 'PHASE13_FROZEN_EVIDENCE.json'),
        TRAIN=TRAIN, development=DEV, heldout='SEALED', Full24=False, official_TEST=False,
        seed_list=[20261008, 20261009, 20261010],
        diagnostics={'methods': ['full20k', 'fixed_question20k', 'set_transformer20k', 'cosine', 'full24k'],
                     'all_three_seeds_when_available': True, 'no_new_full_video_replay_before_saved_artifact_audit': True,
                     'paired_prefix_frames': [40, 128, 224], 'prefix_videos': TRAIN,
                     'factorial_main_effects': ['recent', 'global_mean', 'own_mean', 'other_mean', 'camera_meta',
                         'time', 'geometry', 'counts', 'shared_state'],
                     'factorial_pairs': [['own_mean', 'other_mean'], ['global_mean', 'counts'], ['other_mean', 'camera_meta']],
                     'feature_deletion_claim': 'inference dependency only, not retrained causal architecture advantage'},
        labels={'UNKNOWN': 'never a negative or an absent-identity certificate',
                'Q1': 'multi-positive certified natural lawful identity choices',
                'Q2_positive': 'at least one past-only certified pure matching lawful identity',
                'Q2_negative': 'all lawful histories certifiable and none matches; or explicitly versioned TRAIN-only paired withholding',
                'Q3': 'candidate past observed GT support: >=3 known observations, >=0.8 GT coverage; pure=one GT; contaminated=>=2 GT with >=2 observations each',
                'Q3_scope': 'observed-history purity, never proof of complete biometric purity',
                'natural_Q2_qualification': {'positive_rows_min': 100, 'negative_rows_min': 20, 'videos_min': 2},
                'intervened_Q2_qualification': {'paired_rows_min': 500, 'videos_min': 3},
                'Q3_qualification': {'pure_candidate_examples_min': 100, 'contaminated_candidate_examples_min': 50, 'videos_min': 2},
                'withholding': 'remove ALL certified target options for a certified row; if any UNKNOWN options remain, absence remains UNKNOWN; no fake NEW/REACT'},
        pilot={'updates': 1000, 'seed': 20261009, 'batch_payloads': 4, 'lr': 0.0001, 'weight_decay': 0.01,
               'sampling': 'TRAIN identity/time blocks; source-paired presence/withholding balanced with natural data',
               'train_audit_split': 'deterministic video-local 64-frame blocks; frame//64 %5==4 reserved, never gradient training',
               'models': ['full', 'fixed_question', 'multi_question', 'set_transformer', 'motip', 'shared_mlp'],
               'loss_controls': ['standard_choice', 'legacy_structured', 'cost_sensitive', 'availability_joint'],
               'normal_retention_min': 0.90, 'absence_balanced_accuracy_min': 0.60,
               'finite_gradient_required': True, 'UNKNOWN_softmax_exclusion_control_required': True},
        formal={'max_architectures': 4, 'updates': 20000, 'seeds': [20261008, 20261009, 20261010],
                'checkpoint_selection': 'LAST only, never best-development-HOTA',
                'qualification': 'P1 complete, label qualifications, pilot finite and action semantics passing; architectures chosen from preregistered fixed/multi/set/shared_mlp order',
                'parameter_ratio_tolerance': [0.9, 1.1], 'common_evidence_labels_solver_budget': True},
        onpolicy={'rounds_max': 1, 'extra_updates': 4000, 'same_extra_budget_offpolicy_control': True,
                  'TRAIN_frames_per_video': 256, 'identity_time_block_balanced': True},
        costs={'wrong_existing': 1., 'false_merge': 2., 'false_split': 2., 'false_birth': 2.,
               'wrong_stale_recovery': 2., 'correct_continuation': 0., 'UNKNOWN': None,
               'freeze_basis': 'symmetric merge/split prior, risk severity before new outcomes; no dev-HOTA tuning'},
        loss_weights={'assignment': 0.25, 'brier': 0.1, 'availability': 0.5, 'trust': 0.25},
        scientific_gates={'G0': 'actual GTA-throw native/resume/recycling parity', 'G1': 'artifact-backed event taxonomy and exact CLEAR IDSW reconstruction',
          'G2': 'independent Q2/Q3 labeled support and held TRAIN-block learnability, not gradient alone',
          'G3': 'withholding only DEFER, never NEW or recovery labels; lifecycle learned heads remain disabled unless genuinely qualified',
          'G4': 'HOTA/AssA/IDF1 versus frozen controls; IDSW<=102.5; no increase in certified merges/splits/births relative fixed20k',
          'G5': 'same-supervision multi-question beats both fixed and set by>=0.5 HOTA/AssA mean with nonnegative HOTA per seed and stable action risk; exploratory, not statistical significance',
          'G6': 'actual native MATLAB CVIDF1/CVMA and cross-camera attribution',
          'G7': 'Stage2 p95<=10ms AND full image-to-tracks>=25 sceneFPS, repeat3 same-hardware trials; old G8 remains FAIL',
          'G8': 'unseen frontend scene exposure verified; else independent-full-system claim NO_GO'},
        independent_validation={'scheme_A': 'inventory server external TRAIN data first; freeze official protocol before outcomes',
                                'scheme_B': 'clean Stage1 only after generic initializer/data/config/budget eligibility; no historic checkpoint reuse as clean'},
        incomplete_status='NOT_RUN with exact dependency/reason; never invented metrics')
    save(REPORTS / 'PREREGISTRATION.json', protocol)
    OUT.mkdir(parents=True, exist_ok=True)
    print('PHASE14_FROZEN', BASE, sha(REPORTS / 'PREREGISTRATION.json'), flush=True)

if __name__ == '__main__':
    main()
