"""Audit original evidence and preregister bounded research before new outcomes."""
import collections
import time
from jev_phase15_common import *


def main():
    protect()
    assert not (REPORTS / 'PHASE14_FROZEN_EVIDENCE.json').exists(), 'Never overwrite frozen evidence'
    storage_guard()
    OUT.mkdir(parents=True, exist_ok=True)
    begin = time.monotonic()
    paths = ['docs/JEV_PHASE13_FINAL_RESEARCH_REPORT.md', 'docs/JEV_PHASE14_FINAL_RESEARCH_REPORT.md']
    paths += [str(p.relative_to(ROOT)) for folder in ['JEV_PHASE13', 'JEV_PHASE14']
              for p in sorted((ROOT / 'reports' / folder).glob('*.json'))]
    protected = {}
    for rel in paths:
        actual = sha(ROOT / rel)
        original = subprocess.check_output(['git', 'show', BASE + ':' + rel], cwd=ROOT)
        assert hashlib.sha256(original).hexdigest() == actual, rel
        protected[rel] = actual
    historical = read(ROOT / 'reports/JEV_PHASE14/HISTORICAL_CHECKPOINT_AUDIT.json')
    physical = {}
    checkpoints = []
    for index, entry in enumerate(historical['models']):
        p = Path(entry['path']); s = p.stat(); key = (s.st_dev, s.st_ino)
        assert s.st_size == entry['bytes'], str(p)
        if key not in physical:
            physical[key] = sha(p)
        assert physical[key] == entry['actual_SHA256'], str(p)
        checkpoints.append({'path': str(p), 'SHA256': physical[key], 'bytes': s.st_size})
        if (index + 1) % 32 == 0:
            print('PHASE15_FROZEN_CHECKPOINTS', index + 1, len(historical['models']), flush=True)
            save(OUT / 'freeze/PROGRESS.json', {'status': 'RUNNING', 'checkpoints_verified': index + 1,
                 'total': len(historical['models']), 'seconds': time.monotonic() - begin})
    new_models = read(ROOT / 'reports/JEV_PHASE14/FORMAL_TRAINING_RESULTS.json')['cases']
    for item in new_models:
        ck = item['checkpoint']; p = Path(ck['path']); s = p.stat(); key = (s.st_dev, s.st_ino)
        if key not in physical: physical[key] = sha(p)
        assert physical[key] == ck['SHA256'], str(p)
        checkpoints.append(dict(ck, bytes=s.st_size, variant=item['variant'], seed=item['seed']))
    manifest = ROOT / 'reports/JEV_PHASE14/SOURCE_INTERFACE_MANIFEST.json'
    source_checks = []
    for entry in read(manifest)['sources']:
        path = Path(entry['path'])
        assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=path, text=True).strip() == entry['commit']
        for rel, digest in entry['files'].items(): assert sha(path / rel) == digest, str(path / rel)
        source_checks.append({'path': str(path), 'commit': entry['commit'], 'verified_files': len(entry['files'])})
    ann = sha(ANNOTATIONS)
    assert ann == read(ROOT / 'reports/JEV_PHASE14/ONLINE_VALIDATION.json')['annotation_SHA256']
    frozen = {'status': 'PASS', 'starting_commit': BASE, 'binding': binding(scope='Read-only preservation audit'),
              'protected_report_SHA256': protected, 'historical_models_verified': len(historical['models']),
              'Phase14_formal_models_verified': len(new_models), 'immutable_actor_sources': source_checks,
              'annotation_SHA256': ann, 'seconds': time.monotonic() - begin,
              'historical_scientific_status': 'NO_GO', 'GT_scope': 'offline annotation metadata only; never actor inputs',
              'heldout_20_21_22': 'SEALED', 'official_TEST': False, 'Full24': False,
              'asset_policy': 'Prior paths, failed artifacts, MATLAB outputs, checkpoints and native prefixes retained',
              'worktree_storage': 'Independent local index skips irrelevant old report/fixture files; original Git tree preserved'}
    save(REPORTS / 'PHASE14_FROZEN_EVIDENCE.json', frozen)
    save(REPORTS / 'SOURCE_AND_CHECKPOINT_MANIFEST.json', {'status': 'PASS', 'binding': binding(),
         'checkpoints': checkpoints, 'original_delivery_audit': ref(ROOT / 'reports/JEV_PHASE14/DELIVERY_AUDIT.json'),
         'frozen_evidence': ref(REPORTS / 'PHASE14_FROZEN_EVIDENCE.json')})
    protocol = {
        'status': 'FROZEN_BEFORE_PHASE15_RESULTS', 'starting_commit': BASE, 'TRAIN': TRAIN,
        'development': DEV, 'seeds': SEEDS, 'heldout': 'SEALED', 'Full24': False, 'official_TEST': False,
        'no_development_GT_in_training': True, 'no_development_metric_hyperparameter_selection': True,
        'Phase1': {'primary': 'multi_question seed20261009 video17/18/19',
                   'controls': ['fixed_question seed20261009', 'set_transformer seed20261009',
                                'multi_question seed20261008', 'multi_question seed20261010'],
                   'CLEAR_switch_counts_must_equal_saved_official': True,
                   'long_training_before_failure_reconstruction': False},
        'counterfactual': {'horizons': [8, 16, 32], 'max_H64_extension_prefixes': 8,
            'max_DEV_prefixes_per_video': 8, 'max_TRAIN_prefixes_per_video': 6,
            'sampling': 'deterministic chronological strata: pure hop, polluted switch, corrective risk, false birth/unknown; retain no-safe-option events',
            'actions': ['KEEP_PREVIOUS_COMMITTED_ID', 'SELECT_JEV_BEST_ID',
                        'SELECT_ANOTHER_CERTIFIED_CANDIDATE', 'DEFER', 'NATIVE_CONSERVATIVE_FALLBACK'],
            'continuations': ['pi_multi_frozen20k', 'pi_fixed_frozen20k'],
            'same_RNG_and_native_start': True, 'oracle_choices': 'offline research upper bound only',
            'candidate_fabrication_or_historical_rewrites': False,
            'gate': 'At least8 independent certified native prefixes with fewer future identity changes, no increase in certified false merge or false birth; report legal/safe support fraction and all adverse outcomes'},
        'training': {'Tiny_updates': 128, 'Pilot_updates': 1500, 'Pilot_seed': 20261009,
            'max_evidence_driven_versions': 3, 'formal_updates': 20000, 'checkpoint': 'LAST',
            'onpolicy_rounds': 1, 'onpolicy_extra_updates': 4000,
            'batch': 4, 'optimizer': 'AdamW', 'learning_rate': 0.0003, 'weight_decay': 0.01,
            'arms': ['A_original_multi_original_loss', 'B_multi_ordinary_continuity',
                     'C_multi_commitment_only', 'D_fixed_same_commitment',
                     'E_set_same_commitment', 'F_full_WHO_availability_trust_commitment'],
            'loss_weights': {'WHO': 1.0, 'Availability': 0.5, 'Trust': 0.5,
                            'Commitment': 1.0, 'JointAssignment': 0.2, 'UnsafeSwitchRisk': 0.5},
            'reserved_TRAIN_blocks': 'reuse frozen temporal reserved block split; no reserved examples receive gradients',
            'pilot_gate': {'certified_commitment_accuracy_min': 0.80, 'necessary_correction_accuracy_min': 0.75,
                          'safe_continuation_accuracy_min': 0.90, 'required_nonzero_both_label_classes': True,
                          'native_legality_and_restore_tests': 'PASS',
                          'paired_native_false_merge_and_birth': 'no worsening; all cases retained'},
            'formal_requires_pilot': True,
            'causal_architecture_ablation': ['F_no_dynamic_Q4', 'F_no_commitment_option_reader'],
            'extra_controls_run_only_after_F_pilot_qualification': True,
            'no_blind_epoch_layer_or_random_hyperparameter_expansion': True},
        'scientific_gates': {
            'GO_TRACKING': {'native_contracts': 'ALL_PASS', 'IDSW_mean_max': 195.8335,
                'HOTA_vs_strongest_same_supervision_control_min': 0.0,
                'AssA_vs_strongest_same_supervision_control_min': 0.0,
                'IDSW_each_seed_max': 250, 'CVIDF1_vs_Phase14_multi_tolerance_pp': -0.5,
                'false_merge_false_birth_vs_same_supervision_controls': 'no increase in seed-mean counts',
                'challenge_targets_only': {'HOTA': 75.637, 'IDSW': 102.5, 'CVIDF1': 88.546}},
            'GO_JEV_INDEPENDENT_VALUE': {'requires': 'GO_TRACKING',
                'paired_HOTA_mean_vs_D_and_E_min_pp': 0.5,
                'paired_HOTA_each_seed_vs_D_and_E_min_pp': 0.0,
                'paired_IDSW_vs_D_and_E': 'no seed worse',
                'dynamic_Q4_and_option_reader': 'positive controlled retraining contribution in HOTA or IDSW without merge/birth harm',
                'three_seed_scope': 'exploratory fixed effect gate; no population statistical significance claim'},
            'GO_DEPLOYMENT': {'total_Stage2_p95_ms_max': 10, 'full_two_camera_scene_FPS_min': 25,
                              'cache_FPS_is_not_full_FPS': True}},
        'generalization': {'development': 'historically reused and Stage1 exposed',
            'external': 'separate scene transfer only unless initialization qualification succeeds',
            'clean_frontend_retraining': 'conditional on verified data, initializer and dedicated budget'},
        'storage': {'reuse_prior_cache_no_copy': True, 'home_free_reserve_GiB': 30,
            'new_runtime_target_max_GiB': 32, 'large_weights_prefixes_and_trajectories_on_server_only': True},
        'completion_policy': {'task_execution': 'distinct from scientific success',
            'scientific_failure': 'retain exact events, causal evidence and remaining barrier',
            'unrun_metrics': None, 'stop_after_first_NO_GO': False}}
    save(REPORTS / 'PREREGISTRATION.json', protocol)
    save(REPORTS / 'FINAL_GOAL.json', {'status': 'ACTIVE', 'scientific_status': 'NOT_YET_ESTABLISHED',
         'binding': binding(), 'request': ref(ROOT / 'docs/JEV_PHASE15_RESEARCH_GOAL.md'),
         'preregistration': ref(REPORTS / 'PREREGISTRATION.json')})
    save(OUT / 'freeze/PROGRESS.json', {'status': 'PASS', 'historical_models': len(historical['models']),
         'Phase14_models': len(new_models), 'protected_reports': len(protected), 'seconds': time.monotonic() - begin})
    print('PHASE15_FROZEN_AUDIT_PASS', len(checkpoints), len(protected), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        failure('freeze', e)
        raise
