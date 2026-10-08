"""Read-only scientific-delivery verification; never run/open heldout or TEST.

Checks original artifacts by SHA, frozen sampling and explicit failed gates.
It verifies delivery completeness, not successful JEV architecture learning.
"""
import collections
import json
import subprocess
import time
from jev_phase9_common import *

DOCS = [
    'PHASE9_RESEARCH_GOAL.md', 'PHASE9_EXTERNAL_CODE_AUDIT.md',
    'PHASE9_METHOD_DIFFERENTIATION.md', 'PHASE9_SUPPLEMENTAL_CAPTURE_PROTOCOL.md',
    'PHASE9_CAUSAL_DATASET_REPORT.md', 'PHASE9_NATIVE_CANDIDATE_INTERFACE.md',
    'PHASE9_ARCHITECTURE_DESIGN.md', 'PHASE9_FINAL_RESEARCH_REPORT.md',
    'PHASE9_MODEL_COMPARISON.md',
]
JSONS = [
    'EXTERNAL_REPOSITORY_MANIFEST.json', 'SUPPLEMENTAL_CAPTURE_AUDIT.json',
    'PHASE9_SUPPLEMENTAL_CAPTURE_AUDIT.json', 'PHASE9_SUPPLEMENTAL_CAPTURE_PROTOCOL.json',
    'CAUSAL_DATASET_AUDIT.json', 'PHASE9_CAUSAL_DATASET_MANIFEST.json',
    'CANDIDATE_INTERFACE_TESTS.json', 'PHASE9_CANDIDATE_INTERFACE_TESTS.json',
    'NATIVE_COMMIT_PARITY.json', 'EXISTING_CHOICE_PARITY.json',
    'CANDIDATE_PRODUCTION_REGRESSION.json', 'PHASE9_CANDIDATE_SUBMIT_PARITY.json',
    'FEATURE_BRIDGE_PARITY.json', 'FEATURE_BRIDGE_RESTORATION_PARITY.json',
    'MEMORY_REPRESENTATION_AUDIT.json', 'GALLERY_FAILURE_ATTRIBUTION.json',
    'TINY_OVERFIT.json', 'PHASE9_TINY_OVERFIT.json', 'BASELINE_COMPARISON.json',
    'ARCHITECTURE_ABLATION.json', 'HELDOUT_RESULTS.json', 'FINAL_GO_NO_GO.json',
    'PHASE9_TRAINING_ABLATION.json', 'PHASE9_DATA_SCALING.json',
    'PHASE9_CAPACITY_ABLATION.json', 'PHASE9_SUPERVISION_ABLATION.json',
    'PHASE9_NORMALIZATION_ABLATION.json',
]
NOT_RUN = [n for n in JSONS if n in (
    'TINY_OVERFIT.json', 'PHASE9_TINY_OVERFIT.json', 'BASELINE_COMPARISON.json',
    'ARCHITECTURE_ABLATION.json', 'HELDOUT_RESULTS.json',
    'PHASE9_TRAINING_ABLATION.json', 'PHASE9_DATA_SCALING.json',
    'PHASE9_CAPACITY_ABLATION.json', 'PHASE9_SUPERVISION_ABLATION.json',
    'PHASE9_NORMALIZATION_ABLATION.json')]


def main():
    began = time.monotonic()
    protect()
    docs = [ROOT / 'docs' / n for n in DOCS]
    files = [REPORTS / n for n in JSONS]
    assert all(p.is_file() and p.stat().st_size > 0 for p in docs + files)
    data = {n: json.loads((REPORTS / n).read_text()) for n in JSONS}
    for name in NOT_RUN:
        d = data[name]
        assert d['status'] == 'NOT_RUN' and d['metrics'] is None
        assert d['trained_models'] == 0 and d['trained_checkpoints'] == 0
        assert d['heldout_sealed'] and not d['Full24'] and not d['official_TEST']
    assert not data['HELDOUT_RESULTS.json']['controller_heldout_predictions_opened']
    gates = data['FINAL_GO_NO_GO.json']
    assert gates['status'] == 'STOP_BLOCKED_DEPLOYMENT_CONTRACT'
    assert gates['gates']['G_NATIVE']['status'] == 'BLOCKED_DEPLOYMENT_CONTRACT'
    assert not gates['gates']['G_DATA']['training_authorized']
    assert not gates['JEV_architecture_rejected'] and gates['models_trained'] == 0
    assert len(gates['seven_answers']) == 7 and gates['heldout_sealed']
    assert not gates['Full24_authorized'] and not gates['official_TEST_authorized']

    protocol = data['PHASE9_SUPPLEMENTAL_CAPTURE_PROTOCOL.json']
    assert sha(REPORTS / 'PHASE9_SUPPLEMENTAL_CAPTURE_PROTOCOL.json') == '5d8504af27a351346e683aeeec1bb67678a71249b213a181bd6bb020d788a966'
    assert protocol['train'] == TRAIN and protocol['validation'] == VAL
    selected = protocol['selected_events']
    assert len(selected) == 131
    supp = [r for r in selected if r['distribution'] == 'SUPPLEMENTAL_CORRECTIVE']
    assert len(supp) == 48
    assert collections.Counter(r['key'][0] for r in supp) == {17: 17, 18: 14, 19: 17}
    row_keys = {(tuple(r['key']), r['row']) for r in selected}
    assert len(row_keys) == 131
    assert all(0 < r['sampling_probability'] <= 1 for r in selected)
    support = data['SUPPLEMENTAL_CAPTURE_AUDIT.json']
    assert support['true_verified_corrections'] == {'train': 74, 'validation': 59, 'supplemental': 46}
    assert support['phase8_original_gate_permanent_FAIL']
    assert support['new_native_audited_events'] == 67 and support['new_executed_branches'] == 354
    assert support['captured_unique_rows'] == 131
    assert support['verified_independent_groups'] == {'train': 17, 'validation': 9}
    assert all(support['gate_checks'].values())
    audit = data['CAUSAL_DATASET_AUDIT.json']
    assert len(audit['events']) == 162 and audit['unexecuted_candidate_utility'] is None
    assert audit['final_qualification']['status'] == 'BLOCKED_DEPLOYMENT_CONTRACT'
    assert audit['formal_model_training'] == 'NOT_RUN'
    new_events = [r for r in audit['events'] if r['sampling_records'] is not None]
    assert len(new_events) == 67
    assert all(r['CONTROL_KEEP_complete_state_PASS'] and r['CONTROL_factual_ID_PASS'] for r in audit['events'])
    assert {(tuple(r['key']), r['row']) for r in new_events} == {(tuple(r['key']), r['row']) for r in selected if r['counterfactual_audit']}

    tests = data['CANDIDATE_INTERFACE_TESTS.json']
    assert tests['status'] == 'PASS' and tests['tests'] == 20
    assert tests['failures'] == tests['errors'] == 0
    assert tests['binding']['gtr_rcnn_sha256'] == sha(ROOT / 'gtr/modeling/meta_arch/gtr_rcnn.py')
    full = data['NATIVE_COMMIT_PARITY.json']
    assert full['full_video_complete'] and full['tested_frames'] == 1200
    assert full['all_native_commit_fields_and_RNG_SHA_parity'] and full['actual_postprocessed_output_SHA_parity']
    regression = data['CANDIDATE_PRODUCTION_REGRESSION.json']
    assert regression['status'] == 'PASS_BOUNDED_PRODUCTION_CONTRACT'
    assert regression['binding']['gtr_rcnn_sha256'] == sha(ROOT / 'gtr/modeling/meta_arch/gtr_rcnn.py')
    assert regression['existing_values_deployed_changed_commits'] == 13
    assert regression['semantic_NEW_deployed_changed_commits'] == 15
    assert regression['next_frame_live_score_matrix_changes'] == 14
    assert data['FEATURE_BRIDGE_PARITY.json']['status'] == 'FAIL_OFFLINE_ONLINE_FEATURE_BRIDGE'
    restored = data['FEATURE_BRIDGE_RESTORATION_PARITY.json']
    assert restored['status'] == 'PASS' and not restored['formal_training_authorized_by_this_test']
    memory = data['MEMORY_REPRESENTATION_AUDIT.json']
    assert memory['snapshots'] == 221 and len(memory['failures']) == 94
    assert memory['per_track_instances_checked'] == 39536

    # Deduplicate artifact hashes across report references, but do not elide
    # any negative branch. No raw artifact is staged or uploaded.
    artifacts = {}
    def register(record):
        path, expected = str(record['path']), record['sha256']
        assert path not in artifacts or artifacts[path] == expected
        artifacts[path] = expected
    manifest = data['PHASE9_CAUSAL_DATASET_MANIFEST.json']
    assert manifest['audit_sha256'] == sha(REPORTS / 'CAUSAL_DATASET_AUDIT.json')
    assert manifest['raw_artifacts'] == audit['raw_artifacts']
    assert len(manifest['raw_artifacts']) == 845
    for record in manifest['raw_artifacts'] + memory['records'] + protocol['observational_sources'] + support['capture_manifests']:
        register(record)
    for name in ('NATIVE_COMMIT_PARITY.json', 'EXISTING_CHOICE_PARITY.json', 'CANDIDATE_PRODUCTION_REGRESSION.json'):
        for record in data[name]['trace_artifacts']:
            register(record)
    for name in ('FEATURE_BRIDGE_PARITY.json', 'FEATURE_BRIDGE_RESTORATION_PARITY.json'):
        for record in data[name]['raw_artifacts']:
            register(record)
    register(restored['final_qualification']['original_report'])
    ext = data['EXTERNAL_REPOSITORY_MANIFEST.json']
    assert len(ext['repositories']) == 11
    external_count = 0
    for repository in ext['repositories']:
        assert len(repository['commit']) == 40 and not repository['missing_requested_paths']
        assert not repository['upstream_source_copied'] and not repository['external_environment_installed']
        for record in repository['files']:
            register({'path': str(Path(repository['local_root']) / record['path']), 'sha256': record['sha256']})
            external_count += 1
    for i, (path, expected) in enumerate(artifacts.items(), 1):
        assert sha(path) == expected, path
        if i % 200 == 0:
            print('ARTIFACT_SHA_VERIFIED', i, '/', len(artifacts), flush=True)

    changed = subprocess.check_output(['git', 'diff', '--name-only', BASE, 'HEAD'], cwd=ROOT, text=True).splitlines()
    changed += subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).splitlines()
    compact_bytes = 0
    for entry in changed:
        name = entry[3:] if entry[:2] in (' M', '??', 'A ', 'M ') else entry
        p = ROOT / name
        assert name.startswith(('docs/PHASE9_', 'reports/JEV_PHASE9/', 'reproduction_tools/')) or name in ('gtr/config.py', 'gtr/modeling/meta_arch/gtr_rcnn.py') or name.startswith('gtr/modeling/jev_candidate'), name
        assert p.suffix not in ('.pth', '.pt', '.mp4', '.npz', '.gz'), name
        if p.is_file():
            assert p.stat().st_size < 10 * 1024 * 1024, name
            compact_bytes += p.stat().st_size
    # Count each changed tracked path once (status adds already committed paths).
    all_paths = set(subprocess.check_output(['git', 'diff', '--name-only', BASE, 'HEAD'], cwd=ROOT, text=True).splitlines())
    all_paths.update(x[3:] for x in subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).splitlines())
    compact_bytes = sum((ROOT / n).stat().st_size for n in all_paths if (ROOT / n).is_file())
    result = {
        'status': 'PASS_DELIVERY_WITH_RESEARCH_GATE_BLOCKED', 'binding': binding(),
        'required_docs': len(DOCS), 'required_reports': len(JSONS),
        'raw_artifacts_SHA_verified': len(artifacts), 'external_source_license_files': external_count,
        'frozen_protocol_SHA_verified': True, 'all_selected_rows_and_negative_branches_retained': True,
        'source_matches_latest_bounded_production_regression': True,
        'original_refs_and_B2_protected': True, 'phase8_FAIL_unchanged': True,
        'heldout_sealed': True, 'Full24': False, 'official_TEST': False,
        'new_trained_models': 0, 'historical_checkpoint_deletions': 0,
        'compact_changed_file_bytes': compact_bytes, 'elapsed_seconds': time.monotonic() - began,
        'research_gate': 'BLOCKED_DEPLOYMENT_CONTRACT',
        'interpretation': 'PASS verifies truthful complete delivery, not candidate learning or architecture success.'}
    save(REPORTS / 'DELIVERY_VERIFICATION.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'binding'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
