"""Read-only checks of published diagnostic scope, numbers and frozen sources."""
import csv
import json
from pathlib import Path
import subprocess
import numpy as np
from omega_common import ROOT, RUNTIME, provenance, save_new, sha256
from omega_modules import geometry_module
from omega_background_audit import distinct_matches


def main():
    report = json.loads((ROOT/'results/omega_experiment_manifest.json').read_text())
    assert report['scientific_goal_complete'] is False
    assert report['main_416'] == 'NOT_EXECUTED_BLOCKED_WEIGHT_ACCESS'
    for name, expected in report['published_artifacts'].items():
        assert sha256(ROOT/name) == expected, name
    cohort_path = ROOT/'manifests/omega_fixed_pair_cohort_v1.json'
    assert sha256(cohort_path) == '0a50e7d37a803d528daee0bc811770bc7f4520accbf7405af4385f0749c74665'
    cohort = json.loads(cohort_path.read_text())
    assert len(cohort['pairs']) == 96
    selected = [p for p in cohort['pairs'] if p['stage'] == 'initial24']
    assert len(selected) == 24 and len({p['scene'] for p in selected}) == 11
    assert all(p['sampling'] == 'uniform_without_GT' for p in selected)
    for pair in cohort['pairs']:
        assert [i['camera_id'] for i in pair['images']] == [1, 2]
        assert all(sha256(i['path']) == i['sha256'] for i in pair['images'])
    numeric_path = ROOT/'results/omega_numeric_evidence_v1.json'
    numeric = json.loads(numeric_path.read_text())
    actual = {}
    for name, run in numeric['runs'].items():
        m, c = run['manifest'], run['process_clock']
        assert m['actual_model_calls_attempted'] == m['actual_model_calls_completed'] == 48
        assert m['pairs'] == len(run['rows']) == 24
        assert sha256(c['log']['path']) == c['log']['sha256']
        assert c['exit_code'] == 0 and c['status'] == 'CHILD_FINISHED'
        raw = [json.loads(line) for line in Path(m['outputs']['path']).read_text().splitlines()]
        assert raw == run['rows']
        actual[name] = {row['pair_id']:row for row in raw}
        assert all(o['status'] == 'FINITE_INTERFACE_OUTPUTS' for row in raw for o in row['outputs'].values())
    assert set(actual['omega512_initial24_v1']) == set(actual['old_vggt_initial24_v1']) == {p['pair_id'] for p in selected}
    comparison = json.loads((ROOT/'results/omega_distinct_heldout_comparison_v1.json').read_text())
    geometry = geometry_module('omega_geometry')
    checked = 0
    for corr in numeric['putative_correspondences']:
        pair_id = corr['pair_id']; a = corr['arrays']
        first, second = distinct_matches(np.r_[a['fit_points_A'], a['eval_points_A']].reshape(-1, 2),
                                        np.r_[a['fit_points_B'], a['eval_points_B']].reshape(-1, 2))
        row = actual['omega512_initial24_v1'][pair_id]
        width, height = row['original_sizes'][0]
        blocks = np.floor(first/np.array([width,height])*4).astype(int).clip(0,3)
        is_eval = (blocks[:,0]+blocks[:,1]) % 2 != 0
        for method, tag in [('omega512','omega512_initial24_v1'), ('old_vggt','old_vggt_initial24_v1')]:
            for order in ['AB','BA']:
                output = actual[tag][pair_id]['outputs'][order]
                F = np.array(output['F_original'] if method == 'omega512' else output['F_in_physical_BA'])
                if method == 'omega512' and order == 'BA':
                    F = F.T
                metrics = geometry.background_validity(F, first[is_eval], second[is_eval], row['original_sizes'])
                saved = next(r for r in comparison['rows'] if (r['pair_id'],r['method'],r['order']) == (pair_id,method,order))
                assert metrics['valid'] == saved['valid']
                assert metrics['median_residual'] == saved['median_original_px']
                assert metrics['inlier_ratio'] == saved['inlier_ratio']
                checked += 1
    assert checked == 96
    with (ROOT/'results/omega_camera_pose_comparison.csv').open() as f:
        assert len(list(csv.DictReader(f))) == 96
    for name in ['omega_identity_candidate_scores.csv','omega_full_tracking_metrics.csv']:
        with (ROOT/'results'/name).open() as f:
            rows = list(csv.DictReader(f))
        assert all(r['status'] == 'NOT_EXECUTED' for r in rows)
    native = json.loads((ROOT/'results/omega_native_synchronous_scope_v1.json').read_text())
    assert native['all12_counts']['all_primary_queries'] == 169468
    assert native['all12_counts']['known_wrong_or_missed'] == len(native['errors']) == 1553
    assert native['all12_counts']['errors_with_synchronous_pure_positive'] == 314
    assert native['fit11_counts']['errors_with_synchronous_pure_positive'] == 304
    assert all(r['geometry_reliability'] == 'NOT_MEASURED' for r in native['errors'])
    assert all(r['geometry_candidate_discrimination'] == 'NOT_EXECUTED' for r in native['errors'])
    for name in ['phase0_native_fp16', 'phase0_27851eab']:
        assert not subprocess.check_output(['git','status','--porcelain'],cwd=RUNTIME/'sources'/name,text=True).strip()
    old_tree = Path('/data1/liuyeqiang/WWW_trackvggt_20261015')
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=old_tree,text=True).strip() == 'c6999a7ab1d22549816b5219734f53c8dcb74178'
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=old_tree,text=True).strip()
    result = dict(status='PASS_BOUNDED_DIAGNOSTIC_PUBLICATION_CHECKS_MAIN_RESEARCH_INCOMPLETE',
                  scientific_goal_complete=False, main_416='BLOCKED_WEIGHT_ACCESS',
                  cohort_pairs=96, actual_uniform_pairs=24, actual_model_forwards_total=96,
                  independently_recomputed_model_order_metrics=checked,
                  no_inference_or_GT_matching_during_check=True,
                  comparison=dict(path=str(ROOT/'results/omega_distinct_heldout_comparison_v1.json'), sha256=sha256(ROOT/'results/omega_distinct_heldout_comparison_v1.json')),
                  numeric_evidence=dict(path=str(numeric_path),sha256=sha256(numeric_path)),
                  publication_index=dict(path=str(ROOT/'results/omega_experiment_manifest.json'),sha256=sha256(ROOT/'results/omega_experiment_manifest.json')),
                  old_worktree_unchanged=True, immutable_producer_sources_clean=True, provenance=provenance())
    save_new(ROOT/'results/omega_publication_checks_v1.json', result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['provenance','numeric_evidence','comparison','publication_index']}))


if __name__ == '__main__':
    main()
