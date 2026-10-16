"""Common held-out comparison with distinct descriptor endpoints, no model rerun.

SIFT can yield multiple orientations at a single location or many-to-one
matches. Deduplicate both endpoints before the fixed spatial split and fit.
All rejected/insufficient image pairs remain in the final denominator.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from omega_common import ROOT, provenance, save_new, sha256
from omega_modules import geometry_module


def distinct_matches(first, second):
    first = np.asarray(first).reshape(-1, 2)
    second = np.asarray(second).reshape(-1, 2)
    if len(first) != len(second) or not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError('invalid descriptor correspondence coordinates')
    # Millipixel binning removes numerically duplicated keypoint locations.
    # Coordinate order fixes which many-to-one pair survives, without scores,
    # residuals, model outputs, identity labels or RANSAC masks.
    a = np.round(first*1000).astype(np.int64)
    b = np.round(second*1000).astype(np.int64)
    keys = np.c_[a, b]
    indices = sorted(range(len(keys)), key=lambda i: tuple(keys[i]))
    seen_a, seen_b, selected = set(), set(), []
    for i in indices:
        ka, kb = tuple(a[i]), tuple(b[i])
        if ka not in seen_a and kb not in seen_b:
            selected.append(i); seen_a.add(ka); seen_b.add(kb)
    return first[selected], second[selected]


def load_run(directory):
    manifest = Path(directory)/'manifest.json'
    meta = json.loads(manifest.read_text())
    output = meta['outputs']
    assert sha256(output['path']) == output['sha256']
    return meta, {r['pair_id']:r for r in map(json.loads, Path(output['path']).read_text().splitlines())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--omega-run', type=Path, required=True)
    parser.add_argument('--old-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--csv-output', type=Path, required=True)
    args = parser.parse_args()
    import cv2
    geometry = geometry_module('omega_geometry')
    omega_meta, omega = load_run(args.omega_run)
    old_meta, old = load_run(args.old_run)
    assert omega_meta['status'] == 'BOUNDED_DIAGNOSTIC_COMPLETE'
    assert old_meta['status'] == 'BOUNDED_OLD_VGGT_CONTROL_COMPLETE'
    assert set(omega) == set(old)
    rows, pair_audits = [], []
    for pair_id, row in omega.items():
        corr = row['correspondence_file']
        assert sha256(corr['path']) == corr['sha256']
        with np.load(corr['path']) as arr:
            first = np.r_[arr['fit_points_A'], arr['eval_points_A']]
            second = np.r_[arr['fit_points_B'], arr['eval_points_B']]
        raw_count = len(first)
        first, second = distinct_matches(first, second)
        width, height = row['original_sizes'][0]
        blocks = np.floor(first/np.array([width, height])*4).astype(int).clip(0, 3)
        fit_mask = (blocks[:, 0]+blocks[:, 1]) % 2 == 0
        fit_p, fit_q = first[fit_mask], second[fit_mask]
        eval_p, eval_q = first[~fit_mask], second[~fit_mask]
        reference = None
        fit_inliers = 'NOT_MEASURED'
        if len(fit_p) >= 8:
            cv2.setRNGSeed(20261016)
            reference, mask = cv2.findFundamentalMat(fit_p, fit_q, cv2.FM_RANSAC, 3., .99)
            fit_inliers = int(mask.sum()) if mask is not None else 'NOT_MEASURED'
            if reference is not None and reference.shape != (3, 3):
                reference = None
        models = []
        for name, source in [('omega512', row), ('old_vggt', old[pair_id])]:
            for order, output in source['outputs'].items():
                F = output.get('F_original') if name == 'omega512' else output.get('F_in_physical_BA')
                if F is not None:
                    F = np.array(F)
                    if name == 'omega512' and order == 'BA':
                        F = F.T
                models.append((name, order, F, output['status']))
        models.append(('traditional_RANSAC', 'AB', reference, 'FITTED_ONLY_ON_DISTINCT_FIT_POINTS' if reference is not None else 'NO_FITTED_F'))
        for name, order, F, interface in models:
            metrics = geometry.background_validity(F, eval_p, eval_q, row['original_sizes'])
            rows.append(dict(pair_id=pair_id, scene=row['scene'], group=row['group'], method=name,
                             order=order, interface_status=interface, raw_putative_matches=raw_count,
                             distinct_matches=len(first), fit_matches=len(fit_p), eval_matches=len(eval_p),
                             valid=metrics['valid'], inlier_ratio=metrics['inlier_ratio'],
                             median_original_px=metrics['median_residual'], p90_original_px=metrics['p90_residual'],
                             median_diagonal_normalized=metrics['median_diagonal_normalized'],
                             original_area_coverage=metrics['coverage'], reasons='|'.join(metrics['reasons']),
                             status='MEASURED_DIAGNOSTIC_ONLY_MAIN416_BLOCKED'))
        pair_audits.append(dict(pair_id=pair_id, raw_count=raw_count, distinct_count=len(first),
                               dropped_duplicate_or_many_to_one=raw_count-len(first),
                               fit_count=len(fit_p), eval_count=len(eval_p), traditional_fit_inliers=fit_inliers,
                               all_failure_pairs_retained=True))
    summary = {}
    for method in ['omega512', 'old_vggt', 'traditional_RANSAC']:
        for order in (['AB'] if method == 'traditional_RANSAC' else ['AB', 'BA']):
            part = [r for r in rows if r['method'] == method and r['order'] == order]
            summary[f'{method}_{order}'] = dict(pairs=len(part), valid=sum(r['valid'] for r in part),
                 scenes_with_valid=sorted({r['scene'] for r in part if r['valid']}),
                 finite_median_pairs=sum(isinstance(r['median_original_px'], float) for r in part))
    result = dict(status='COMPLETE_COMMON_DISTINCT_HELDOUT_DIAGNOSTIC',
                  main_416_status='BLOCKED_WEIGHT_ACCESS', scientific_main_complete=False,
                  raw_producer_validity='retained as preliminary; final common comparison uses distinct endpoints before fit/split',
                  pairing='all models share the same distinct held-out points for each physical pair',
                  point_truth='putative masked SIFT matches, not GT camera pose or identity correspondences',
                  unchanged_reference_thresholds=dict(min_eval_matches=20, min_inlier_ratio=.25, max_median_original_px=12., min_original_area_coverage=.01),
                  source_manifests=[dict(path=str(Path(d)/'manifest.json'), sha256=sha256(Path(d)/'manifest.json')) for d in [args.omega_run,args.old_run]],
                  summary=summary, pair_audits=pair_audits, rows=rows, provenance=provenance())
    save_new(args.output, result)
    with args.csv_output.open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
