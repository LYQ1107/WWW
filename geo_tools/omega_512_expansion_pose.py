"""Conditional preregistered remaining72 independent512 camera predictions, no GMT/GT."""
import argparse
import collections
import json
from pathlib import Path
import sys
import time
import numpy as np
from omega_common import DATA, NATIVE, ROOT, RUNTIME, jsonl, provenance, save_new, sha256
from omega_modules import geometry_module


def angle(a, b):
    value = float(np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b)))
    return float(np.degrees(np.arccos(np.clip(value, -1., 1.))))


def spatial_split(points, width, height):
    # Fixed4x4 original-camera-A blocks; no residual or GT selects either split.
    indices = np.floor(np.asarray(points)/np.array([width, height])*4).astype(int).clip(0, 3)
    return (indices[:, 0]+indices[:, 1]) % 2 == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resolution', type=int, choices=[512], required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--limit', type=int, default=72, choices=[72])
    args = parser.parse_args()
    import cv2, torch
    cv2.setNumThreads(4)
    torch.set_num_threads(4)
    torch.manual_seed(20261016)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    out = RUNTIME/'pose_audits'/args.tag
    out.mkdir(parents=True, exist_ok=False)
    plan = ROOT/'manifests/omega_fixed_pair_cohort_v1.json'
    cohort = json.loads(plan.read_text())
    pairs = [p for p in cohort['pairs'] if p['stage'] != 'initial24'][:args.limit]
    assert len(pairs)==72
    pilot=json.loads((ROOT/'results/omega512_independent_evidence_v1.json').read_text())
    assert pilot['manifest']['admitted_arms']==['C2_sift6000_mutual']
    source = provenance()
    if source['source_status'].strip():
        raise ValueError('immutable clean source required')
    for pair in pairs:
        for image in pair['images']:
            if sha256(image['path']) != image['sha256']:
                raise ValueError('changed original image')
    boxes = collections.defaultdict(list)
    wanted = {i['image_id'] for p in pairs for i in p['images']}
    detector_sources = []
    for scene in sorted({p['scene'] for p in pairs}):
        path = NATIVE/f'scene{scene}_full_v1/detector_provenance.jsonl.gz'
        receipt = next(r for r in cohort['native_source_receipts'] if Path(r['path']).parent.name == f'scene{scene}_full_v1')
        assert sha256(path) == receipt['artifacts']['detector_provenance.jsonl.gz']
        detector_sources.append(dict(path=str(path), sha256=sha256(path)))
        for row in jsonl(path):
            if row['source'][0] in wanted:
                x, y, w, h = row['detection']['bbox']
                boxes[row['source'][0]].append([x, y, x+w, y+h])
    adapter_module = geometry_module('omega_adapter')
    geometry = geometry_module('omega_geometry')
    confidence = geometry_module('geometry_confidence')
    align = geometry_module('geometry_alignment')
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    adapter = adapter_module.OmegaAdapter(RUNTIME/'official_vggt_omega', args.checkpoint, args.resolution)
    rows = []
    with (out/'pair_results.jsonl').open('x') as stream:
        for pair in pairs:
            pair_started = time.perf_counter()
            images = pair['images']
            paths = [i['path'] for i in images]
            camera_ids = [i['camera_id'] for i in images]
            sizes = [(i['original_width'], i['original_height']) for i in images]
            bboxes = [np.asarray(boxes[i['image_id']], dtype=float).reshape(-1, 4) for i in images]
            gray = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in paths]
            if any(i is None for i in gray):
                raise ValueError('failed raw image read')
            # Existing extraction is reused; its all-point F/H and self-fit
            # statistics are discarded, never used to filter held-out points.
            t = time.perf_counter()
            matches = confidence.estimate_background(*gray, *bboxes, method='sift', mask_targets=True)
            background_seconds = time.perf_counter()-t
            p, q = matches['points1'], matches['points2']
            row = dict(pair_id=pair['pair_id'], scene=pair['scene'], group=pair['group'],
                       resolution=args.resolution, role='MAIN_416_REPRODUCTION' if args.resolution == 416 else 'INDEPENDENT_512_CONDITIONAL_REMAINING72',
                       original_sizes=sizes, descriptor_ratio_test_matches=matches['matches'],
                       retained_putative_sift_matches=len(p), background_seconds=background_seconds,
                       outputs={}, order={}, actual_GT_or_identity_evaluation='NOT_EXECUTED')
            results = []
            for name, order in [('AB', [0, 1]), ('BA', [1, 0])]:
                result = adapter.infer_pair([paths[i] for i in order], [camera_ids[i] for i in order], pair['group'])
                item = {k: v for k, v in result.items() if k not in ['predictions', 'mappings']}
                item.pop('extrinsics', None); item.pop('intrinsics', None)
                if result['status'] == 'FINITE_INTERFACE_OUTPUTS':
                    item['shapes'] = {k: list(v.shape) for k, v in result['predictions'].items()}
                    item['mappings'] = [m.record() for m in result['mappings']]
                    item['extrinsics'] = result['extrinsics'].tolist()
                    item['intrinsics'] = result['intrinsics'].tolist()
                    item['depth_conf_summary'] = [float(v) for v in torch.quantile(result['predictions']['depth_conf'].flatten(), torch.tensor([0., .5, .9, 1.], device='cuda'))]
                    item['depth_summary'] = [float(v) for v in torch.quantile(result['predictions']['depth'].flatten(), torch.tensor([0., .5, .9, 1.], device='cuda'))]
                    try:
                        result['geometry'] = geometry.original_fundamental(result['extrinsics'], result['intrinsics'], result['mappings'])
                        item['F_original'] = result['geometry']['F_original'].tolist()
                    except (ValueError, np.linalg.LinAlgError) as exc:
                        item['camera_failure'] = str(exc)
                # Dense outputs are transient; never persisted as a video cache.
                result.pop('predictions', None)
                row['outputs'][name] = item
                results.append(result)
            keep = np.ones(len(p), dtype=bool)
            for result, order in zip(results, ([0, 1], [1, 0])):
                if 'mappings' in result:
                    for mapping, physical_index in zip(result['mappings'], order):
                        _, valid = mapping.forward((p, q)[physical_index])
                        keep &= valid
            p, q = p[keep], q[keep]
            train_mask = spatial_split(p, *sizes[0])
            fit_p, fit_q = p[train_mask], q[train_mask]
            eval_p, eval_q = p[~train_mask], q[~train_mask]
            corr = out/(pair['pair_id']+'_background.npz')
            np.savez_compressed(corr, fit_points_A=fit_p, fit_points_B=fit_q,
                                eval_points_A=eval_p, eval_points_B=eval_q)
            row.update(common_valid_fit_matches=len(fit_p), common_valid_eval_matches=len(eval_p),
                       correspondence_file=dict(path=str(corr), sha256=sha256(corr)))
            reference_F = None
            if len(fit_p) >= 8:
                cv2.setRNGSeed(20261016)
                reference_F, mask = cv2.findFundamentalMat(fit_p, fit_q, cv2.FM_RANSAC, 3., .99)
                if reference_F is not None and reference_F.shape != (3, 3):
                    reference_F = None
                row['traditional_fit_inliers'] = int(mask.sum()) if mask is not None else 0
            else:
                row['traditional_fit_inliers'] = 'NOT_MEASURED'
            row['traditional_heldout'] = geometry.background_validity(reference_F, eval_p, eval_q, sizes)
            row['traditional_F'] = reference_F.tolist() if reference_F is not None else None
            for name, result in zip(('AB', 'BA'), results):
                F = result.get('geometry', {}).get('F_original')
                if F is not None and name == 'BA':
                    F = F.T
                row['outputs'][name]['heldout_background'] = geometry.background_validity(F, eval_p, eval_q, sizes)
            if all('geometry' in r for r in results):
                first, second = [r['geometry'] for r in results]
                R = first['R_BA'] @ second['R_BA']
                rotation_error = float(np.degrees(np.arccos(np.clip((np.trace(R)-1)/2, -1., 1.))))
                forward_t = first['t_BA']
                reverse_recovered_t = -second['R_BA'].T @ second['t_BA']
                first_scores = align.soft_box_evidence(first['F_original'], *bboxes, True)
                second_scores = align.soft_box_evidence(second['F_original'].T, *bboxes, True)
                row['order'] = dict(F_transpose_scale_sign_error=geometry.fundamental_equivalence(first['F_original'], second['F_original'].T),
                                    relative_rotation_difference_degrees=rotation_error,
                                    translation_direction_difference_degrees=angle(forward_t, reverse_recovered_t),
                                    raw_weak_box_score_max_difference=float(np.max(np.abs(first_scores-second_scores))) if first_scores.size else 'NOT_MEASURED',
                                    weak_box_pairs=int(first_scores.size), identity_interpretation='NOT_ESTABLISHED')
            row['pair_wall_seconds'] = time.perf_counter()-pair_started
            rows.append(row)
            stream.write(json.dumps(row, allow_nan=False)+'\n'); stream.flush()
            print(json.dumps(dict(pair_id=pair['pair_id'], interface={k:v['status'] for k,v in row['outputs'].items()},
                  eval_matches=len(eval_p), valid={k:v['heldout_background']['valid'] for k,v in row['outputs'].items()})), flush=True)
            if len(rows) == 1 and any(v['status'] != 'FINITE_INTERFACE_OUTPUTS' for v in row['outputs'].values()):
                print('STOP_FIRST_PAIR_INTERFACE_FAILURE_BEFORE_EXPANDING', flush=True)
                break
    torch.cuda.synchronize()
    valid = sum(r['outputs']['AB']['heldout_background']['valid'] for r in rows)
    save_new(out/'manifest.json', dict(status='BOUNDED_DIAGNOSTIC_COMPLETE' if len(rows) == len(pairs) else 'STOPPED_FIRST_PAIR_INTERFACE_FAILURE', resolution=args.resolution,
             role='MAIN_416_REPRODUCTION' if args.resolution == 416 else 'INDEPENDENT_512_CONDITIONAL_REMAINING72',
             cohort=dict(path=str(plan), sha256=sha256(plan)), checkpoint=dict(path=str(args.checkpoint), sha256=sha256(args.checkpoint)),
             model_strict_state_keys=adapter.state_keys, model_load_seconds=adapter.load_seconds,
             load_through_pair_export_seconds=time.perf_counter()-started,
             CUDA_allocator_peak_bytes=torch.cuda.max_memory_allocated(),
             CUDA_reserved_peak_bytes=torch.cuda.max_memory_reserved(),
             pairs=len(rows), actual_model_calls_attempted=adapter.calls_attempted,
             actual_model_calls_completed=adapter.calls_completed, reliable_AB_pairs=valid,
             outputs=dict(path=str(out/'pair_results.jsonl'), sha256=sha256(out/'pair_results.jsonl')),
             detector_sources=detector_sources, no_GT_input=True, unchanged_GMT_no_inference=True,
             no_cross_time_propagation=True, inference_seed=20261016,
             torch=torch.__version__, torch_cuda=torch.version.cuda, numpy=np.__version__, opencv=cv2.__version__,
             inference_precision='official autocast=False constructor plus explicit native-supported outer AMP; original fp32 camera/depth head guards retained',
             hardware=dict(name=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()),
                           bf16_reported_including_emulation=torch.cuda.is_bf16_supported(),
                           aggregator_dtype=str(adapter.amp_dtype)),
             official_416_status='BLOCKED_WEIGHT_ACCESS' if args.resolution == 512 else 'LOADED',
             main_scientific_goal_complete=False, provenance=source))


if __name__ == '__main__':
    main()
