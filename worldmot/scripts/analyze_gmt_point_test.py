#!/usr/bin/env python3
"""Post-hoc raw-GT error attribution for immutable TEST predictions; CPU only.

The fixed-IoU identity diagnostics here complement the official metrics. They
are never HOTA/IDF1/CVIDF1 and never feed training, inference or selection.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np

from gmt_point_report_common import NAS, LEGAL, artifact, new_routes, read_json, sha, validate_route


def match_boxes(detections, gt):
    """One-to-one descending-IoU matching at fixed .5; deterministic ties."""
    if not detections or not gt:
        return {}
    a = np.asarray([x['bbox_xywh'] for x in detections], dtype=float)
    b = np.asarray([x['bbox'] for x in gt], dtype=float)
    lo = np.maximum(a[:, None, :2], b[None, :, :2])
    hi = np.minimum(a[:, None, :2] + a[:, None, 2:], b[None, :, :2] + b[None, :, 2:])
    intersection = np.maximum(hi - lo, 0).prod(-1)
    union = a[:, 2:].prod(-1)[:, None] + b[:, 2:].prod(-1)[None, :] - intersection
    iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
    candidates = [(float(iou[i, j]), int(i), int(j)) for i, j in zip(*np.where(iou >= .5))]
    matched, used = {}, set()
    for _, i, j in sorted(candidates, reverse=True):
        if i not in matched and j not in used:
            matched[i] = j
            used.add(j)
    return matched


def size_bin(box):
    area = float(box[2]) * float(box[3])
    return 'small_area_lt_32sq' if area < 32 ** 2 else 'medium_area_lt_96sq' if area < 96 ** 2 else 'large_area_ge_96sq'


def raw_gt(scene):
    by_view = {}
    for record in scene['raw_gt']:
        if sha(record['path']) != record['sha256']:
            raise ValueError('Original GT changed: ' + record['path'])
        frames = defaultdict(list)
        for line in Path(record['path']).read_text().splitlines():
            row = [float(x) for x in line.split(',')]
            if len(row) >= 8 and (row[6] == 0 or row[7] != 1):
                continue
            frames[int(row[0])].append({'id': int(row[1]), 'bbox': row[2:6],
                                       'visibility': row[8] if len(row) > 8 else None})
        by_view[int(record['view'])] = frames
    return by_view


def analyze_scene(task):
    route, sid, scene, manifest = task
    path = route.predictions / f'scene_{sid:02d}/tracks.jsonl'
    digest = hashlib.sha256()
    candidate_digest = hashlib.sha256()
    counters = Counter()
    grouped = defaultdict(Counter)
    history, previous = {}, {}
    failures = []
    gt_by_view = raw_gt(scene) if sid in LEGAL else None
    expected_time = int(scene['first_frame'])
    with path.open('rb') as stream:
        for line in stream:
            digest.update(line)
            if gt_by_view is None:
                continue
            row = json.loads(line)
            timestamp = int(row['timestamp'])
            if timestamp != expected_time or row.get('causal') is not True:
                raise ValueError(f'Noncausal/incomplete stream: {route.key}/{sid}/{timestamp}')
            expected_time += 1
            counters['timestamps'] += 1
            # Identical canonical input signature across methods; model and
            # output fields are deliberately absent from this digest.
            canonical = []
            by_det, by_view_identity, emitted = {}, {}, set()
            for view_text, detections in sorted(row['detections_by_camera'].items()):
                view = int(view_text)
                canonical.append([view, [[int(d['detection_id']), d['bbox_xywh'], d['score']] for d in detections]])
                gt = gt_by_view[view][timestamp]
                matching = match_boxes(detections, gt)
                counters.update(raw_candidates=len(detections), raw_GT_boxes=len(gt),
                                raw_detector_TP=len(matching), raw_detector_FP=len(detections)-len(matching),
                                raw_detector_FN=len(gt)-len(matching))
                crowd = 'crowd_lt10' if len(gt) < 10 else 'crowd_10to24' if len(gt) < 25 else 'crowd_ge25'
                matched_gt = set(matching.values())
                for gi, ann in enumerate(gt):
                    keys = [size_bin(ann['bbox']), crowd]
                    if ann['visibility'] is not None:
                        keys.append('visibility_lt0.5' if ann['visibility'] < .5 else 'visibility_ge0.5')
                    for key in keys:
                        grouped[key]['GT_boxes'] += 1
                        grouped[key]['detector_detected_GT' if gi in matched_gt else 'detector_missed_GT'] += 1
                detections_by_id = {int(d['detection_id']): (i, d) for i, d in enumerate(detections)}
                used, ids = set(), set()
                identity_outputs = {}
                for obj in row['tracks_2d_by_camera'].get(view_text, []):
                    if not obj.get('visible', True) or not obj.get('evaluation_eligible', True):
                        continue
                    if 'detection_id' in obj:
                        index, det = detections_by_id[int(obj['detection_id'])]
                        if det['bbox_xywh'] != obj['bbox_xywh']:
                            raise ValueError('Output changed the shared detection box')
                    else:
                        indices = [i for i, d in enumerate(detections) if d['bbox_xywh'] == obj['bbox_xywh'] and i not in used]
                        if not indices:
                            raise ValueError('Output has no exact current detection support')
                        index = indices[0]
                        det = detections[index]
                    gid = int(obj['global_id'])
                    if index in used or gid in ids:
                        raise ValueError('Duplicate observation or global ID within a camera')
                    used.add(index)
                    ids.add(gid)
                    emitted.add((view, int(det['detection_id'])))
                    counters['output_boxes'] += 1
                    if index in matching:
                        ann = gt[matching[index]]
                        identity_outputs[ann['id']] = gid
                        history[timestamp, view, gid] = ann['id']
                        counters['detected_GT_retained_in_output'] += 1
                        for key in (size_bin(ann['bbox']), crowd):
                            grouped[key]['GT_retained_in_output'] += 1
                        if ann['visibility'] is not None:
                            key = 'visibility_lt0.5' if ann['visibility'] < .5 else 'visibility_ge0.5'
                            grouped[key]['GT_retained_in_output'] += 1
                        old = previous.get((view, ann['id']))
                        if old is not None and old[1] != gid:
                            gap = timestamp - old[0]
                            counters['diagnostic_identity_label_changes'] += 1
                            counters['change_adjacent_frames' if gap == 1 else 'change_gap_2to30' if gap <= 30 else 'change_gap_gt30'] += 1
                        previous[view, ann['id']] = timestamp, gid
                    else:
                        counters['unmatched_output_boxes'] += 1
                for index, det in enumerate(detections):
                    target = gt[matching[index]]['id'] if index in matching else None
                    by_det[view, int(det['detection_id'])] = target
                    if index in matching and index not in used:
                        counters['detected_GT_lost_after_detector'] += 1
                by_view_identity[view] = identity_outputs
            candidate_digest.update(json.dumps([timestamp, canonical], separators=(',', ':'), allow_nan=False).encode() + b'\n')
            for target in {x for (v, _), x in by_det.items() if v == 1 and x is not None} & {x for (v, _), x in by_det.items() if v == 2 and x is not None}:
                counters['same_person_detected_in_both_views'] += 1
                a, b = by_view_identity.get(1, {}).get(target), by_view_identity.get(2, {}).get(target)
                if a is not None and b is not None:
                    counters['same_person_output_in_both_views'] += 1
                    correct = a == b
                    counters['same_person_shared_ID' if correct else 'same_person_split_ID'] += 1
                    if not correct and len(failures) < 3:
                        failures.append({'type': 'SAME_PERSON_CROSS_VIEW_SPLIT', 'timestamp': timestamp,
                                         'GT_identity': target, 'predicted_IDs': [a, b]})
                else:
                    counters['same_person_cross_view_output_missing'] += 1
            inverse_b = {gid: identity for identity, gid in by_view_identity.get(2, {}).items()}
            for identity, gid in by_view_identity.get(1, {}).items():
                if gid in inverse_b and inverse_b[gid] != identity:
                    counters['different_person_cross_view_shared_ID'] += 1
                    if len(failures) < 6:
                        failures.append({'type': 'DIFFERENT_PERSON_CROSS_VIEW_MERGE', 'timestamp': timestamp,
                                         'GT_identities': [identity, inverse_b[gid]], 'predicted_ID': gid})
            for decision in row.get('decisions', []):
                counters['decision_' + decision['kind']] += 1
                evidence = decision.get('evidence', {})
                if evidence.get('kind') == 'TEMPORAL':
                    source = history.get((evidence['reference_timestamp'], evidence['source_camera'], evidence['global_id']))
                    target = by_det.get((decision['view_id'], decision['detection_id']))
                    if source is not None and target is not None:
                        counters['selected_temporal_known_identity'] += 1
                        counters['selected_temporal_correct_identity' if source == target else 'selected_temporal_wrong_identity'] += 1
            for ev in row.get('point_evidence', []):
                if 'valid_point_count' not in ev:
                    continue
                kind = ev['kind'].lower()
                counters[kind + '_candidate_edges'] += 1
                counters[kind + '_valid_point_votes_repeated_per_edge'] += ev['valid_point_count']
                counters[kind + '_inside_point_votes_repeated_per_edge'] += ev['inside_box_count']
                if ev.get('rejection_reason'):
                    counters['point_reject_' + ev['rejection_reason']] += 1
                if ev['association_score'] <= 0:
                    continue
                target = by_det.get((ev['target_camera'], ev['detection_id']))
                source = (history.get((ev['reference_timestamp'], ev['source_camera'], ev['global_id']))
                          if ev['kind'] == 'TEMPORAL' else by_det.get((ev['source_camera'], ev['source_detection_id'])))
                if source is not None and target is not None:
                    counters[kind + '_positive_edges_known_identity'] += 1
                    counters[kind + ('_positive_edges_correct_identity' if source == target else '_positive_edges_wrong_identity')] += 1
            for prediction in row.get('point_predictions', []):
                valid = np.asarray(prediction['valid'], dtype=bool)
                visibility = np.asarray(prediction['visibility'])
                confidence = np.asarray(prediction['confidence'])
                counters['logged_point_prediction_samples'] += int(valid.size)
                counters['logged_points_in_original_image'] += int(valid.sum())
                settings = manifest.get('settings', {})
                good = valid & (visibility >= settings.get('minimum_visibility', .5)) & (confidence >= settings.get('minimum_confidence', .5))
                counters['logged_points_passing_frozen_gates'] += int(good.sum())
            alignment = row.get('alignment', {})
            if alignment:
                counters['alignment_frames'] += 1
                counters['alignment_success' if alignment.get('success') else 'alignment_failure'] += 1
            runtime = row.get('query_runtime', {})
            counters['query_capacity_overflow'] += int(runtime.get('query_capacity_overflow', 0))
            for rejection in runtime.get('rejections', []):
                counters['query_reject_' + rejection['reason']] += 1
            for obs in row.get('world_observations', []):
                counters['world_observations'] += 1
                counters['world_geometry_reliable' if obs.get('geometry_reliable', obs.get('geometry_reliability', 0)) else 'world_geometry_unreliable'] += 1
            history = {key: value for key, value in history.items() if key[0] >= timestamp - 4}
    if digest.hexdigest() != manifest['prediction_sha256']:
        raise ValueError(f'Frozen tracks changed: {path}')
    if gt_by_view is not None and counters['timestamps'] != manifest['frames']:
        raise ValueError('Incomplete diagnostic video')
    return {'method': route.key, 'scene': sid, 'official_GT_legal': sid in LEGAL,
            'frames': manifest['frames'], 'tracks': {'path': str(path), 'bytes': path.stat().st_size,
                                                     'sha256': digest.hexdigest()},
            'candidate_digest': candidate_digest.hexdigest() if gt_by_view is not None else None,
            'counts': dict(counters), 'groups': {k: dict(v) for k, v in grouped.items()}, 'failure_examples': failures}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nas', type=Path, default=NAS)
    parser.add_argument('--results', type=Path, default=Path(__file__).resolve().parents[1] / 'results')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--methods', nargs='+')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve the prior diagnostic: ' + str(args.output))
    audit_path = args.nas / 'fulltrain_20261022/audit/data_audit.json'
    scenes = {int(x['scene']): x for x in read_json(audit_path)['test']['scenes']}
    tasks, provenance = [], {}
    for route in new_routes(args.nas):
        if args.methods and route.key not in args.methods:
            continue
        _, manifests = validate_route(route, args.results)
        provenance[route.key] = {'metrics': artifact(route.metrics), 'gate': artifact(args.results / route.gate)}
        tasks.extend((route, sid, scenes[sid], manifest) for sid, manifest in manifests.items())
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(analyze_scene, tasks, chunksize=1):
            records.append(result)
            print(json.dumps({'completed_scenes': len(records), 'total_scenes': len(tasks),
                              'method': result['method'], 'scene': result['scene']}), flush=True)
    methods = {}
    for method in provenance:
        items = [r for r in records if r['method'] == method]
        counts, groups = Counter(), defaultdict(Counter)
        for item in items:
            counts.update(item['counts'])
            for key, values in item['groups'].items():
                groups[key].update(values)
        def ratio(a, b):
            return counts[a] / counts[b] if counts[b] else None
        methods[method] = {'counts': dict(counts), 'groups': {k: dict(v) for k, v in groups.items()},
                           'same_person_shared_ID_ratio': ratio('same_person_shared_ID', 'same_person_detected_in_both_views'),
                           'selected_temporal_identity_consistency': ratio('selected_temporal_correct_identity', 'selected_temporal_known_identity'),
                           'logged_valid_point_ratio': ratio('logged_points_passing_frozen_gates', 'logged_point_prediction_samples'),
                           'scenes': items, **provenance[method]}
    candidate_checks = {str(sid): {r['method']: r['candidate_digest'] for r in records if r['scene'] == sid} for sid in LEGAL}
    if any(len(set(values.values())) != 1 for values in candidate_checks.values()):
        raise ValueError('Methods did not consume identical raw detection candidates')
    report = {'status': 'COMPLETE_POSTHOC_TEST_DIAGNOSTICS', 'official_tracking_metrics': False,
              'IoU': .5, 'matching': 'descending IoU one-to-one; raw person GT with marked=1; fixed before diagnostics',
              'raw_GT_modified': False, 'TEST_used_for_selection': False, 'audit': artifact(audit_path),
              'legal_scenes': list(LEGAL), 'blocked_scenes_inference_only': [5, 12, 13, 14, 17, 20, 21, 22],
              'all_22_scene_prediction_hashes_verified': True, 'shared_candidate_digests': candidate_checks,
              'point_accuracy_definition': 'Box-level identity consistency, not dense correspondence truth. Candidate edges, selected decisions, and logged point samples are separate populations.',
              'occlusion_definition': 'Raw GT visibility below .5 when available. Identity label changes grouped by observation gap are descriptive, not proof that an occlusion caused the change.',
              'causal_attribution_limit': 'Alignment failure and unreliable geometry are measured exposures; they do not alone prove the cause of a specific wrong identity.',
              'methods': methods}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': report['status'], 'output': str(args.output),
                      'methods': {k: v['counts'] for k, v in methods.items()}}), flush=True)


if __name__ == '__main__':
    main()
