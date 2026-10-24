#!/usr/bin/env python3
"""Rebuild official tables from complete frozen routes; CPU-only reporting."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics
import numpy as np
from gmt_point_report_common import NAS, FIELDS, artifact, new_routes, read_json, sha, validate_route

HERE = Path(__file__).resolve().parents[1]
NA = 'NOT_AVAILABLE'


def write_csv(path, rows, fields=None):
    fields = fields or list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore', lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def extract(data):
    return {**{k: v for k, v in data['TrackEval']['combined'].items() if k in FIELDS},
            **{k: v for k, v in data['VisionTrack_MATLAB']['combined'].items() if k in FIELDS}}


def throughput(manifests):
    frames = sum(m['frames'] for m in manifests)
    seconds = sum(m['real_e2e_seconds'] for m in manifests)
    values = [m['real_e2e_fps'] for m in manifests]
    return dict(frames=frames, real_e2e_seconds=seconds, frame_weighted_fps=frames / seconds,
                mean_scene_fps=statistics.mean(values), median_scene_fps=statistics.median(values),
                harmonic_scene_fps=len(values) / sum(1 / x for x in values))


def checkpoint_parameters(path):
    if not Path(path).is_file():
        return NA
    import torch
    payload = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    return sum(t.numel() for t in payload['model'].values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nas', type=Path, default=NAS)
    parser.add_argument('--output', type=Path, default=HERE / 'results')
    parser.add_argument('--diagnostics', type=Path)
    parser.add_argument('--attach-diagnostics-only', action='store_true',
                        help='attach a completed diagnostic to existing verified tables without rescanning timing streams')
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if args.attach_diagnostics_only:
        if not args.diagnostics:
            parser.error('--attach-diagnostics-only requires --diagnostics')
        diagnostics = read_json(args.diagnostics)
        if (diagnostics.get('status') != 'COMPLETE_POSTHOC_TEST_DIAGNOSTICS'
                or set(diagnostics.get('methods', {})) != {r.key for r in new_routes(args.nas)}
                or diagnostics.get('all_22_scene_prediction_hashes_verified') is not True):
            raise ValueError('Cannot attach incomplete diagnostic evidence')
        report = read_json(out / 'gmt_vggt_point_final_comparison.json')
        if report.get('status') != 'COMPLETE_OFFICIAL_ROUTE_TABLE_SINGLE_SEED_EFFECT_GATE':
            raise ValueError('Generate the verified official table before attaching diagnostics')
        native_audit = read_json(out / 'gmt_native_parameter_audit_20261024.json')
        native_manifest = read_json(args.nas / 'fulltrain_20261022/test/gmt_native/scene_01/manifest.json')
        if native_manifest['checkpoint_sha256'] != native_audit['checkpoint_sha256']:
            raise ValueError('Native parameter audit checkpoint mismatch')
        native_row = next(r for r in report['official_rows'] if r['Method'] == 'GMT Native')
        native_row['Trainable_Assoc_Params'] = native_audit['native_association_transformer_parameters']
        report['parameter_definition'] = 'Native counts the pretrained association transformer; V2 counts its learned checkpoint tensors. Frozen point methods/fixed rules optimize zero added association parameters in this study. Native descriptor and classifier totals are separate in gmt_native_parameter_audit_20261024.json.'
        (out / 'gmt_vggt_point_error_attribution.json').write_text(json.dumps(diagnostics, indent=2, allow_nan=False) + '\n')
        summary = []
        for method, data in diagnostics['methods'].items():
            summary.append(dict(method=method, **data['counts'],
                                same_person_shared_ID_ratio=data['same_person_shared_ID_ratio'],
                                selected_temporal_identity_consistency=data['selected_temporal_identity_consistency']))
        write_csv(out / 'gmt_vggt_point_error_summary.csv', summary)
        report['posthoc_error_attribution'] = artifact(args.diagnostics)
        report['remaining'] = 'Final hardware scope, reproduction and GitHub delivery audit; no additional seed experiments.'
        (out / 'gmt_vggt_point_final_comparison.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        with (out / 'gmt_vggt_point_final_comparison.csv').open() as stream:
            fields = next(csv.reader(stream))
        write_csv(out / 'gmt_vggt_point_final_comparison.csv', report['official_rows'], fields)
        print(json.dumps(dict(status='COMPLETE_DIAGNOSTICS_ATTACHED', methods=len(summary))), flush=True)
        return
    rows, scene_rows, camera_rows, fps_rows, scene_fps, evidence = [], [], [], [], [], []

    def record(key, name, detector, association, pred, metrics_path, metrics, manifests, params, frozen):
        speed = throughput(list(manifests.values()))
        row = dict(Method=name, Detector=detector, Association=association, Trainable_Assoc_Params=params,
                   Parameters=frozen, **extract(metrics), E2E_FPS=speed['frame_weighted_fps'],
                   E2E_FPS_mean_scene=speed['mean_scene_fps'], E2E_FPS_harmonic_scene=speed['harmonic_scene_fps'],
                   Status='OFFICIAL_COMPLETE_14_VALID_GT_22_INFERENCE',
                   Timing='29019 timestamps / sum(full-video wall seconds); shared GPU; initialization/evaluation excluded')
        if key == 'gmt_native':
            row['Timing'] += '; final IDs delivered after native full-video processing'
        rows.append(row)
        evidence.append(dict(method=key, metrics=artifact(metrics_path), prediction_root=str(pred), timing=speed,
                             manifests=[dict(scene=sid, **artifact(pred / f'scene_{sid:02d}/manifest.json')) for sid in manifests]))
        for scene, values in metrics['TrackEval']['per_scene'].items():
            cv = metrics['VisionTrack_MATLAB']['per_scene'][scene]
            scene_rows.append(dict(method=name, scene=scene, **values, CVIDF1=cv['CVIDF1'], CVMA=cv['CVMA']))
        for camera, values in metrics['TrackEval']['per_camera'].items():
            camera_rows.append(dict(method=name, camera=camera, **values,
                                    CVIDF1='NOT_APPLICABLE_CROSS_CAMERA_METRIC', CVMA='NOT_APPLICABLE_CROSS_CAMERA_METRIC'))
        durations = defaultdict(list)
        for sid, m in manifests.items():
            scene_fps.append(dict(method=name, scene=sid, frames=m['frames'], full_video_seconds=m['real_e2e_seconds'],
                                  E2E_FPS=m['real_e2e_fps'], gpu=m.get('gpu', 'see frozen detector manifest'),
                                  resource_mode=m.get('resource_mode', 'historical; see manifest'),
                                  manifest_sha256=sha(pred / f'scene_{sid:02d}/manifest.json')))
            timing_path = pred / f'scene_{sid:02d}/timings.jsonl'
            if not timing_path.exists():
                continue
            count = 0
            with timing_path.open() as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    timing = json.loads(line)
                    count += 1
                    for stage, value in timing.get('stages_s', {}).items():
                        if isinstance(value, (int, float)):
                            durations[stage].append(value)
                    for stage in ('rgb_to_delivered_ids_s', 'rgb_read_s'):
                        if stage in timing and stage not in timing.get('stages_s', {}):
                            durations[stage].append(timing[stage])
            if count != m['frames']:
                raise ValueError('Incomplete timing stream: ' + str(timing_path))
        fps_rows.append(dict(scope='COMPLETE_22_TEST_VIDEOS', method=key, stage='full_video_delivery',
                             timestamps=speed['frames'], real_e2e_FPS=speed['frame_weighted_fps'],
                             mean_scene_FPS=speed['mean_scene_fps'], harmonic_scene_FPS=speed['harmonic_scene_fps'],
                             model_only_FPS='NOT_APPLICABLE', P50_s=NA, P95_s=NA, peak_GPU_memory_bytes=NA,
                             conditions=row['Timing'],
                             missing_reason='Full-video runners did not log process peak GPU memory. Native has no causal per-frame final-ID timing stream.',
                             source=str(pred)))
        for stage, values in durations.items():
            fps_rows.append(dict(scope='COMPLETE_22_TEST_VIDEOS', method=key, stage=stage, timestamps=len(values),
                                 real_e2e_FPS='NOT_APPLICABLE',
                                 model_only_FPS=len(values) / sum(values) if stage in ('VGGT_model_only_s', 'omega_forward') else 'NOT_APPLICABLE',
                                 P50_s=statistics.median(values), P95_s=float(np.quantile(values, .95)),
                                 peak_GPU_memory_bytes=NA, conditions='Pooled real wall times; nested stages overlap and must not be summed.',
                                 source=str(pred / 'scene_*/timings.jsonl')))

    old = args.nas / 'fulltrain_20261022'
    historical = ['gmt_native', 'v1', 'v2_joint_seed20261022', 'v2_joint_seed20261023',
                  'v2_joint_seed20261024', 'v2_no_omega_seed20261022', 'v2_no_appearance_seed20261022',
                  'v2_two_stage_seed20261022', 'v2_no_geometry_seed20261022']
    historical_scores, historical_speed = [], []
    for key in historical:
        path = old / 'test_evaluation' / key / 'metrics.json'
        metrics = read_json(path)
        pred = old / 'test' / key
        manifests = {sid: read_json(pred / f'scene_{sid:02d}/manifest.json') for sid in range(1, 23)}
        if (metrics.get('status') != 'COMPLETE_VALID_SCOPE' or metrics.get('raw_GT_modified') is not False
                or metrics.get('custom_metrics_used') is not False or metrics.get('box_source') != 'tracks_2d_by_camera'
                or len(metrics['evaluated_scenes']) != 14 or len(metrics['coverage']) != 22
                or sum(m['frames'] for m in manifests.values()) != 29019
                or not all(m.get('status') == 'COMPLETE' and m.get('full_video') is True and m.get('GT_in_inference') is False for m in manifests.values())):
            raise ValueError('Historical official gate is incomplete: ' + key)
        name = 'GMT Native' if key == 'gmt_native' else 'Original V1' if key == 'v1' else 'Original V2 ' + key.removeprefix('v2_')
        params = 0 if key == 'v1' else checkpoint_parameters(old / 'final' / (key.removeprefix('v2_') + '.frozen.pt')) if key.startswith('v2_') else NA
        if key == 'gmt_native':
            audit = read_json(out / 'gmt_native_parameter_audit_20261024.json')
            if manifests[1]['checkpoint_sha256'] != audit['checkpoint_sha256']:
                raise ValueError('Native parameter audit belongs to a different checkpoint')
            params = audit['native_association_transformer_parameters']
        record(key, name, 'GMT' if key == 'gmt_native' else 'RTMDet', 'Native GTA' if key == 'gmt_native' else 'Omega 3D rules' if key == 'v1' else 'PWQT',
               pred, path, metrics, manifests, params, 'HISTORICAL_FROZEN_REFERENCE')
        if key.startswith('v2_joint_seed'):
            historical_scores.append(extract(metrics))
            historical_speed.append(throughput(list(manifests.values()))['frame_weighted_fps'])
    # Only this already-completed original reference is averaged. No new seed runs.
    rows.insert(2, dict(Method='Original V2 (3-seed mean)', Detector='RTMDet', Association='PWQT',
                        Trainable_Assoc_Params=rows[2]['Trainable_Assoc_Params'], Parameters='HISTORICAL_ALREADY_COMPLETE',
                        **{k: statistics.mean(r[k] for r in historical_scores) for k in FIELDS},
                        E2E_FPS=statistics.mean(historical_speed),
                        Status='HISTORICAL_ALREADY_COMPLETE_THREE_SEED_REFERENCE',
                        Timing='Historical original reference completed before the new multi-seed cancellation.',
                        standard_deviation={k: statistics.pstdev(r[k] for r in historical_scores) for k in FIELDS}))
    for route in new_routes(args.nas):
        metrics, manifests = validate_route(route, out)
        checkpoint = args.nas / 'gmt_detector_comparison_20261023/final' / (route.key + '.frozen.pt')
        params = checkpoint_parameters(checkpoint) if route.association.startswith('PWQT') else 0
        frozen = json.dumps(manifests[1].get('settings', {}), sort_keys=True, separators=(',', ':'))
        record(route.key, route.name, 'GMT', route.association, route.predictions, route.metrics,
               metrics, manifests, params, frozen)
    native = rows[0]
    for row in rows:
        for field in ('HOTA', 'IDF1', 'CVIDF1', 'MOTA', 'IDSW'):
            row['delta_' + field + '_vs_GMT'] = row[field] - native[field]
    report = dict(status='COMPLETE_OFFICIAL_ROUTE_TABLE_SINGLE_SEED_EFFECT_GATE', official_rows=rows,
                  raw_GT_modified=False, new_TEST_used_for_selection=False, official_valid_GT_scenes=14,
                  inference_required_scenes=22,
                  speed_definition='Primary E2E_FPS = timestamps / summed full-video wall seconds for every route. Scene mean and harmonic mean are separate columns. Earlier hand-maintained new-route rows used scene arithmetic means.',
                  parameter_definition='Trainable_Assoc_Params counts learned association model parameters (native: transformer only; V2: frozen checkpoint state). Fixed rules and frozen VGGT point methods optimize zero added association parameters in this study. Native descriptor/ROI totals are separate in gmt_native_parameter_audit_20261024.json; do not treat zero as backbone size.',
                  multi_seed_policy=read_json(out / 'gmt_v2_joint_seed20261024_cancelled_20261024.json'),
                  scientific_status='NO_DEMONSTRATED_OVERALL_IDENTITY_GAIN_VS_GMT_NATIVE',
                  evidence_inventory='gmt_vggt_official_evidence_20261024.json',
                  remaining='Post-hoc attribution, parameter/hardware limitations and representative figures require final audit before goal completion.')
    (out / 'gmt_vggt_point_final_comparison.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    write_csv(out / 'gmt_vggt_point_final_comparison.csv', rows,
              ['Method', 'Detector', 'Association', 'Trainable_Assoc_Params', 'Parameters', *FIELDS,
               'E2E_FPS', 'E2E_FPS_mean_scene', 'E2E_FPS_harmonic_scene',
               *['delta_' + k + '_vs_GMT' for k in ('HOTA', 'IDF1', 'CVIDF1', 'MOTA', 'IDSW')], 'Status', 'Timing'])
    write_csv(out / 'gmt_vggt_point_per_scene.csv', scene_rows)
    write_csv(out / 'gmt_vggt_point_per_camera.csv', camera_rows)
    write_csv(out / 'gmt_vggt_point_fps.csv', fps_rows)
    write_csv(out / 'gmt_vggt_point_fps_per_scene.csv', scene_fps)
    (out / 'gmt_vggt_official_evidence_20261024.json').write_text(json.dumps(
        dict(status='VERIFIED_OFFICIAL_FILES_AND_FULLVIDEO_MANIFESTS', routes=evidence,
             prediction_hash_scope='New route manifest SHA equals official evaluator SHA; raw rehash is done by analyze_gmt_point_test.py.',
             reproduce='PYTHONPATH=. python scripts/build_gmt_point_comparison.py; original evaluator and frozen lock paths are in Gate artifacts.'), indent=2) + '\n')
    if args.diagnostics:
        diagnostics = read_json(args.diagnostics)
        if diagnostics['status'] != 'COMPLETE_POSTHOC_TEST_DIAGNOSTICS':
            raise ValueError('Incomplete error analysis')
        (out / 'gmt_vggt_point_error_attribution.json').write_text(json.dumps(diagnostics, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status=report['status'], official_rows=len(rows), per_scene_rows=len(scene_rows),
                          per_camera_rows=len(camera_rows), new_multi_seed_aggregate=False)), flush=True)


if __name__ == '__main__':
    main()
