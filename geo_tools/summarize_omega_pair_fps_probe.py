"""Seal the bounded probe without treating module timings as system FPS."""
import json
from pathlib import Path

from omega_pair_geometry_probe import PLAN, ROOT, RUNTIME, save, sha256


def main():
    plan = json.loads(PLAN.read_text())
    geometry_path = ROOT / 'results/omega_pair_geometry_fps_probe_v1.json'
    geometry = json.loads(geometry_path.read_text())
    configs = {}
    phase_seconds = 0.
    for config in plan['configs']:
        name = config['name']
        paths = dict(initial_official_auto_bf16=RUNTIME / name / 'manifest.json',
                     native_fp16_eager=RUNTIME / 'native_fp16' / name / 'manifest.json',
                     native_fp16_cuda_graph=RUNTIME / 'cuda_graph_fp16' / name / 'manifest.json')
        reports = {key: json.loads(path.read_text()) for key, path in paths.items()}
        if not all(r['finite_pairs'] == 54 and r['total_pairs'] == 54 for r in reports.values()):
            raise ValueError('incomplete/nonfinite experiment retained, cannot seal as successful')
        graph = reports['native_fp16_cuda_graph']
        parity = max(v for row in graph['rows'] for v in row['eager_parity_max_normalized_error'].values())
        if parity > 1e-4:
            raise ValueError('graph parity gate failed')
        records = {}
        for mode, report in reports.items():
            phase_seconds += report['process_seconds']
            records[mode] = dict(manifest=str(paths[mode]), manifest_sha256=sha256(paths[mode]),
                                 compute=report['compute'], serial_read_to_geometry=report['serial_read_to_geometry'],
                                 model_pairs_per_second=1000 / report['compute']['mean_ms'],
                                 read_to_geometry_pairs_per_second=1000 / report['serial_read_to_geometry']['mean_ms'],
                                 peak_allocated_bytes=report['peak_allocated_bytes'],
                                 load_seconds=report['load_seconds'])
        counters = geometry['configs'][name]['scores']['projected_support']
        answered = counters['correct'] + counters['wrong']
        configs[name] = dict(input_b_s_c_h_w=graph['rows'][0]['shape'], measurements=records,
                             parity_max_normalized_error=parity,
                             graph_setup_seconds=graph['graph_setup_seconds'],
                             projected_support_ranking=counters,
                             answered_ranking_accuracy=counters['correct'] / answered,
                             coverage=answered / counters['eligible'],
                             foot_3d_ranking=geometry['configs'][name]['scores']['foot_3d'],
                             cpu_geometry=geometry['configs'][name]['cpu_geometry_all_candidate_latency'],
                             detector_height_network_pixels=geometry['configs'][name]['detector_height_network_pixels'],
                             reduced_resolution_512_weights=config['resolution'] != 512)
    baselines = {}
    for scene in (2, 8, 17):
        path = RUNTIME / f'gmt_scene{scene}.json'
        report = json.loads(path.read_text())
        baselines[str(scene)] = {key: value for key, value in report.items() if key != 'rows'}
        baselines[str(scene)].update(manifest=str(path), manifest_sha256=sha256(path))
    raw = RUNTIME / 'posthoc_score_matrices.json'
    cache_bytes = sum(p.stat().st_size for p in RUNTIME.rglob('*') if p.is_file())
    if cache_bytes > plan['resources']['cache_limit_bytes']:
        raise ValueError('research cache budget exceeded')
    if phase_seconds > plan['resources']['max_inference_process_seconds_sum']:
        raise ValueError('reported inference-phase budget exceeded')
    report = dict(status='COMPLETE_BOUNDED_RECONSTRUCTION_FEASIBILITY_AND_FPS_RESEARCH',
                  plan=str(PLAN), plan_sha256=sha256(PLAN), pairs=54, fit_scenes=18,
                  physical_gpus_requested=[2, 3, 4, 5], checkpoint_sha256=plan['checkpoint_sha256'],
                  official_source_commit=plan['source_commit'], configs=configs,
                  precision_correction='Initial manifests mislabel FP16; actual official automatic BF16 path on emulating V100 preserved and separately audited. Native FP16 and graph runs explicit.',
                  geometry_summary=str(geometry_path), geometry_summary_sha256=sha256(geometry_path),
                  score_matrices=str(raw), score_matrices_sha256=sha256(raw),
                  gmt_prefix_baselines=baselines,
                  resources=dict(research_cache_bytes=cache_bytes, cache_limit_bytes=2147483648,
                                 reported_inference_phase_seconds_sum=phase_seconds,
                                 time_scope='per-child internal timer excludes Python/Torch import; not total launch wall time or GPU utilization integral'),
                  interpretation=dict(reconstruction='All 54 two-view pairs yield finite pose/depth at all four sizes; finite geometry does not certify physical accuracy.',
                                      identity='Candidate-row ranking signal with substantial errors; only cases with a known positive and negative in the other view; no same/different probability calibration.',
                                      speed='CUDA Graph parity exact on all four configurations, 216 physical config-pairs; models resident; neither model-only nor serial reconstruction numbers include detector/GMT.',
                                      recommendation='Reuse original GMT detections; full-view joint pose/depth; sparse person-point lift/projection; reliable geometry as a candidate gate; static-camera geometry cached and mobile-camera geometry refreshed/pose-tracked.',
                                      limitations=['development TRAIN already seen; not blind validation',
                                                   'two directions per physical pair are correlated; 764 rows are not 764 independent identities',
                                                   'no background geometry certification in this new probe',
                                                   'no absent-positive/open-world rejection test',
                                                   'no trajectory identity propagation or IDF1/HOTA gain tested',
                                                   'no four-GPU end-to-end ordered online pipeline tested',
                                                   'separate CUDA context/model residency coexistence with GMT not measured']),
                  gmt_integration='CLOSED; prior preregistered gates unchanged',
                  full_video_system_fps='NOT_MEASURED', gmt_plus_omega_system_fps='NOT_MEASURED',
                  test='NOT_ACCESSED', checkpoint_416='NOT_EXECUTED')
    path = ROOT / 'results/omega_pair_reconstruction_fps_research_v1.json'
    save(path, report)
    print(json.dumps(dict(report=str(path), cache_bytes=cache_bytes,
                          reported_inference_phase_seconds_sum=phase_seconds), indent=2))


if __name__ == '__main__':
    main()
