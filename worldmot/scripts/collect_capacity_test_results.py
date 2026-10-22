#!/usr/bin/env python3
"""Compare four complete capacity TEST streams under the original official scope."""
import argparse
import collections
import json
import statistics
from pathlib import Path

from cache_fulltrain import atomic_json, sha
from collect_fulltrain_results import METRICS, csv_file, summary


REPO = Path(__file__).resolve().parents[1]
ORIGINAL = Path('/data1/liuyeqiang/nas/A100/WorldMOT/fulltrain_20261022')
SCOPE = [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 15, 16, 18, 19]


def collect(root):
    frozen = json.loads((root / 'final_test_lock.json').read_text())
    if len(frozen['checkpoints']) != 4:
        raise RuntimeError('capacity comparison requires all four registered final trials')
    metrics, timings = [], {}
    for trial in frozen['checkpoints']:
        label = f"capacity_{trial['profile']}_seed{trial['seed']}"
        prediction = root / 'test' / label
        manifests = [json.loads(path.read_text()) for path in sorted(prediction.glob('scene_*/manifest.json'))]
        if (len(manifests) != 22 or sum(m['frames'] for m in manifests) != 29019
                or any(m['status'] != 'COMPLETE' or not m['full_video'] or m['perception_cache_used']
                       or m['checkpoint_sha256'] != trial['sha256'] for m in manifests)):
            raise RuntimeError('capacity official comparison rejects partial or different-model TEST: ' + label)
        evaluation = json.loads((root / 'test_evaluation' / label / 'metrics.json').read_text())
        if evaluation['status'] != 'COMPLETE_VALID_SCOPE' or sorted(evaluation['evaluated_scenes']) != SCOPE:
            raise RuntimeError('capacity TEST must use the original common legal GT scope')
        native = evaluation['VisionTrack_MATLAB']['combined']
        values = {**evaluation['TrackEval']['combined'], **{k: native[k] for k in ('CVIDF1', 'CVMA')}}
        if sum(m['IDs_with_at_least_two_timestamps'] for m in manifests) <= 0:
            raise RuntimeError('capacity final model produced no persistent independent TEST IDs')
        seconds = sum(m['real_e2e_seconds'] for m in manifests)
        stages, latencies = collections.defaultdict(list), []
        for path in sorted(prediction.glob('scene_*/timings.jsonl')):
            for line in path.open():
                record = json.loads(line)
                latencies.append(record['rgb_to_delivered_ids_s'])
                stages['rgb_read'].append(record['rgb_read_s'])
                for name, value in record['stages_s'].items():
                    stages[name].append(value)
        if len(latencies) != 29019:
            raise RuntimeError('capacity timing samples do not cover all delivered timestamps')
        fps = 29019 / seconds
        row = {'method': label, 'profile': trial['profile'], 'seed': trial['seed'], **values,
               'real_e2e_fps': fps, 'checkpoint_sha256': trial['sha256'], 'TEST_scenes': 14, 'TEST_inference_scenes': 22}
        metrics.append(row)
        timings[label] = {'real_e2e_fps': fps, 'unit': 'synchronized dual-camera timestamps/second',
                          'total_s': seconds, 'e2e_latency': summary(latencies),
                          'stages': {k: summary(v) for k, v in stages.items()},
                          'four_GPU_aggregate_is_not_stream_FPS': True, 'perception_cache_used': False,
                          'timing_scope': manifests[0]['timing_scope']}
        omega = stages.get('omega_forward', [])
        if omega:
            timings[label]['omega_model_only_fps'] = len(omega) / sum(omega)
            timings[label]['omega_model_only_scope'] = 'same synchronized forward/camera decoding scope as original comparison; excludes RGB resize and geometry postprocess'
    large = [r for r in metrics if r['profile'] == 'large']
    if len(large) != 3:
        raise RuntimeError('capacity main comparison requires all three large seeds')
    result = {'status': 'COMPLETE_OFFICIAL_CAPACITY_TEST', 'metrics': metrics, 'evaluation_scope': SCOPE,
              'original_GT_modified': False, 'TEST_used_for_selection': False,
              'frozen_lock_sha256': sha(root / 'final_test_lock.json'),
              'large_three_seed_summary': {k: {'mean': statistics.mean(r[k] for r in large),
                  'sample_std': statistics.stdev(r[k] for r in large)} for k in METRICS + ['real_e2e_fps'] if all(k in r for r in large)},
              'capacity_effect_comparison': 'large seed 20261022 versus small seed 20261022 at the shared selected epoch budget'}
    original_file = Path('/data1/liuyeqiang/WWW_fulltrain/worldmot/results/worldmot_v2_test_official_metrics.json')
    if original_file.exists():
        original_report = json.loads(original_file.read_text())
        if original_report['status'] != 'COMPLETE_OFFICIAL_INDEPENDENT_TEST' or sorted(original_report['evaluation_scope']) != SCOPE:
            raise RuntimeError('original comparator report is incomplete or has a different scope')
        result['original_comparators'] = [r for r in original_report['metrics'] if r['method'] in ('v1', 'gmt_native', 'v2_joint_seed20261022')]
        result['short_budget_comparison_changes_both_capacity_and_budget'] = True
    output = REPO / 'results'
    atomic_json(output / 'worldmot_v2_capacity_official_metrics.json', result)
    atomic_json(output / 'worldmot_v2_capacity_fps.json', timings)
    rows = metrics + result.get('original_comparators', [])
    csv_file(output / 'worldmot_v2_capacity_vs_small_v1_gmt.csv', rows,
             ['method', 'profile', 'seed', *METRICS, 'real_e2e_fps', 'checkpoint_sha256'])
    document = REPO / 'docs/WORLDMOT_V2_CAPACITY_STUDY.md'
    text = document.read_text()
    marker = '<!-- BEGIN ACTUAL CAPACITY TEST -->'
    if marker in text:
        text = text[:text.index(marker)].rstrip()
    lines = [marker, '## Actual independent TEST results', '', 'Four frozen capacity trials completed all 22 raw RGB videos. Official metrics use the same 14 valid original-GT scenes.', '',
             '| Method | HOTA | AssA | IDF1 | CVIDF1 | IDSW | E2E pairs/s |', '|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append('| ' + row['method'] + ' | ' + ' | '.join(f'{row[k]:.3f}' for k in ('HOTA', 'AssA', 'IDF1', 'CVIDF1', 'IDSW', 'real_e2e_fps')) + ' |')
    lines += ['', 'Use the matched-epoch seed-20261022 large/small pair for a capacity conclusion. The original four-epoch comparison also changes training budget and learning-rate schedule. Large-model mean and sample deviation are saved for all three seeds; no three-seed variance is claimed for the single small extension control.']
    document.write_text(text + '\n\n' + '\n'.join(lines) + '\n')
    return {'status': result['status'], 'official_methods': len(metrics)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('/data1/liuyeqiang/nas/A100/WorldMOT/capacity_20261022'))
    args = parser.parse_args()
    print(json.dumps(collect(args.root), indent=2))
