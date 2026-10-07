"""Frozen-checkpoint component ablations; no training or GT-dependent actions."""
from __future__ import annotations

import argparse
from collections import Counter
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BASELINE = Path('/home/liuyeqiang/WWW_jev_rng_v4_runtime/segmented_gate_20261007_v4')
CHECKPOINT = BASELINE / 'small_h8_training_current_head_video06_video07/methods_v1/jev/seed_20261003/calibration/model_calibrated.pth'
VARIANTS = {
    'A0': (),
    'A1': ('MATCH_DECISION',),
    'A2': ('MATCH_DECISION', 'REACTIVATION_DECISION'),
    'A3': ('MATCH_DECISION', 'MEMORY_DECISION'),
    'A4': ('MATCH_DECISION', 'MEMORY_DECISION', 'REACTIVATION_DECISION'),
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.replace(tmp, path)


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--variant', required=True, choices=VARIANTS)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-frame', type=int)
    args = p.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ.update(JEV_PILOT_ROOT=str(output), JEV_VIDEO_ID='1',
                      JEV_TRACE_PATH='/home/liuyeqiang/WWW_jev_rng_v4_runtime/segmented_native_trace_20261007/video01/trace_video_01.jsonl',
                      JEV_RECORDS_PATH=str(BASELINE / 'video01_records.jsonl'))
    for path in (ROOT, ROOT / 'third_party/CenterNet2', ROOT / 'reproduction_tools'):
        sys.path.insert(0, str(path))
    import run_early_pilot_tracking as pilot
    controlled = frozenset(VARIANTS[args.variant])
    original_choose = pilot.choose
    decision_log = (output / 'online_decisions.jsonl').open('w')
    last_update = 0.0

    def choose(policy, feature, question, legal, off_action, context=None):
        nonlocal last_update
        # Routing changes only which policy owns each typed decision.
        action = original_choose(policy, feature, question, legal, off_action, context=context) if question in controlled else off_action
        if action not in legal:
            raise RuntimeError('routed policy selected an illegal action')
        item = {'question': question, 'action': action, 'off_action': off_action,
                'legal_actions': list(legal), 'controller': 'JEV' if question in controlled else 'GMT',
                'feature_vector': feature.detach().cpu().tolist(), 'context': context or {}}
        decision_log.write(json.dumps(item, sort_keys=True, allow_nan=False) + '\n')
        stamp = time.monotonic()
        if stamp - last_update > 2:
            decision_log.flush()
            save(output / 'runtime_status.json', {'phase': 'inference', 'variant': args.variant, 'updated_utc': now(),
                 'pid': os.getpid(), 'question': question, 'frame': (context or {}).get('frame'), 'view': (context or {}).get('view')})
            last_update = stamp
        return action

    pilot.choose = choose
    annotations, subset, image_lookup, by_key, records = pilot.load_inputs()
    modules = pilot.load_runtime_modules()
    names = ['build_formal_gmt_engine', 'MutableGMTState', 'FrozenPerceptionCache', 'JEVRuntimePolicy',
             'build_controller_from_checkpoint', 'build_state_features', 'association_window_length',
             'candidate_entropy', 'count_memory_observations', 'count_track_history', 'feature_names',
             'legacy_acceptance_threshold']
    kwargs = dict(zip(names, modules))
    kwargs.pop('feature_names')
    name = 'gmt_off' if args.variant == 'A0' else 'jev'
    result = pilot.run_method(name, None if args.variant == 'A0' else CHECKPOINT,
                            image_lookup=image_lookup, by_key=by_key, records=records,
                            feature_source_mode='runtime', parity_report=None, device='cuda:0',
                            max_frame=args.max_frame, **kwargs)
    decision_log.close()
    save(output / 'runtime_status.json', {'phase': 'evaluation', 'variant': args.variant, 'updated_utc': now(), 'pid': os.getpid()})
    dataset_root = pilot.prepare_eval_dataset(subset)
    prepared, evaluated = pilot.run_eval(name, Path(result['predictions']), dataset_root)
    metrics = pilot.extract_metrics(evaluated)
    decisions = [json.loads(line) for line in (output / 'online_decisions.jsonl').read_text().splitlines()]
    by_question = {}
    for question in ('MATCH_DECISION', 'MEMORY_DECISION', 'REACTIVATION_DECISION'):
        counts = Counter(d['action'] for d in decisions if d['question'] == question)
        total = sum(counts.values())
        by_question[question] = {'count': total, 'actions': dict(counts), 'rates': {k: v / max(1,total) for k,v in counts.items()}}
    summary = {'status': 'PASS', 'classification': 'TRAIN_HELD_OUT_FROZEN_CHECKPOINT_COMPONENT_DIAGNOSTIC',
               'variant': args.variant, 'controlled_questions': sorted(controlled), 'metrics': metrics,
               'result': result, 'action_counts_by_question': by_question,
               'checkpoint': None if args.variant == 'A0' else str(CHECKPOINT),
               'checkpoint_sha256': None if args.variant == 'A0' else sha(CHECKPOINT),
               'predictions_sha256': sha(result['predictions']), 'decisions_sha256': sha(result['decisions']),
               'online_decisions_sha256': sha(output / 'online_decisions.jsonl'),
               'legacy_label_based_counts_are_off_state_proxies_not_causal_truth': True,
               'trained': False, 'official_test_read': False, 'full24_authorized': False,
               'prepared': str(prepared), 'evaluation': str(evaluated), 'completed_utc': now()}
    if args.max_frame is None and args.variant in {'A0','A4'}:
        original = BASELINE / 'closed_loop/tracking_predictions' / (name + '.json')
        previous = json.loads((BASELINE / 'final_summary.json').read_text())['models'][name]['metrics']
        summary['baseline_reproduction'] = {'predictions_sha_identical': sha(original) == summary['predictions_sha256'],
                                            'max_metric_error': max(abs(metrics[k]-previous[k]) for k in metrics)}
        if not summary['baseline_reproduction']['predictions_sha_identical'] or summary['baseline_reproduction']['max_metric_error'] > 1e-8:
            raise RuntimeError('frozen baseline prediction/metric reproduction failed')
    save(output / 'result.json', summary)
    save(output / 'runtime_status.json', {'phase': 'COMPLETE', 'variant': args.variant, 'updated_utc': now()})
    print(json.dumps({'variant': args.variant, 'metrics': metrics, 'status': 'PASS'}), flush=True)


if __name__ == '__main__':
    main()
