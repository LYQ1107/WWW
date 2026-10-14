"""Audit actual cached online prefixes without mistaking unit tests for runs."""
import argparse
from common import ROOT, train_data, sha256, save_json
import json
from pathlib import Path


def compare_prefix(short, long, image_ids):
    """Inputs contain parsed manifests, predictions and scheduler traces."""
    sm, lm = short['manifest'], long['manifest']
    n = sm['synchronized_groups']
    if not 0 < n < lm['synchronized_groups']:
        raise ValueError('strictly increasing prefix lengths required')
    for report in (sm, lm):
        if not report['causal_online'] or report['full_video']:
            raise ValueError('two explicit causal partial runs required')
        if not report['geo_options']['REUSE_TARGET_FEATURES']:
            raise ValueError('target feature reuse must be enabled')
        if report['geo_failures']:
            raise ValueError('failed encoder/reuse run needs separate failure audit')
    for field in ('video_id', 'checkpoint_sha256', 'config_sha256',
                  'annotation_sha256', 'head_checkpoint_sha256', 'geo_options',
                  'seed', 'source_hashes', 'validation_corrected_gmt_camera_index',
                  'validation_reversed_vggt_order'):
        if sm[field] != lm[field]:
            raise ValueError('paired prefix mismatch: ' + field)
    if len(short['schedule']) != n or len(long['schedule']) != lm['synchronized_groups']:
        raise ValueError('one schedule observation per group required')
    prefix = [row for row in long['predictions'] if row['image_id'] in image_ids]
    if any(row['image_id'] not in image_ids for row in short['predictions']):
        raise ValueError('short run emitted an image outside declared prefix')
    exact_predictions = short['predictions'] == prefix
    exact_schedule = short['schedule'] == long['schedule'][:n]
    cache = sm['target_feature_cache']
    # Covers retained tensor payload, not total process or original GTA memory.
    dim = sm['head_projection_dimensions']
    bound = cache['max_cameras'] * cache['max_targets_per_camera'] * (1152 + 4 + dim) * 4
    if cache['peak_payload_bytes'] > bound:
        raise ValueError('retained cache tensor payload exceeds declared bound')
    if not cache['stats']['reused_targets']:
        raise ValueError('prefix never reused a target; cannot validate active reuse')
    calls = sum(bool(row['chosen']) for row in short['schedule']) * sm['planned_update_cost']
    actual = sm['total_vggt_calls']
    if actual != calls:
        raise ValueError('VGGT call count differs from successful chosen update cost')
    return dict(status='PASS' if exact_predictions and exact_schedule else 'FAIL',
                groups=n, exact_identity_output_prefix=exact_predictions,
                exact_scheduler_prefix=exact_schedule,
                detections=len(short['predictions']), actual_vggt_calls=actual,
                target_cache=cache, retained_tensor_payload_bound_bytes=bound,
                limitations='Cache source ages are enforced in implementation and tested separately; these runs did not log per-target source-age traces. Prefix equality checks actual identity outputs and scheduling, not every hidden tensor.')


def read_run(path):
    manifest = json.loads((path/'manifest.json').read_text())
    for filename, field in [('predictions.json', 'predictions_sha256'),
                            ('schedule_trace.json', 'schedule_trace_sha256')]:
        if sha256(path/filename) != manifest[field]:
            raise ValueError('changed runtime artifact: ' + filename)
    return dict(manifest=manifest,
                predictions=json.loads((path/'predictions.json').read_text()),
                schedule=json.loads((path/'schedule_trace.json').read_text()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--short-run', type=Path, required=True)
    parser.add_argument('--long-run', type=Path, required=True)
    parser.add_argument('--tag', required=True)
    args = parser.parse_args()
    short, long = [read_run(path) for path in (args.short_run, args.long_run)]
    n, scene = short['manifest']['synchronized_groups'], short['manifest']['video_id']
    records = [r for r in train_data()['images'] if r['video_id'] == scene]
    frames = sorted({r['frame_id'] for r in records})[:n]
    image_ids = {r['id'] for r in records if r['frame_id'] in frames}
    if len(image_ids) != short['manifest']['images']:
        raise ValueError('annotation image count disagrees with observed prefix')
    report = compare_prefix(short, long, image_ids)
    report.update(scope='two actual synchronous causal cached VGGT runs; exact serialized identity output and scheduler prefix; no GT used',
                  source_runs=[str(args.short_run), str(args.long_run)],
                  manifest_sha256=[sha256(p/'manifest.json') for p in (args.short_run, args.long_run)],
                  audit_source_sha256=sha256(Path(__file__)))
    save_json(ROOT/'results'/f'{args.tag}.json', report)
    if report['status'] != 'PASS':
        raise SystemExit('actual cached prefix differs')


if __name__ == '__main__':
    main()
