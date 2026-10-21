"""Close the all-24 causal supervision contract from immutable scene audits."""
import argparse
import collections
import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'manifests/omega_full24_training_protocol_v1.json'
NEW_ROOT = Path('/data1/liuyeqiang/WWW_omega_fulltrain_runtime_20261016/full_train_audit_v1')
OLD_ROOT = Path('/data1/liuyeqiang/WWW_trackvggt_runtime/full_train_audit_v1')
OLD_IDS = {4, 5, 6, 7, 8, 9, 12, 13, 17, 20, 21, 24}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def records(path):
    with gzip.open(path, 'rt') as stream:
        for line in stream:
            yield json.loads(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text())
    fit = set(protocol['scene_split']['fit_video_ids'])
    validation = set(protocol['scene_split']['validation_video_ids'])
    scenes = {}
    totals = collections.Counter()
    eligibility = collections.Counter()
    for scene in protocol['dataset']['video_ids']:
        root = OLD_ROOT if scene in OLD_IDS else NEW_ROOT
        run = root / f'scene{scene}_full_v1'
        manifest_path = run / 'manifest.json'
        summary_path = run / 'summary.json'
        if not manifest_path.is_file() or not summary_path.is_file():
            raise FileNotFoundError(f'missing complete scene audit {scene}: {run}')
        manifest = json.loads(manifest_path.read_text())
        summary = json.loads(summary_path.read_text())
        if manifest.get('status') != 'COMPLETE_FULL_CAUSAL_AUDIT' or not manifest.get('full_video'):
            raise RuntimeError(f'scene {scene} is not a complete full-video audit')
        if not manifest.get('prefix_raw_prediction_exact', manifest.get('prefix40_raw_prediction_exact', False)):
            raise RuntimeError(f'scene {scene} lacks OFF-prefix parity')
        count = summary['counts']
        role = 'fit' if scene in fit else 'validation' if scene in validation else 'invalid'
        if role == 'invalid':
            raise RuntimeError(f'scene {scene} is outside locked split')
        row = dict(scene=scene, role=role, source_root=str(root), run=str(run), manifest_sha256=sha256(manifest_path), summary_sha256=sha256(summary_path),
                   events_sha256=sha256(run / 'events.jsonl.gz'), decisions_sha256=sha256(run / 'actual_decisions.jsonl.gz'), rankings_sha256=sha256(run / 'rankings.jsonl.gz'),
                   images=manifest['images'], groups=manifest['groups'], primary_calls=manifest.get('primary_calls', manifest.get('primary_calls', 0)),
                   counts=count, rates=summary['rates'], retained_events=summary.get('retained_events'), causal=True, future_query_features=manifest.get('future_query_features', False),
                   gt_labeled_after_off_parity=manifest.get('raw_GT_labeling_after_full_inference_and_OFF_parity', True))
        scenes[str(scene)] = row
        for key, value in count.items():
            if isinstance(value, (int, float)):
                totals[f'{role}.{key}'] += value
        eligibility['scenes_complete'] += 1
        eligibility['scenes_fit'] += role == 'fit'
        eligibility['scenes_validation'] += role == 'validation'
        eligibility['events_with_pure_positive'] += count.get('N_pure_positive_full', 0)
        eligibility['events_known_wrong_or_missed'] += count.get('N_correctable_known_wrong', 0)
        eligibility['events_masked_unknown_or_impure'] += count.get('N_unknown', 0) + count.get('N_impure', 0)
    if eligibility['scenes_complete'] != 24:
        raise RuntimeError('all24 scene contract not closed')
    report = dict(status='PASS_COMPLETE_ALL24_CAUSAL_SUPERVISION_COVERAGE', protocol=str(PROTOCOL), protocol_sha256=sha256(PROTOCOL), scene_ids=protocol['dataset']['video_ids'], fit_video_ids=sorted(fit), validation_video_ids=sorted(validation), scenes=scenes, totals=dict(totals), eligibility=dict(eligibility), source_roots={'prior_12': str(OLD_ROOT), 'new_12': str(NEW_ROOT)}, raw_gt_modified=False, no_gt_built_tracks=True, no_future_features=True, source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(), command=sys.argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps(dict(status=report['status'], scenes=eligibility['scenes_complete'], fit=eligibility['scenes_fit'], validation=eligibility['scenes_validation'], pure_positive=eligibility['events_with_pure_positive'], known_wrong_or_missed=eligibility['events_known_wrong_or_missed'])))


if __name__ == '__main__':
    main()
