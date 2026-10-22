#!/usr/bin/env python3
"""Collect actual larger-model fitting evidence; absent scores stay absent."""
import argparse
import csv
import json
from pathlib import Path

from cache_fulltrain import atomic_json, sha


REPO = Path(__file__).resolve().parents[1]


def collect(root):
    output = REPO / 'results'
    trials = []
    for phase in ('internal', 'final'):
        for path in sorted((root / phase).glob('*.json')):
            if path.name.endswith(('.frozen.json', '.delivery.json')):
                continue
            manifest = json.loads(path.read_text())
            trials.append({'phase': phase, 'trial': path.stem, 'profile': manifest['model_profile'],
                           'seed': manifest['seed'], 'status': manifest['status'],
                           'epochs_completed': manifest['epochs_completed'], 'train_scenes': manifest['train_scenes'],
                           'timestamps_per_epoch': manifest['total_timestamps'], 'history': manifest['history'],
                           'model_config': manifest['model_config'],
                           'checkpoint_sha256': sha(path.with_suffix('.pt')),
                           'source_provenance': manifest['source_provenance']})
    final_lock = root / 'final_test_lock.json'
    complete = final_lock.exists() and len([r for r in trials if r['phase'] == 'final' and r['status'] == 'FULLTRAIN_COMPLETE']) == 4
    report = {'status': 'FOUR_FINAL_FITS_FROZEN_AWAITING_TEST' if complete else 'IN_PROGRESS',
              'trials': trials, 'final_fitting_complete': complete, 'independent_TEST_complete': False,
              'registration_sha256': sha(REPO / 'configs/worldmot_v2_capacity_study.json'),
              'final_test_lock_sha256': sha(final_lock) if final_lock.exists() else None}
    atomic_json(output / 'worldmot_v2_capacity_training_curves.json', report)
    validation = []
    for path in sorted((root / 'selection').glob('*_epoch*.json')):
        record = json.loads(path.read_text())
        for row in record['threshold_candidates']:
            validation.append({'profile': record['profile'], 'epoch': record['epoch'],
                               'validation_loss': record['validation_loss']['total'], **row})
    if validation:
        with (output / 'worldmot_v2_capacity_TRAIN_validation.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['profile', 'epoch', 'validation_loss', 'null_margin',
                'birth_threshold', 'pair_threshold', 'identity_f1_surrogate', 'assigned', 'births'], extrasaction='ignore')
            writer.writeheader()
            writer.writerows(validation)
    document = REPO / 'docs/WORLDMOT_V2_CAPACITY_STUDY.md'
    text = document.read_text()
    marker = '<!-- BEGIN ACTUAL CAPACITY TRAINING -->'
    if marker in text:
        text = text[:text.index(marker)].rstrip()
    if trials:
        text = text.replace('Status: REGISTERED_AND_IMPLEMENTED. Formal larger-model training has not started.',
                            'Status: IN_PROGRESS. Formal training evidence is recorded below.')
    lines = [marker, '## Actual training evidence', '', 'Status: ' + report['status'] + '.', '',
             '| Trial | Profile | Seed | Completed epochs | Timestamps / epoch | Status |',
             '|---|---|---:|---:|---:|---|']
    for row in trials:
        lines.append('| ' + ' | '.join(str(row[k]) for k in ('trial', 'profile', 'seed', 'epochs_completed', 'timestamps_per_epoch', 'status')) + ' |')
    if not trials:
        lines += ['', 'No formal fitting checkpoint is present. Configuration, unit tests and a gradient probe do not count as completed training.']
    lines += ['', 'Independent TEST scores and real inference FPS are pending. TRAIN optimization losses and cached identity surrogates are not independent MOT accuracy.']
    document.write_text(text + '\n\n' + '\n'.join(lines) + '\n')
    return {'status': report['status'], 'actual_training_trials': len(trials)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('/data1/liuyeqiang/nas/A100/WorldMOT/capacity_20261022'))
    args = parser.parse_args()
    print(json.dumps(collect(args.root), indent=2))
