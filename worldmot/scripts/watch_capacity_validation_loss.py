#!/usr/bin/env python3
"""Read-only full TRAIN-validation loss curves while fixed training continues.

This CPU diagnostic never supplies selections to the registered GPU selector.
It uses complete held-out TRAIN clips, not TEST or an inference FPS proxy.
"""
import argparse
import json
import time
from pathlib import Path

import torch

from cache_fulltrain import atomic_json, sha
from experiment_provenance import source_stamp
from select_query_model import validation_loss
from worldmot.training.clip_dataset import WorldQueryClipDataset


REPO = Path(__file__).resolve().parents[1]
ROOT = Path('/data1/liuyeqiang/nas/A100/WorldMOT/capacity_20261022')


def load(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def main(watch):
    torch.set_num_threads(2)
    registration = load(REPO / 'configs/worldmot_v2_capacity_study.json')
    output = ROOT / 'CPU_validation_loss_diagnostic.json'
    report = load(output) or {'status': 'WAITING_FOR_FIXED_CANDIDATES', 'records': [],
                             'selection_uses_this_file': False, 'TEST_read_or_used': False,
                             'teacher_forced_loss_is_not_MOT_accuracy': True,
                             'device': 'cpu', 'validation_scenes': registration['validation_scenes']}
    seen = {(row['profile'], row['epoch']) for row in report['records']}
    dataset = None
    while True:
        for profile in ('large', 'small'):
            manifest = load(ROOT / 'internal' / (profile + '.json'))
            for epoch in registration['epoch_candidates']:
                checkpoint = ROOT / 'internal' / f'{profile}.epoch{epoch}.pt'
                if ((profile, epoch) in seen or manifest.get('epochs_completed', 0) < epoch
                        or not checkpoint.exists()):
                    continue
                saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
                if saved['epoch'] != epoch or saved['manifest']['source_provenance'] != source_stamp():
                    raise RuntimeError('candidate checkpoint/source differs from registered diagnostic')
                if dataset is None:
                    dataset = WorldQueryClipDataset([registration['immutable_cache']],
                        scenes=registration['validation_scenes'], clip_length=32)
                    if sum(map(len, dataset.scenes.values())) != 5860:
                        raise RuntimeError('CPU loss diagnostic requires all 5860 held-out TRAIN timestamps')
                values = validation_loss(dataset, checkpoint, 'cpu')
                record = {'profile': profile, 'epoch': epoch, 'checkpoint_sha256': sha(checkpoint),
                          'values': values, 'source_provenance': source_stamp()}
                report['records'].append(record)
                report['status'] = 'PARTIAL_FIXED_TRAIN_VALIDATION_DIAGNOSTIC'
                seen.add((profile, epoch))
                atomic_json(output, report)
                atomic_json(REPO / 'results/worldmot_v2_capacity_CPU_validation_loss.json', report)
                print(json.dumps({'profile': profile, 'epoch': epoch, 'full_TRAIN_validation_loss': values['total']}), flush=True)
        if len(seen) == 2 * len(registration['epoch_candidates']):
            report['status'] = 'ALL_FIXED_TRAIN_VALIDATION_LOSS_POINTS_COMPLETE'
            atomic_json(output, report)
            atomic_json(REPO / 'results/worldmot_v2_capacity_CPU_validation_loss.json', report)
            break
        if not watch or load(ROOT / 'experiment_status.json').get('stage', '').startswith('FAILED'):
            break
        time.sleep(30)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    main(args.watch)
