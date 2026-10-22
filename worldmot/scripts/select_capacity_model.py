#!/usr/bin/env python3
"""Select a shared budget for the registered capacity comparison using TRAIN."""
import argparse
import json
from pathlib import Path

import torch

from cache_fulltrain import atomic_json, sha
from experiment_provenance import source_stamp
from select_query_model import calibration, continuous_audit, validation_loss
from worldmot.training.clip_dataset import WorldQueryClipDataset


def select(root, registration, device):
    torch.set_num_threads(2)
    scenes = registration['validation_scenes']
    dataset = WorldQueryClipDataset([registration['immutable_cache']], scenes=scenes, clip_length=32)
    if sorted(dataset.scenes) != sorted(scenes) or sum(map(len, dataset.scenes.values())) != 5860:
        raise RuntimeError('capacity selection requires all six complete held-out TRAIN scenes')
    output = root / 'selection'
    output.mkdir(exist_ok=True)
    rows = {}
    for profile in ('large', 'small'):
        rows[profile] = []
        for epoch in registration['epoch_candidates']:
            checkpoint = root / 'internal' / f'{profile}.epoch{epoch}.pt'
            record = output / f'{profile}_epoch{epoch}.json'
            checkpoint_sha = sha(checkpoint)
            if record.exists():
                row = json.loads(record.read_text())
                if row['checkpoint_sha256'] != checkpoint_sha or row['source_provenance'] != source_stamp():
                    raise RuntimeError('preserve different-model capacity selection before rerunning')
            else:
                loss = validation_loss(dataset, checkpoint, device)
                grid, best = calibration(dataset, checkpoint, device)
                row = {'profile': profile, 'epoch': epoch, 'checkpoint_sha256': checkpoint_sha,
                       'source_provenance': source_stamp(), 'validation_loss': loss,
                       'threshold_candidates': grid, 'best': best, 'TEST_read_or_used': False}
                atomic_json(record, row)
            rows[profile].append(row)
            print(json.dumps({'capacity_validation': profile, 'epoch': epoch,
                              'loss': row['validation_loss']['total'],
                              'identity_F1_SURROGATE': row['best']['identity_f1_surrogate']}), flush=True)
    maximum = max(row['best']['identity_f1_surrogate'] for row in rows['large'])
    eligible = [row for row in rows['large'] if row['best']['identity_f1_surrogate'] >= maximum - .002]
    epoch = min(row['epoch'] for row in eligible)
    result = {'status': 'FROZEN_FROM_TRAIN_ONLY', 'selected_epochs': epoch,
              'shared_epoch_budget_for_capacity_comparison': True,
              'epoch_selection_rule': registration['epoch_selection'],
              'TEST_used_for_selection': False, 'source_provenance': source_stamp(),
              'validation_scenes': scenes, 'profiles': {},
              'per_epoch': {p: [{'epoch': r['epoch'], 'loss': r['validation_loss'], 'best': r['best'],
                                 'checkpoint_sha256': r['checkpoint_sha256']} for r in rr] for p, rr in rows.items()}}
    for profile in ('large', 'small'):
        row = next(row for row in rows[profile] if row['epoch'] == epoch)
        thresholds = {key: row['best'][key] for key in ('null_margin', 'birth_threshold', 'pair_threshold')}
        checkpoint = root / 'internal' / f'{profile}.epoch{epoch}.pt'
        delivery = continuous_audit(dataset, checkpoint, thresholds, device)
        result['profiles'][profile] = {'thresholds': thresholds, 'TRAIN_delivery': delivery,
                                       'selected_checkpoint_sha256': sha(checkpoint)}
        if delivery['assigned_observations'] <= 0 or delivery['max_ID_distinct_timestamps'] < 2:
            result['status'] = 'FAILED_TRAIN_DELIVERY_GATE'
            atomic_json(output / 'selection.json', result)
            raise RuntimeError(profile + ' has no persistent TRAIN identity delivery; retain failure before TEST')
    atomic_json(output / 'selection.json', result)
    return {'status': result['status'], 'selected_epochs': epoch}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    print(json.dumps(select(args.root, json.loads(args.registration.read_text()), args.device), indent=2))
