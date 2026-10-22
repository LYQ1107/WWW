#!/usr/bin/env python3
"""Check the matched small control against the original four-epoch prefix."""
import argparse
import ast
import json
import time
from pathlib import Path

import torch

from cache_fulltrain import atomic_json, sha


ORIGINAL = Path('/data1/liuyeqiang/nas/A100/WorldMOT/fulltrain_20261022/internal/joint.epoch4.pt')
CAPACITY = Path('/data1/liuyeqiang/nas/A100/WorldMOT/capacity_20261022')


def ready():
    try:
        manifest = json.loads((CAPACITY / 'internal/small.json').read_text())
        return manifest.get('epochs_completed', 0) >= 4 and (CAPACITY / 'internal/small.epoch4.pt').is_file()
    except (FileNotFoundError, json.JSONDecodeError):
        return False


def audit():
    torch.set_num_threads(2)
    newer = CAPACITY / 'internal/small.epoch4.pt'
    old = torch.load(ORIGINAL, map_location='cpu', weights_only=True)
    new = torch.load(newer, map_location='cpu', weights_only=True)
    a, b = old['manifest'], new['manifest']
    fields = ('model_config', 'cache_provenance', 'seed', 'train_scenes', 'total_timestamps',
              'tbptt', 'clip_curriculum', 'learning_rate', 'gradient_clip_norm', 'optimizer', 'max_missed')
    matched = {key: a.get(key) == b.get(key) for key in fields}
    repo = Path(__file__).resolve().parents[1]
    original_registration = json.loads((repo / 'configs/worldmot_v2_fulltrain_test.json').read_text())
    new_registration = json.loads((repo / 'configs/worldmot_v2_capacity_study.json').read_text())
    matched['registered_batch_sizes'] = original_registration['batch_size'] == new_registration['batch_size']
    shared_sources = {key: value for key, value in a['source_provenance']['files'].items() if key.startswith('worldmot/')}
    matched['shared_model_and_loss_source'] = all(b['source_provenance']['files'].get(key) == value
                                                 for key, value in shared_sources.items())
    def update_ast(path):
        module = ast.parse(path.read_text())
        function = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == 'train_step')
        return ast.dump(function, include_attributes=False)
    matched['frame_loss_and_update_function'] = update_ast(Path('/data1/liuyeqiang/WWW_fulltrain/worldmot/scripts/train_world_query.py')) == update_ast(repo / 'scripts/train_world_query.py')
    old_rates = [r.get('learning_rate', a['learning_rate']) for r in a['history'][:4]]
    new_rates = [r.get('learning_rate', b['learning_rate']) for r in b['history'][:4]]
    matched['first_four_learning_rates'] = old_rates == new_rates
    matched['first_four_clip_lengths'] = [r['clip_length'] for r in a['history'][:4]] == [r['clip_length'] for r in b['history'][:4]]
    matched['first_four_coverage'] = [r['coverage'] for r in a['history'][:4]] == [r['coverage'] for r in b['history'][:4]]
    differences = {key: float((value - new['model'][key]).abs().max())
                   for key, value in old['model'].items()}
    report = {'status': 'MATCHED_RECIPE_PREFIX_VERIFIED' if all(matched.values()) else 'RECIPE_MISMATCH_REQUIRES_AUDIT',
              'original_checkpoint_sha256': sha(ORIGINAL), 'new_small_checkpoint_sha256': sha(newer),
              'recipe_comparison': matched, 'first_four_learning_rates': old_rates,
              'batch_size_scope': 'agreement of frozen registrations/defaults; batch size is absent from the per-checkpoint manifests',
              'bitwise_model_equal': all(value == 0 for value in differences.values()),
              'max_parameter_absolute_difference': max(differences.values()),
              'per_tensor_max_absolute_difference': differences,
              'TEST_read_or_used': False, 'model_or_threshold_changed': False,
              'scope': 'read-only actual checkpoint comparison; recipe agreement does not guarantee bitwise CUDA determinism or tracking quality'}
    target = CAPACITY / 'small_control_prefix_audit.json'
    atomic_json(target, report)
    atomic_json(Path(__file__).resolve().parents[1] / 'results/worldmot_v2_capacity_control_prefix_audit.json', report)
    return {key: report[key] for key in ('status', 'bitwise_model_equal', 'max_parameter_absolute_difference')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    if args.wait:
        while not ready():
            time.sleep(30)
    print(json.dumps(audit() if ready() else {'status': 'WAITING_FOR_SMALL_CONTROL_EPOCH4'}, indent=2))
