#!/usr/bin/env python3
"""Run the isolated, preregistered larger-PWQT TRAIN study on one GPU slot."""
import argparse
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

from cache_fulltrain import atomic_json, sha
from experiment_provenance import source_stamp
from select_query_model import continuous_audit
from worldmot.training.clip_dataset import WorldQueryClipDataset


REPO = Path(__file__).resolve().parents[1]
REGISTRATION = REPO / 'configs/worldmot_v2_capacity_study.json'
ORIGINAL = Path('/data1/liuyeqiang/nas/A100/WorldMOT/fulltrain_20261022')


def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def execute(root, name, command):
    environment = {**os.environ, 'PYTHONPATH': str(REPO), 'OMP_NUM_THREADS': '2',
                   'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '2'}
    with (root / 'logs' / (name + '.log')).open('a') as stream:
        stream.write('\nCOMMAND ' + json.dumps(command) + '\n')
        stream.flush()
        completed = subprocess.run(command, cwd=REPO, env=environment, stdout=stream, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise RuntimeError(name + ' failed; see ' + str(root / 'logs' / (name + '.log')))


def state(root, stage, **values):
    record = {'stage': stage, 'updated_utc': stamp(), **values}
    atomic_json(root / 'experiment_status.json', record)
    with (root / 'events.jsonl').open('a') as stream:
        stream.write(json.dumps(record) + '\n')
    print(json.dumps(record), flush=True)


def original_training_gate():
    trials = [('joint', seed) for seed in (20261022, 20261023, 20261024)]
    trials += [(method, 20261022) for method in ('no_omega', 'no_appearance', 'two_stage', 'no_geometry')]
    hashes = {}
    for method, seed in trials:
        checkpoint = ORIGINAL / 'final' / f'{method}_seed{seed}.pt'
        manifest = json.loads(checkpoint.with_suffix('.json').read_text())
        if (manifest['status'] != 'FULLTRAIN_COMPLETE' or not manifest['full_train_coverage']
                or manifest['train_scenes'] != list(range(1, 25)) or manifest['total_timestamps'] != 28754):
            raise RuntimeError('original seven fits must complete before the capacity study starts')
        hashes[checkpoint.stem] = sha(checkpoint)
    return hashes


def fit(root, registration, profile, seed, phase, epochs, scenes):
    output = root / phase / (profile + (f'_seed{seed}' if phase == 'final' else '') + '.pt')
    if output.exists():
        saved = torch.load(output, map_location='cpu', weights_only=True)
        manifest = saved['manifest']
        if manifest['source_provenance'] != source_stamp() or manifest['model_profile'] != profile:
            raise RuntimeError('capacity checkpoint source/profile differs; preserve previous run')
        if saved['epoch'] == epochs and manifest['train_scenes'] == scenes:
            return output
    command = [sys.executable, 'scripts/train_world_query.py', '--cache', registration['immutable_cache'],
               '--output', str(output), '--epochs', str(epochs), '--seed', str(seed),
               '--device', 'cuda:0', '--model-profile', profile, '--lr', str(registration['learning_rate']),
               '--lr-milestones', *map(str, registration['lr_milestones']),
               '--lr-drop-factor', str(registration['lr_drop_factor']), '--scenes', *map(str, scenes)]
    if output.exists():
        command += ['--resume', str(output)]
    execute(root, phase + '_' + output.stem, command)
    return output


def train(root):
    torch.set_num_threads(2)
    registration = json.loads(REGISTRATION.read_text())
    for name in ('logs', 'internal', 'selection', 'final'):
        (root / name).mkdir(parents=True, exist_ok=True)
    verification = ORIGINAL / 'cache_verification/coverage.json'
    if sha(verification) != registration['cache_verification_sha256']:
        raise RuntimeError('shared immutable TRAIN verification differs from registration')
    prior_checkpoints = original_training_gate()
    scripts = {name: sha(REPO / 'scripts' / name) for name in (
        'run_capacity_study.py', 'select_capacity_model.py', 'probe_query_capacity.py', 'collect_capacity_results.py')}
    signature = {'registration_sha256': sha(REGISTRATION), 'source_provenance': source_stamp(),
                 'execution_script_sha256': scripts, 'original_final_checkpoint_sha256': prior_checkpoints}
    lock_path = root / 'study_lock.json'
    if lock_path.exists():
        previous = json.loads(lock_path.read_text())
        if previous['signature'] != signature:
            raise RuntimeError('capacity study source/registration differs; retain prior experiment')
    else:
        atomic_json(lock_path, {'status': 'FROZEN_BEFORE_CAPACITY_FITTING', 'frozen_utc': stamp(),
                               'signature': signature, 'TEST_used_for_selection': False,
                               'physical_gpu': os.environ.get('CUDA_VISIBLE_DEVICES')})
    state(root, 'CAPACITY_TRAIN_STARTING', profiles=['large', 'small'], max_epochs=registration['max_epochs'])
    probe = root / 'real_TRAIN_capacity_probe.json'
    if not probe.exists():
        execute(root, 'real_TRAIN_capacity_probe', [sys.executable, 'scripts/probe_query_capacity.py', '--output', str(probe)])
    for profile in ('large', 'small'):
        state(root, 'CAPACITY_INTERNAL_TRAIN_RUNNING', profile=profile, epochs=registration['max_epochs'])
        fit(root, registration, profile, registration['internal_seed'], 'internal', registration['max_epochs'], registration['internal_training_scenes'])
        execute(root, 'collect_actual_training', [sys.executable, 'scripts/collect_capacity_results.py', '--root', str(root)])
    selected = root / 'selection/selection.json'
    if not selected.exists() or json.loads(selected.read_text())['status'] != 'FROZEN_FROM_TRAIN_ONLY':
        state(root, 'CAPACITY_TRAIN_ONLY_SELECTING')
        execute(root, 'TRAIN_only_selection', [sys.executable, 'scripts/select_capacity_model.py',
                '--root', str(root), '--registration', str(REGISTRATION), '--device', 'cuda:0'])
    choice = json.loads(selected.read_text())
    if choice['source_provenance'] != source_stamp():
        raise RuntimeError('capacity TRAIN selection source changed')
    epoch = choice['selected_epochs']
    validation = WorldQueryClipDataset([registration['immutable_cache']], scenes=registration['validation_scenes'], clip_length=32)
    trials = [('large', seed) for seed in registration['final_refit']['large_seeds']]
    trials.append(('small', registration['final_refit']['matched_small_seed']))
    frozen = []
    for profile, seed in trials:
        state(root, 'CAPACITY_FINAL_ALL24_RUNNING', profile=profile, seed=seed, epochs=epoch)
        path = fit(root, registration, profile, seed, 'final', epoch, list(range(1, 25)))
        saved = torch.load(path, map_location='cpu', weights_only=True)
        manifest = saved['manifest']
        if not manifest['full_train_coverage'] or manifest['status'] != 'FULLTRAIN_COMPLETE':
            raise RuntimeError('capacity final fit must include all 24 TRAIN scenes')
        thresholds = choice['profiles'][profile]['thresholds']
        audit_path = path.with_suffix('.delivery.json')
        if audit_path.exists():
            delivery = json.loads(audit_path.read_text())
            if delivery['checkpoint_sha256'] != sha(path) or delivery['thresholds'] != thresholds:
                raise RuntimeError('different final-weight delivery audit must be preserved')
        else:
            delivery = {'checkpoint_sha256': sha(path), 'thresholds': thresholds,
                        'not_independent_quality': True, 'thresholds_changed': False,
                        'TRAIN_scenes_already_seen': True,
                        'audit': continuous_audit(validation, path, thresholds, 'cuda:0')}
            atomic_json(audit_path, delivery)
        if delivery['audit']['assigned_observations'] <= 0 or delivery['audit']['max_ID_distinct_timestamps'] < 2:
            raise RuntimeError('capacity final fit has no persistent TRAIN identities; retain failure')
        manifest.update(thresholds=thresholds, formal_inference_allowed=True,
                        selection_sha256=sha(selected), trained_checkpoint_sha256=sha(path))
        target = path.with_name(path.stem + '.frozen.pt')
        torch.save(saved, target)
        atomic_json(target.with_suffix('.json'), manifest)
        frozen.append({'profile': profile, 'seed': seed, 'path': str(target), 'sha256': sha(target)})
        execute(root, 'collect_actual_training', [sys.executable, 'scripts/collect_capacity_results.py', '--root', str(root)])
    lock = {'status': 'FROZEN_BEFORE_INDEPENDENT_TEST', 'checkpoint_sha256': [r['sha256'] for r in frozen],
            'checkpoints': frozen, 'source_provenance': source_stamp(), 'selection_sha256': sha(selected),
            'TEST_metadata_sha256': sha(ORIGINAL / 'audit/test_metadata.json'),
            'TEST_used_for_selection': False, 'TRAIN_scenes': list(range(1, 25)),
            'shared_final_epoch_budget': epoch, 'study_lock_sha256': sha(lock_path)}
    atomic_json(root / 'final_test_lock.json', lock)
    execute(root, 'collect_actual_training', [sys.executable, 'scripts/collect_capacity_results.py', '--root', str(root)])
    if prior_checkpoints != original_training_gate():
        raise RuntimeError('original trained checkpoints changed during isolated capacity research')
    state(root, 'CAPACITY_ALL24_FITS_COMPLETE_AWAITING_INDEPENDENT_TEST', frozen_checkpoints=frozen,
          final_epochs=epoch, original_checkpoints_unchanged=True, goal_complete=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('/data1/liuyeqiang/nas/A100/WorldMOT/capacity_20261022'))
    args = parser.parse_args()
    try:
        train(args.root)
    except Exception as error:
        args.root.mkdir(parents=True, exist_ok=True)
        state(args.root, 'FAILED_CAPACITY_STUDY_REQUIRES_FIX', error=f'{type(error).__name__}: {error}')
        raise
