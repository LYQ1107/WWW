#!/usr/bin/env python3
"""Describe saved TRAIN evidence without selecting a model or changing a budget."""
import argparse
import collections
import datetime
import json
from pathlib import Path

from cache_fulltrain import atomic_json, sha
from experiment_provenance import source_stamp
from worldmot.models.persistent_world_queries import PersistentWorldQueryTransformer


def audit(root, output):
    selection_path = root / 'selection/selection.json'
    selection = json.loads(selection_path.read_text())
    internal_path = root / 'internal/joint.json'
    internal = json.loads(internal_path.read_text())
    model = PersistentWorldQueryTransformer(**internal['model_config'])
    components = collections.Counter()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            components[name.split('.')[0]] += parameter.numel()
    trials = []
    for seed in (20261022, 20261023, 20261024):
        path = root / 'final' / f'joint_seed{seed}.json'
        if not path.exists():
            continue
        manifest = json.loads(path.read_text())
        trials.append({'seed': seed, 'status': manifest['status'],
                       'epochs_completed': manifest['epochs_completed'],
                       'timestamps_per_epoch': manifest['total_timestamps'],
                       'train_scenes': manifest['train_scenes'],
                       'history': manifest['history'],
                       'manifest_sha256': sha(path),
                       'checkpoint_sha256': sha(path.with_suffix('.pt'))})
    verification = json.loads((root / 'cache_verification/coverage.json').read_text())
    result = {
        'status': 'CONVERGENCE_NOT_ESTABLISHED',
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'scope': 'read-only audit of already registered TRAIN runs; no new fit or model selection',
        'TEST_read_or_used': False,
        'selected_epochs_unchanged': selection['selected_epochs'],
        'trainable_parameters': sum(components.values()),
        'trainable_parameters_by_component': dict(sorted(components.items())),
        'frozen_perception': ['official RTMDet-Ins-tiny', 'official VGGT-Omega 512 weight variant'],
        'model_config': internal['model_config'],
        'learning_rate': internal['learning_rate'],
        'optimizer': internal['optimizer'],
        'gradient_clip_norm': internal['gradient_clip_norm'],
        'logged_gradient_norm_is_pre_clip': True,
        'clip_curriculum': internal['clip_curriculum'],
        'tbptt': internal['tbptt'],
        'internal_TRAIN_scenes': internal['train_scenes'],
        'held_out_TRAIN_scenes': selection['validation_scenes'],
        'internal_training_history': internal['history'],
        'held_out_TRAIN_epoch_loss': selection['epoch_validation'],
        'main_final_trials': trials,
        'TRAIN_input_audit_sha256': sha(root / 'cache_verification/coverage.json'),
        'verified_TRAIN_timestamps': verification['timestamps'],
        'source_provenance': source_stamp(),
        'interpretation': [
            'Four epochs are the fixed budget selected from eligible budgets 4 and 6 by held-out TRAIN loss; epoch 2 is diagnostic only.',
            'Clip length changes from 8 to 16 to 32. Loss differences across those changes are not a fixed-objective convergence curve.',
            'The saved curves and sparse validation checkpoints do not establish a stable plateau or optimal training budget.',
            'Pre-clipping gradient spikes and high pair loss motivate optimization and geometry-conditioning audits; they do not prove a geometry-caused failure.',
            'Query slot sufficiency and network expressive capacity are different quantities. No registered width comparison has identified a capacity bottleneck.',
            'Final refits see all 24 TRAIN scenes. Their TRAIN replay cannot serve as independent validation accuracy.',
            'Any longer-budget or wider-model study must retain these runs, register its own TRAIN-only comparison, and keep TEST out of selection.',
        ],
        'capacity_cause_identified': False,
        'longer_training_guaranteed_to_help': False,
    }
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / 'worldmot_v2_convergence_audit.json', result)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for trial in trials:
        history = trial['history']
        axes[0].plot([h['epoch'] for h in history], [h['total'] for h in history],
                     marker='o', label=str(trial['seed']))
        axes[2].plot([h['epoch'] for h in history], [h['gradient_norm'] for h in history], marker='o')
    for axis in (axes[0], axes[2]):
        axis.axvline(2.5, color='grey', linestyle=':', alpha=.6)
        axis.set_xlabel('Epoch (clip lengths: 8, 16, 32, 32)')
        axis.set_yscale('log')
    axes[0].set_title('All-24 TRAIN refit loss (log scale)')
    axes[0].set_ylabel('Total training loss')
    axes[0].legend(fontsize=8)
    validation = selection['epoch_validation']
    axes[1].plot([r['epoch'] for r in validation], [r['total'] for r in validation], marker='o')
    axes[1].set_title('Held-out TRAIN loss: internal seed')
    axes[1].set_xlabel('Saved epoch (2 is diagnostic only)')
    axes[1].set_ylabel('Validation loss; fixed clip length 32')
    axes[2].set_title('Mean gradient norm BEFORE clipping')
    axes[2].set_ylabel('Norm (clipped to 1 before optimizer step)')
    figure.suptitle('WorldMOT v2: current evidence does not establish convergence', fontsize=11)
    figure.tight_layout()
    figure.savefig(output / 'worldmot_v2_convergence_audit.png', dpi=160)
    plt.close(figure)
    return {k: result[k] for k in ('status', 'trainable_parameters', 'selected_epochs_unchanged', 'capacity_cause_identified')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('/data1/liuyeqiang/nas/A100/WorldMOT/fulltrain_20261022'))
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1] / 'results')
    args = parser.parse_args()
    print(json.dumps(audit(args.root, args.output), indent=2))
