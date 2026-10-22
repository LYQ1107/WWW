#!/usr/bin/env python3
"""Finite real-TRAIN gradient/memory check, not a training-quality result."""
import argparse
import json
import time
from pathlib import Path

import torch

from cache_fulltrain import atomic_json, sha
from experiment_provenance import source_stamp
from train_world_query import train_step
from worldmot.models.query_model_profiles import build_profile
from worldmot.training.clip_dataset import WorldQueryClipDataset, collate_clips
from worldmot.training.train_utils import seed_everything
from worldmot.training.world_query_criterion import TemporalQueryCriterion


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    cache = '/data1/liuyeqiang/nas/A100/WorldMOT/fulltrain_20261022/train_cache'
    dataset = WorldQueryClipDataset([cache], scenes=[10, 24], clip_length=32)
    # Four complete clips, including the two most crowded in each scene.
    chosen = []
    for scene in (10, 24):
        candidates = [i for i, (s, start, end) in enumerate(dataset.clips) if s == scene and end-start == 32]
        ranked = sorted(candidates, reverse=True, key=lambda i: max(len(f['observations']) for f in
            dataset.scenes[scene][dataset.clips[i][1]:dataset.clips[i][2]]))
        chosen.extend(ranked[:2])
    batch = collate_clips([dataset[i] for i in chosen])
    device = torch.device(args.device)
    rows = []
    for profile in ('small', 'large'):
        seed_everything(20261022)
        model = build_profile(profile).to(device).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        row = {k: v.to(device) for k, v in batch.items()}
        metric = train_step(model, row, TemporalQueryCriterion(), 8, optimizer)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        rows.append({'profile': profile, 'trainable_parameters': sum(p.numel() for p in model.parameters()),
                     'config': model.config, 'gradient_step_metrics': metric,
                     'one_BATCH_training_step_wall_s': elapsed,
                     'GPU_peak_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
                     'all_parameters_finite': all(torch.isfinite(p).all().item() for p in model.parameters())})
        del model, optimizer, row
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    report = {'status': 'PASS_REAL_TRAIN_GRADIENT_MEMORY_PROBE', 'scene_clips': [list(dataset.clips[i]) for i in chosen],
              'batch_shape': list(batch['observations'].shape), 'device': str(device),
              'GPU': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
              'source_provenance': source_stamp(), 'TEST_read_or_used': False,
              'is_formal_training_or_quality_evaluation': False,
              'is_inference_or_E2E_FPS': False,
              'registration_sha256': sha(Path(__file__).resolve().parents[1] / 'configs/worldmot_v2_capacity_study.json'),
              'probe_script_sha256': sha(Path(__file__)),
              'profiles': rows}
    if not all(r['all_parameters_finite'] for r in rows):
        raise RuntimeError('capacity probe produced nonfinite parameters')
    atomic_json(args.output, report)
    print(json.dumps({k: v for k, v in report.items() if k not in ('source_provenance', 'profiles')}, indent=2))
    print(json.dumps({'profiles': [{k: r[k] for k in ('profile', 'trainable_parameters', 'one_BATCH_training_step_wall_s', 'GPU_peak_allocated_bytes')} for r in rows]}))
