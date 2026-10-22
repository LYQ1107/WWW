#!/usr/bin/env python3
"""Reload only the stopped coordinator after all registered fits finish."""
import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

import torch

from cache_fulltrain import atomic_json, sha
from experiment_provenance import source_stamp


ROOT = Path('/data1/liuyeqiang/nas/A100/WorldMOT/fulltrain_20261022')
CWD = Path(__file__).resolve().parents[1]
PYTHON = '/data1/liuyeqiang/envs/worldmot311/bin/python'
TRIALS = [('joint', seed) for seed in (20261022, 20261023, 20261024)] + [
    (method, 20261022) for method in ('no_omega', 'no_appearance', 'two_stage', 'no_geometry')]


def ready():
    for method, seed in TRIALS:
        path = ROOT / 'final' / f'{method}_seed{seed}.json'
        try:
            manifest = json.loads(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return False
        if manifest.get('status') != 'FULLTRAIN_COMPLETE' or manifest.get('epochs_completed') != 4:
            return False
    hold = json.loads((ROOT / 'capacity_queue_boundary_hold.json').read_text())
    for pid in hold['live_control_fits'].values():
        path = Path('/proc') / str(pid) / 'stat'
        try:
            if path.read_text().split()[2] != 'Z':
                return False
        except FileNotFoundError:
            pass
    return True


def reload():
    if not ready():
        raise RuntimeError('checkpoint/worker boundary is not ready')
    source = source_stamp()
    hashes = {}
    for method, seed in TRIALS:
        path = ROOT / 'final' / f'{method}_seed{seed}.pt'
        saved = torch.load(path, map_location='cpu', weights_only=True)
        m = saved['manifest']
        if (saved['epoch'] != 4 or m['source_provenance'] != source or not m['full_train_coverage']
                or m['train_scenes'] != list(range(1, 25)) or m['total_timestamps'] != 28754
                or len(m['history']) != 4 or any(sum(r['coverage'].values()) != 28754 for r in m['history'])
                or not saved['optimizer']['state'] or not saved['rng_state']['torch'].numel()):
            raise RuntimeError('incomplete or different-source final checkpoint: ' + str(path))
        hashes[path.stem] = sha(path)
    choice = json.loads((ROOT / 'selection/selection.json').read_text())
    if choice['status'] != 'FROZEN_FROM_TRAIN_VALIDATION' or choice['selected_epochs'] != 4:
        raise RuntimeError('original TRAIN choice differs from registered budget')
    registered = json.loads((ROOT / 'additional_capacity_job.json').read_text())
    if registered['status'] != 'READY_AND_REGISTERED_BEFORE_TEST':
        raise RuntimeError('capacity job is not ready')
    cwd = Path(registered['cwd'])
    if any(sha(cwd / name) != expected for name, expected in registered['execution_sha256'].items()):
        raise RuntimeError('capacity code differs from its frozen queue registration')
    hold = json.loads((ROOT / 'capacity_queue_boundary_hold.json').read_text())
    old_pid = hold['controller_pid']
    proc = Path('/proc') / str(old_pid)
    if (b'scripts/orchestrate_fulltrain_test.py' not in (proc / 'cmdline').read_bytes().split(b'\0')
            or (proc / 'cwd').resolve() != CWD or (proc / 'stat').read_text().split()[2] not in ('T', 't')):
        raise RuntimeError('old coordinator is missing, replaced, or no longer stopped')
    # All its training children have exited. Never signal their process group.
    os.kill(old_pid, signal.SIGKILL)
    environment = {**os.environ, 'PYTHONPATH': str(CWD), 'OMP_NUM_THREADS': '2',
                   'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '2'}
    with (ROOT / 'logs/orchestrator.log').open('a') as stream:
        stream.write('\nRELOAD_AT_COMPLETE_FINAL_CHECKPOINT_BOUNDARY\n')
        stream.flush()
        child = subprocess.Popen([PYTHON, '-u', 'scripts/orchestrate_fulltrain_test.py'],
                                 cwd=CWD, env=environment, stdout=stream, stderr=subprocess.STDOUT,
                                 start_new_session=True)
    report = {'status': 'APPLIED_AT_COMPLETE_FINAL_CHECKPOINT_BOUNDARY',
              'updated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'old_controller_pid': old_pid, 'new_controller_pid': child.pid,
              'driver_source_sha256': sha(CWD / 'scripts/orchestrate_fulltrain_test.py'),
              'all_seven_final_checkpoint_sha256': hashes,
              'model_source_unchanged': source['sha256'],
              'selected_epochs': 4, 'TEST_started_before_reload': False,
              'original_training_children_exited': True, 'independent_capacity_job_registered': True}
    atomic_json(ROOT / 'driver_capacity_upgrade_pending.json', report)
    atomic_json(ROOT / 'orchestrator_process.json', {'pid': child.pid, **report})
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reload-when-ready', action='store_true')
    args = parser.parse_args()
    if args.reload_when_ready:
        deadline = time.monotonic() + 1800
        while not ready():
            if time.monotonic() >= deadline:
                raise RuntimeError('final fits did not reach a complete boundary; no coordinator signal sent')
            time.sleep(10)
        reload()
    else:
        print(json.dumps({'ready': ready(), 'signal_sent': False}))
