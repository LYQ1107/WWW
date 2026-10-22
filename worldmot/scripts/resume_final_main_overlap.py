#!/usr/bin/env python3
"""Validate final main fits, audit TRAIN ID delivery, then resume their controller.

This does not select thresholds or read TEST. The controller waits only to
prevent two writers from fitting the same checkpoint during overlapped work.
"""
import argparse
import json
import os
import signal
import time
from pathlib import Path

from cache_fulltrain import atomic_json, sha
from experiment_provenance import source_stamp


def read(path):
    for attempt in range(3):
        try:
            return json.loads(Path(path).read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            if attempt < 2:
                time.sleep(.1)
    return None


def inspect(root):
    overlap = read(root / 'final_main_overlap.json')
    lock = read(root / 'final_main_configuration_lock.json')
    choice = read(root / 'selection/selection.json')
    if not overlap or not lock or not choice:
        raise RuntimeError('missing registered overlap/selection provenance')
    if choice['selected_epochs'] != lock['selected_epochs'] or choice['methods']['joint']['thresholds'] != lock['thresholds']:
        raise RuntimeError('TRAIN joint decision differs from the configuration frozen before final fitting')
    trials = []
    for job in overlap['main_trials']:
        manifest = read(Path(job['checkpoint']).with_suffix('.json'))
        ready = bool(manifest and manifest.get('status') == 'FULLTRAIN_COMPLETE'
                     and manifest.get('epochs_completed') == lock['selected_epochs']
                     and manifest.get('full_train_coverage')
                     and manifest.get('train_scenes') == list(range(1, 25))
                     and manifest.get('total_timestamps') == 28754)
        trials.append({'seed': job['seed'], 'checkpoint': job['checkpoint'],
                       'epochs_completed': manifest.get('epochs_completed', 0) if manifest else 0,
                       'ready': ready})
    return overlap, lock, trials


def audit_and_resume(root, device):
    overlap, lock, trials = inspect(root)
    if not all(t['ready'] for t in trials):
        return {'status': 'WAITING_FOR_FINAL_MAIN', 'trials': trials}
    if not overlap.get('controller_intentionally_waiting'):
        return {'status': 'CONTROLLER_ALREADY_RELEASED', 'trials': trials}
    import torch
    from select_query_model import continuous_audit, VAL_SCENES
    from worldmot.training.clip_dataset import WorldQueryClipDataset

    torch.set_num_threads(2)
    provenance = source_stamp()
    if provenance != lock['source_provenance']:
        raise RuntimeError('model/training source differs from frozen main choice')
    checkpoint_hashes = {}
    for trial in trials:
        path = Path(trial['checkpoint'])
        saved = torch.load(path, map_location='cpu', weights_only=True)
        m = saved['manifest']
        if (saved['epoch'] != lock['selected_epochs'] or m['source_provenance'] != provenance
                or m['seed'] != trial['seed'] or m['resume_checkpoint'] is not None
                or m['model_config'] != lock['model_config']
                or len(m['history']) != lock['selected_epochs']
                or any(sum(h['coverage'].values()) != 28754 for h in m['history'])):
            raise RuntimeError('final checkpoint does not match fresh full-TRAIN registration')
        checkpoint_hashes[str(trial['seed'])] = sha(path)

    report_path = root / 'final_main_train_delivery_audit.json'
    previous = read(report_path)
    results = {}
    if previous:
        if previous['checkpoint_sha256'] != checkpoint_hashes or previous['thresholds'] != lock['thresholds']:
            raise RuntimeError('preserve the different-model TRAIN delivery audit before rerunning')
        results = previous.get('seeds', {})
    report = {'status': 'RUNNING_TRAIN_DELIVERY_AUDIT', 'checkpoint_sha256': checkpoint_hashes,
              'thresholds': lock['thresholds'], 'thresholds_changed': False,
              'TRAIN_scenes_already_seen_in_final_fitting': True,
              'cached_perception_replay': True, 'is_independent_quality_score': False,
              'formal_inference_FPS_measured': False, 'GT_used_for_assignment': False,
              'device': device, 'seeds': results}
    dataset = WorldQueryClipDataset([str(root / 'train_cache')], scenes=VAL_SCENES, clip_length=32)
    if sorted(dataset.scenes) != sorted(VAL_SCENES) or sum(len(f) for f in dataset.scenes.values()) != 5860:
        raise RuntimeError('delivery audit must cover all six complete TRAIN timelines')
    for trial in trials:
        seed = str(trial['seed'])
        if seed not in results:
            results[seed] = continuous_audit(dataset, trial['checkpoint'], lock['thresholds'], device)
            atomic_json(report_path, report)
        audit = results[seed]
        if (audit['timestamps'] != 5860 or audit['assigned_observations'] <= 0
                or audit['max_ID_distinct_timestamps'] < 2):
            report['status'] = 'FAILED_TRAIN_DELIVERY_GATE'
            atomic_json(report_path, report)
            raise RuntimeError('final seed has no persistent TRAIN identities; retain failure before TEST')
    if any(sha(t['checkpoint']) != checkpoint_hashes[str(t['seed'])] for t in trials):
        raise RuntimeError('checkpoint changed during delivery audit')
    report['status'] = 'ALL_THREE_FINAL_SEEDS_PASS_TRAIN_DELIVERY_GATE'
    atomic_json(report_path, report)

    # The PID, command and cwd are all checked before sending a signal.
    pid = overlap['controller_pid']
    proc = Path('/proc') / str(pid)
    if (b'scripts/orchestrate_fulltrain_test.py' not in (proc / 'cmdline').read_bytes().split(b'\0')
            or (proc / 'cwd').resolve() != Path(__file__).resolve().parents[1]
            or (proc / 'stat').read_text().split()[2] not in ('T', 't')):
        raise RuntimeError('registered controller is missing, replaced, or no longer waiting')
    current = read(root / 'selection/selection.json')
    if current['selected_epochs'] != lock['selected_epochs'] or current['methods']['joint']['thresholds'] != lock['thresholds']:
        raise RuntimeError('joint configuration changed during audit')
    stamp = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    state = {'stage': 'FINAL_MAIN_COMPLETE_AWAITING_CONTROL_SELECTION', 'updated_utc': stamp,
             'main_checkpoint_sha256': checkpoint_hashes, 'TRAIN_delivery_audit_sha256': sha(report_path)}
    atomic_json(root / 'experiment_status.json', state)
    with (root / 'events.jsonl').open('a') as stream:
        stream.write(json.dumps(state) + '\n')
    os.kill(pid, signal.SIGCONT)
    overlap.update(status='MAIN_FITS_AND_TRAIN_DELIVERY_AUDITS_COMPLETE_CONTROLLER_RESUMED',
                   controller_intentionally_waiting=False, resumed_utc=stamp,
                   checkpoint_sha256=checkpoint_hashes, TRAIN_delivery_audit_sha256=sha(report_path))
    atomic_json(root / 'final_main_overlap.json', overlap)
    return {'status': overlap['status'], 'controller_pid': pid, 'audit': str(report_path)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='/data1/liuyeqiang/nas/A100/WorldMOT/fulltrain_20261022')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--audit-and-resume', action='store_true')
    args = parser.parse_args()
    root = Path(args.root)
    if args.audit_and_resume:
        result = audit_and_resume(root, args.device)
    else:
        _, _, trials = inspect(root)
        result = {'status': 'READY_FOR_AUDIT' if all(t['ready'] for t in trials) else 'WAITING_FOR_FINAL_MAIN',
                  'signal_sent': False, 'trials': trials}
    print(json.dumps(result, indent=2), flush=True)
