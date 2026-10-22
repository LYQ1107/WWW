#!/usr/bin/env python3
"""Run four frozen capacity trials after original comparisons release the cards."""
import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from cache_fulltrain import atomic_json, sha
from experiment_provenance import source_stamp


REPO = Path(__file__).resolve().parents[1]
ORIGINAL = Path('/data1/liuyeqiang/nas/A100/WorldMOT/fulltrain_20261022')
PYTHON = '/data1/liuyeqiang/envs/worldmot311/bin/python'
VISION_PYTHON = '/data1/liuyeqiang/envs/worldmot_mmdet311/bin/python'


def load(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def ready(root):
    return (load(ORIGINAL / 'experiment_status.json').get('stage') == 'RESULTS_READY_FOR_REVIEW_AND_GITHUB_PUSH'
            and load(root / 'experiment_status.json').get('stage') == 'CAPACITY_ALL24_FITS_COMPLETE_AWAITING_INDEPENDENT_TEST')


def execute(root, name, command, gpu=None):
    environment = {**os.environ, 'PYTHONPATH': str(REPO), 'OMP_NUM_THREADS': '2',
                   'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '2'}
    if gpu is not None:
        environment['CUDA_VISIBLE_DEVICES'] = str(gpu)
    with (root / 'logs' / (name + '.log')).open('a') as stream:
        stream.write('\nCOMMAND ' + json.dumps(command) + '\n')
        stream.flush()
        completed = subprocess.run(command, cwd=REPO, env=environment, stdout=stream, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise RuntimeError('capacity ' + name + ' failed; see its log')


def run(root, gpus):
    previous = load(root / 'test_status.json')
    if previous.get('status') == 'INDEPENDENT_TEST_RUNNING' and previous.get('pid') != os.getpid():
        proc = Path('/proc') / str(previous['pid'])
        if proc.exists() and (proc / 'cwd').resolve() == REPO and b'scripts/run_capacity_test.py' in (proc / 'cmdline').read_bytes().split(b'\0'):
            raise RuntimeError('the registered capacity TEST runner is already active')
    if len(gpus) != 4 or len(set(gpus)) != 4 or set(gpus) != {0, 2, 3, 4}:
        raise RuntimeError('capacity TEST uses the four assigned physical cards 0/2/3/4')
    lock_path = root / 'final_test_lock.json'
    lock = load(lock_path)
    if lock.get('source_provenance') != source_stamp() or len(lock.get('checkpoints', [])) != 4:
        raise RuntimeError('all four full-TRAIN capacity weights must be frozen with matching source')
    if lock.get('TEST_metadata_sha256') != sha(ORIGINAL / 'audit/test_metadata.json'):
        raise RuntimeError('capacity TEST metadata changed')
    for row in lock['checkpoints']:
        if sha(row['path']) != row['sha256']:
            raise RuntimeError('capacity weight changed after freezing')
    protocol_path = root / 'test_protocol_lock.json'
    protocol = {'final_lock_sha256': sha(lock_path), 'source_provenance': source_stamp(),
                'runner_sha256': sha(Path(__file__)),
                'collector_sha256': sha(REPO / 'scripts/collect_capacity_test_results.py'),
                'original_results_finished_before_capacity_TEST': True, 'physical_GPUs': gpus,
                'one_trial_per_GPU': True, 'TEST_used_for_selection': False}
    if protocol_path.exists() and load(protocol_path) != protocol:
        raise RuntimeError('preserve changed capacity TEST protocol before resuming')
    atomic_json(protocol_path, protocol)
    audit = root / 'audit'
    audit.mkdir(exist_ok=True)
    for name in ('data_audit.json', 'test_metadata.json'):
        shutil.copyfile(ORIGINAL / 'audit' / name, audit / name)
    def inference(row, gpu):
        label = f"capacity_{row['profile']}_seed{row['seed']}"
        execute(root, 'TEST_' + label, [VISION_PYTHON, 'scripts/run_fullvideo.py', '--split', 'test',
            '--metadata', str(audit / 'test_metadata.json'), '--method', 'v2', '--checkpoint', row['path'],
            '--lock', str(lock_path), '--output', str(root / 'test' / label), '--device', 'cuda:0'], gpu)
        return label
    atomic_json(root / 'test_status.json', {'status': 'INDEPENDENT_TEST_RUNNING', 'pid': os.getpid(),
                                           'protocol_sha256': sha(protocol_path)})
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(inference, row, gpu) for row, gpu in zip(lock['checkpoints'], gpus)]
        labels = [future.result() for future in futures]
    for label in labels:
        execute(root, 'EXPORT_' + label, [PYTHON, 'scripts/export_fullvideo_predictions.py',
                '--metadata', str(audit / 'test_metadata.json'), '--predictions', str(root / 'test' / label),
                '--output', str(root / 'test_submissions' / label)])
        execute(root, 'EVAL_' + label, [PYTHON, 'scripts/evaluate_fullvideo.py', '--split', 'test',
                '--audit', str(audit / 'data_audit.json'), '--predictions', str(root / 'test' / label),
                '--output', str(root / 'test_evaluation' / label)])
    execute(root, 'capacity_failure_analysis', [PYTHON, 'scripts/analyze_fulltrain_test_failures.py',
            '--root', str(root), '--output', str(REPO / 'results/worldmot_v2_capacity_failure_analysis.json'), '--render'])
    execute(root, 'collect_capacity_TEST', [PYTHON, 'scripts/collect_capacity_test_results.py', '--root', str(root)])
    atomic_json(root / 'test_status.json', {'status': 'COMPLETE_REQUIRES_REVIEW_AND_GITHUB_PUSH',
                                           'protocol_sha256': sha(protocol_path), 'goal_complete': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('/data1/liuyeqiang/nas/A100/WorldMOT/capacity_20261022'))
    parser.add_argument('--gpus', nargs='+', type=int, default=[0, 2, 3, 4])
    parser.add_argument('--wait-for-original', action='store_true')
    args = parser.parse_args()
    (args.root / 'logs').mkdir(parents=True, exist_ok=True)
    if args.wait_for_original:
        atomic_json(args.root / 'test_waiter.json', {'pid': os.getpid(), 'status': 'WAITING_FOR_ORIGINAL_TEST_AND_CAPACITY_FITS',
                                                   'original_metrics_used_for_selection': False})
        while not ready(args.root):
            time.sleep(30)
    elif not ready(args.root):
        raise RuntimeError('original comparisons and capacity final fits must complete first')
    try:
        run(args.root, args.gpus)
    except Exception as error:
        atomic_json(args.root / 'test_status.json', {'status': 'FAILED_REQUIRES_AGENT_FIX', 'error': str(error)})
        raise
