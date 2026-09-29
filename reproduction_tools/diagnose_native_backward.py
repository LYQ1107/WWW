"""Isolated 100-step replay of recorded rank seeds, with native-stack capture."""
from pathlib import Path
import ctypes
import faulthandler
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / 'code/GMT'
VARIANT = os.environ.get('GMT_NATIVE_VARIANT', 'v2')
WORLD_SIZE = int(os.environ.get('GMT_WORLD_SIZE', '10'))
BATCH_SIZE = int(os.environ.get('GMT_BATCH_SIZE', '10'))
assert WORLD_SIZE > 0 and BATCH_SIZE > 0 and BATCH_SIZE % WORLD_SIZE == 0
if not re.fullmatch(r'[a-z0-9_]+', VARIANT):
    raise ValueError('Invalid diagnostic variant')
OUT = ROOT / ('outputs/diagnose_native_backward_' + VARIANT)
if os.environ.get('GMT_DIAGNOSTIC_BACKEND'):
    os.environ['GMT_DISTRIBUTED_BACKEND'] = os.environ['GMT_DIAGNOSTIC_BACKEND']
if os.environ.get('GMT_DIAGNOSTIC_CUDA_BLOCKING') is not None:
    os.environ['CUDA_LAUNCH_BLOCKING'] = os.environ['GMT_DIAGNOSTIC_CUDA_BLOCKING']
os.chdir(REPO)
sys.path[:0] = [str(REPO), str(REPO / 'third_party/CenterNet2')]
import torch
import train_net
from detectron2.engine import default_argument_parser, launch
from detectron2.utils import comm
_trace_files = []


def observed_loader_init(worker_id):
    """Preserve official worker seeding, enabling diagnosis of this owned worker."""
    parent = int(os.environ['GMT_NATIVE_PARENT'])
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(0x59616D61, parent, 0, 0, 0):
        raise OSError(ctypes.get_errno(), 'loader diagnostic ptracer registration')
    directory = Path(os.environ['GMT_NATIVE_OUTPUT']) / 'runtime'
    log = (directory / f'stacks_loader_pid{os.getpid()}.log').open('a', buffering=1)
    _trace_files.append(log)
    faulthandler.register(signal.SIGUSR1, file=log, all_threads=True)
    from detectron2.data.build import worker_init_reset_seed
    worker_init_reset_seed(worker_id)


def worker(args, seeds, parent):
    # Permit only this diagnostic parent and its descendants to collect native stacks.
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(0x59616D61, parent, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), 'PR_SET_PTRACER diagnostic parent failed')
    rank = comm.get_rank()
    os.environ['GMT_NATIVE_PARENT'] = str(parent)
    os.environ['GMT_NATIVE_OUTPUT'] = str(OUT)
    import gtr.data.gtr_dataset_dataloader as loader
    loader.worker_init_reset_seed = observed_loader_init
    if seeds is not None:
        args.opts += ['SEED', str(seeds[rank] - rank)]
    original_build = train_net.build_model
    def build(cfg):
        model = original_build(cfg)
        directory = OUT / 'runtime'
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f'last_convolution_rank{rank}.json'
        def before(name, module, inputs):
            x = inputs[0]
            path.write_text(json.dumps({'pid': os.getpid(), 'rank': rank, 'module': name,
                'shape': list(x.shape), 'stride': list(x.stride()), 'dtype': str(x.dtype),
                'weight_shape': list(module.weight.shape), 'groups': module.groups,
                'convolution_stride': module.stride, 'padding': module.padding,
                'dilation': module.dilation, 'time': time.time(), 'event': 'convolution_start'}))
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                module.register_forward_pre_hook(lambda m, inputs, name=name: before(name, m, inputs))
        return model
    train_net.build_model = build
    train_net.main(args)


def monitor(done):
    captured = set()
    while not done.wait(10):
        request_file = OUT / 'capture_request.json'
        try:
            request = json.loads(request_file.read_text()) if request_file.exists() else {}
        except json.JSONDecodeError:
            request = {}
        max_iteration = 0
        for progress_file in (OUT / 'runtime').glob('progress_rank*.jsonl'):
            try:
                max_iteration = max(max_iteration, json.loads(progress_file.read_text().splitlines()[-1])['iteration'])
            except (ValueError, IndexError):
                pass
        for rank in range(WORLD_SIZE):
            path = OUT / 'runtime' / f'progress_rank{rank}.jsonl'
            if not path.exists():
                continue
            try:
                item = json.loads(path.read_text().splitlines()[-1])
            except (ValueError, IndexError):
                continue
            forced = rank in request.get('ranks', [])
            key = (rank, item['iteration'], item['event'], str(request.get('id', 'auto')) if forced else 'auto')
            if item['event'] not in ('forward_start', 'forward_done') or (not forced and time.time() - item['time'] < 120) or key in captured:
                continue
            proc = Path('/proc') / str(item['pid'])
            if not proc.exists() or proc.stat().st_uid != os.getuid():
                continue
            if b'multiprocessing.spawn' not in (proc / 'cmdline').read_bytes():
                continue
            captured.add(key)
            log = OUT / 'runtime' / f'native_rank{rank}_iteration{item["iteration"]}_{item["event"]}_{len(captured)}.log'
            with log.open('x') as output:
                try:
                    result = subprocess.run(['gdb', '-q', '-nx', '-batch', '-ex', 'set pagination off',
                        '-ex', 'set debuginfod enabled off', '-ex', 'thread apply all bt 40',
                        '-ex', 'detach', '-p', str(item['pid'])], stdout=output,
                        stderr=subprocess.STDOUT, timeout=45)
                    print('NATIVE_STACK_CAPTURE', rank, item['iteration'], result.returncode, flush=True)
                except subprocess.TimeoutExpired:
                    print('NATIVE_STACK_CAPTURE_TIMEOUT', rank, item['iteration'], flush=True)
            if item['iteration'] < max_iteration or request.get('include_loaders', False):
                children = proc / 'task' / str(item['pid']) / 'children'
                if not children.exists():
                    continue
                for child in children.read_text().split():
                    child_proc = Path('/proc') / child
                    try:
                        if child_proc.stat().st_uid != os.getuid():
                            continue
                        status = (child_proc / 'status').read_text()
                        mask = int(next(line.split()[1] for line in status.splitlines() if line.startswith('SigCgt:')), 16)
                        if mask & (1 << (signal.SIGUSR1 - 1)):
                            os.kill(int(child), signal.SIGUSR1)
                        child_log = OUT / 'runtime' / f'native_loader_pid{child}_capture{len(captured)}.log'
                        with child_log.open('x') as output:
                            subprocess.run(['gdb', '-q', '-nx', '-batch', '-ex', 'set pagination off',
                                '-ex', 'set debuginfod enabled off', '-ex', 'thread apply all bt 40',
                                '-ex', 'detach', '-p', child], stdout=output, stderr=subprocess.STDOUT, timeout=45)
                    except (FileNotFoundError, ProcessLookupError, subprocess.TimeoutExpired):
                        print('LOADER_STACK_CAPTURE_INCOMPLETE', child, flush=True)


if __name__ == '__main__':
    if OUT.exists():
        raise RuntimeError('Refusing to overwrite prior diagnostic')
    seeds = []
    for rank in range(WORLD_SIZE):
        name = 'log.txt' + (f'.rank{rank}' if rank else '')
        source = (ROOT / 'outputs/stage1' / name).read_text()
        seeds.append(int(re.search(r'Using a generated random seed (\d+)', source).group(1)))
    (ROOT / f'manifests/native_backward_diagnostic_{VARIANT}_seeds.json').write_text(json.dumps({
        'seeds': seeds, 'source': 'failed formal Stage1 per-rank logs', 'formal_training': False,
        'runtime': {key: os.environ.get(key) for key in ['GMT_DISTRIBUTED_BACKEND', 'CUDA_LAUNCH_BLOCKING', 'CUDA_MODULE_LOADING', 'CUDA_MODULE_DATA_LOADING', 'NCCL_P2P_DISABLE']},
        'note': 'Recorded rank RNG seeds replayed; exact sample replay still needs verification. Native stack capture briefly suspends diagnostic workers only.'}, indent=2))
    args = default_argument_parser().parse_args(['--num-gpus', str(WORLD_SIZE), '--config-file',
        str(REPO / 'configs/VISION_stage1.yaml'), 'SOLVER.IMS_PER_BATCH', str(BATCH_SIZE),
        'SOLVER.TRAIN_ITER', '100', 'MODEL.WEIGHTS',
        str(ROOT / 'checkpoints/backbone/CH_FPN_1x_key_adapted.pth'), 'OUTPUT_DIR', str(OUT)])
    args.dist_url = 'tcp://127.0.0.1:{}'.format(torch.randint(12000, 59000, (1,)).item())
    done = threading.Event()
    watcher = threading.Thread(target=monitor, args=(done,), daemon=True)
    watcher.start()
    try:
        launch(worker, WORLD_SIZE, args=(args, seeds, os.getpid()), dist_url=args.dist_url)
        print('DIAGNOSTIC_100_UPDATES_COMPLETED_NOT_FORMAL', flush=True)
    finally:
        done.set()
        watcher.join(timeout=50)
