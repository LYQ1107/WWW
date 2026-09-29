"""Independent per-GPU host/device transfer probe; never a training checkpoint."""
from pathlib import Path
import ctypes
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cuda_transfer_probe_v6'

if len(sys.argv) > 1:
    gpu = int(sys.argv[1])
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(0x59616D61, os.getppid(), 0, 0, 0):
        raise OSError(ctypes.get_errno(), 'diagnostic ptracer registration')
    def record(phase):
        (OUT / f'gpu{gpu}.json').write_text(json.dumps({'gpu': gpu, 'pid': os.getpid(), 'phase': phase, 'time': time.time()}))
    record('import')
    import torch
    torch.set_num_threads(1)
    record('cuda_initialization')
    torch.cuda.set_device(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = True
    record('host_allocation')
    host = torch.arange(1024 * 1024, dtype=torch.float32)
    for step in range(10):
        record(f'host_to_device_{step}')
        device = host.to('cuda')
        record(f'device_compute_{step}')
        device.add_(1)
        record(f'device_to_host_{step}')
        output = device.cpu()
        assert torch.equal(output, host + 1)
    torch.cuda.synchronize()
    record('PASS')
else:
    OUT.mkdir(exist_ok=False)
    jobs = []
    for gpu in [0, 5]:
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=str(gpu), CUDA_LAUNCH_BLOCKING='0', CUDA_MODULE_LOADING='EAGER', CUDA_MODULE_DATA_LOADING='EAGER', OMP_NUM_THREADS='1')
        log = (OUT / f'gpu{gpu}.log').open('x')
        p = subprocess.Popen([sys.executable, __file__, str(gpu)], env=env, stdout=log, stderr=subprocess.STDOUT)
        jobs.append((gpu, p, log))
    (OUT / 'jobs.json').write_text(json.dumps({'pid': os.getpid(), 'workers': [{'gpu': g, 'pid': p.pid} for g, p, _ in jobs]}, indent=2))
    captured = set()
    started = time.time()
    while any(p.poll() is None for _, p, _ in jobs):
        if time.time() - started > 30:
            for gpu, p, _ in jobs:
                if gpu in captured or p.poll() is not None:
                    continue
                captured.add(gpu)
                with (OUT / f'native_gpu{gpu}.log').open('x') as log:
                    try:
                        subprocess.run(['gdb', '-q', '-nx', '-batch', '-ex', 'set pagination off',
                            '-ex', 'set debuginfod enabled off', '-ex', 'thread apply all bt 15',
                            '-ex', 'detach', '-p', str(p.pid)], stdout=log, stderr=subprocess.STDOUT, timeout=45)
                    except subprocess.TimeoutExpired:
                        print('native capture timed out', gpu, flush=True)
        time.sleep(2)
    result = [{'gpu': g, 'pid': p.pid, 'exit_code': p.returncode} for g, p, _ in jobs]
    (OUT / 'result.json').write_text(json.dumps(result, indent=2))
    print(result, flush=True)
