"""Small independent per-GPU convolution probe; never a training checkpoint."""
from pathlib import Path
import ctypes
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cuda_responsiveness_probe'

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
    record('allocation')
    x = torch.ones(8, 3, 64, 64, device='cuda', requires_grad=True)
    layer = torch.nn.Conv2d(3, 16, 3, padding=1).cuda()
    record('convolution')
    y = layer(x)
    record('backward')
    y.sum().backward()
    torch.cuda.synchronize()
    assert torch.isfinite(x.grad).all()
    record('PASS')
else:
    OUT.mkdir(exist_ok=False)
    jobs = []
    for gpu in range(10):
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=str(gpu), CUDA_LAUNCH_BLOCKING='1', OMP_NUM_THREADS='1')
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
