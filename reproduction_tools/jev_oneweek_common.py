"""Independent bounded MATCH-only sprint; frozen historical assets are read-only."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT/'reproduction_tools', ROOT/'third_party/CenterNet2'):
    sys.path.insert(0, str(directory))
REPORTS = ROOT/'reports/JEV_ONEWEEK'
DOCS = ROOT/'docs/JEV_ONEWEEK'
OUT = Path('/data1/liuyeqiang/WWW_jev_oneweek_runtime/20261012_v1')
BASE = 'd49c90e58dee9919a955d56947908928fc95f18b'
PYTHON = '/home/liuyeqiang/anaconda3/envs/GMT/bin/python'
TRAIN = [12, 13, 14, 16]
DEV = [17, 18, 19]
SEEDS = [20261008, 20261009, 20261010]
PHASE13 = Path('/home/liuyeqiang/WWW_jev_phase13_runtime/20261009_v2')
PHASE14 = Path('/data1/liuyeqiang/WWW_jev_phase14_runtime/20261010_v1')
PHASE15 = Path('/home/liuyeqiang/WWW_jev_phase15_runtime/20261010_v1')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4*2**20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    assert any(path.resolve().is_relative_to(r.resolve()) for r in (REPORTS, OUT)), path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp.'+str(os.getpid()))
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')
    temporary.replace(path)


def ref(path):
    path = Path(path)
    return dict(path=str(path), SHA256=sha(path), bytes=path.stat().st_size)


def protect():
    assert subprocess.check_output(['git', 'symbolic-ref', '--short', 'HEAD'], cwd=ROOT, text=True).strip() == 'jev/www-jev-oneweek-20261012'
    subprocess.run(['git', 'merge-base', '--is-ancestor', BASE, 'HEAD'], cwd=ROOT, check=True)
    changes = subprocess.check_output(['git', 'diff', BASE, '--name-only'], cwd=ROOT, text=True).splitlines()
    allowed = ('docs/JEV_ONEWEEK/', 'reports/JEV_ONEWEEK/', 'reproduction_tools/jev_oneweek_',
               'gtr/modeling/jev_oneweek/', 'configs/JEV_ONEWEEK/')
    assert all(name.startswith(allowed) for name in changes), changes
    for partition in ('/home/liuyeqiang', '/data1/liuyeqiang'):
        assert shutil.disk_usage(partition).free >= 30*2**30, 'Storage reserve below 30 GiB'
    assert runtime_bytes() <= 8*2**30, 'Sprint runtime retention budget exceeded'


def runtime_bytes():
    total=0
    for path in OUT.rglob('*'):
        try:
            if path.is_file():total+=path.stat().st_size
        except FileNotFoundError:
            pass  # Another task atomically replaced a small progress temporary.
    return total


def binding(scope):
    sources = list((ROOT/'reproduction_tools').glob('jev_oneweek_*.py'))
    sources += list((ROOT/'gtr/modeling/jev_oneweek').glob('*.py'))
    sources += [ROOT/'gtr/modeling/meta_arch/gtr_rcnn.py', ROOT/'gtr/modeling/jev_stage2/native.py',
                ROOT/'gtr/modeling/jev_stage2/assignment.py', ROOT/'gtr/modeling/jev_phase14/model.py',
                ROOT/'gtr/modeling/jev_phase15/native_commit_adapter.py']
    return dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                sources={str(p.relative_to(ROOT)): sha(p) for p in sources}, scope=scope,
                CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),
                protocol=ref(REPORTS/'PREREGISTRATION.json'), official_TEST=False,
                sealed20_21_22=True, Full24=False)


def seed_all(seed=20261009):
    import random
    import numpy as np
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def failure(stage, error):
    save(OUT/'failures'/f'{stage}_{time.time_ns()}.json',
         dict(status='FAIL', stage=stage, error=str(error), binding=binding(stage)))
