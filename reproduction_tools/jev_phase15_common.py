"""Phase XV provenance and write boundaries; all prior science is read-only."""
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
for p in [ROOT, ROOT / 'reproduction_tools', ROOT / 'third_party/CenterNet2']:
    sys.path.insert(0, str(p))
BASE = '0dc9607192c22bb4487b77b41de20f188caf7f1a'
BRANCH = 'jev/www-jev-phase15-persistent-identity-commitment-20261010'
REPORTS = ROOT / 'reports/JEV_PHASE15'
OUT = Path('/home/liuyeqiang/WWW_jev_phase15_runtime/20261010_v1')
PHASE14 = Path('/data1/liuyeqiang/WWW_jev_phase14_runtime/20261010_v1')
PHASE13 = Path('/home/liuyeqiang/WWW_jev_phase13_runtime/20261009_v2')
PYTHON = '/home/liuyeqiang/anaconda3/envs/GMT/bin/python'
TRAIN = [12, 13, 14, 16]
DEV = [17, 18, 19]
SEEDS = [20261008, 20261009, 20261010]
ANNOTATIONS = Path('/data/DATASETS/TRACKING/JDE/VisionTrack/annotations/train.json')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    p = Path(path)
    assert p.resolve().is_relative_to(OUT.resolve()) or p.resolve().is_relative_to(REPORTS.resolve()), p
    p.parent.mkdir(parents=True, exist_ok=True)
    temporary = p.with_name(p.name + '.tmp.' + str(os.getpid()))
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(p)


def ref(path):
    p = Path(path)
    return {'path': str(p), 'SHA256': sha(p)}


def binding(seed=None, checkpoints=None, dataset=None, evaluator=None, scope=None):
    paths = list((ROOT / 'reproduction_tools').glob('jev_phase15*'))
    for folder in ['jev_stage2', 'jev_phase14', 'jev_phase15']:
        paths += list((ROOT / 'gtr/modeling' / folder).glob('*.py'))
    paths += [ROOT / 'gtr/modeling/meta_arch/gtr_rcnn.py', ROOT / 'gtr/modeling/jev_native_state.py',
              ROOT / 'reproduction_tools/jev_phase13_runtime.py']
    return {'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'source_root': str(ROOT), 'source_SHA256': {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(paths)) if p.is_file()},
            'config_SHA256': {p: sha(ROOT / p) for p in ['configs/VISION_stage1.yaml', 'configs/VISION_test.yaml']},
            'seed': seed, 'checkpoints': checkpoints, 'dataset': dataset, 'evaluator': evaluator, 'scope': scope,
            'environment': {'python': sys.version, 'platform': platform.platform(),
                            'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES')},
            'TRAIN': TRAIN, 'development': DEV, 'heldout_20_21_22': 'SEALED',
            'official_TEST': False, 'Full24': False,
            'preregistration_SHA256': sha(REPORTS / 'PREREGISTRATION.json') if (REPORTS / 'PREREGISTRATION.json').exists() else None}


def protect():
    subprocess.run(['git', 'merge-base', '--is-ancestor', BASE, 'HEAD'], cwd=ROOT, check=True)
    names = subprocess.check_output(['git', 'diff', BASE, '--name-only'], cwd=ROOT, text=True).splitlines()
    allowed = ('reports/JEV_PHASE15/', 'docs/JEV_PHASE15_', 'gtr/modeling/jev_phase15/', 'reproduction_tools/jev_phase15_')
    assert all(p.startswith(allowed) for p in names), names
    if (REPORTS / 'PHASE14_FROZEN_EVIDENCE.json').exists():
        for path, digest in read(REPORTS / 'PHASE14_FROZEN_EVIDENCE.json')['protected_report_SHA256'].items():
            assert sha(ROOT / path) == digest, ('prior evidence changed', path)


def storage_guard():
    free = shutil.disk_usage(OUT.parent if OUT.parent.exists() else ROOT.parent).free
    assert free >= 30 * 2**30, 'Phase XV home reserve below frozen30GiB; preserve all science and stop producer'


def old_case(variant, seed, video):
    assert video in TRAIN + DEV
    return PHASE14 / 'formal_online_v2' / f'{variant}_seed{seed}' / f'video{video:02d}'


def load_old_policy(variant, seed):
    import torch
    from gtr.modeling.jev_phase14.model import ReliableIdentityPolicy
    p = PHASE14 / 'formal_v2' / variant / 'availability_joint' / f'seed{seed}' / 'RESULT.json'
    result = read(p)
    assert result['status'] == 'COMPLETE' and result['actual_updates'] == 20000
    ck = result['checkpoint']
    assert sha(ck['path']) == ck['SHA256']
    policy = ReliableIdentityPolicy(variant, result['loss']).cuda().eval()
    policy.load_state_dict(torch.load(ck['path'], map_location='cpu')['model'], strict=True)
    return policy, ck, ref(p)


def failure(stage, error):
    save(OUT / 'failures' / f'{stage}_{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")}_{os.getpid()}.json',
         {'status': 'FAIL', 'binding': binding(), 'error': str(error)})
