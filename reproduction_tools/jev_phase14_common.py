"""Phase XIV owns its outputs; all earlier phases and evidence are read-only."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
for p in [ROOT, ROOT / 'reproduction_tools', ROOT / 'third_party/CenterNet2']:
    sys.path.insert(0, str(p))
BASE = '2881fbdbca6501591470f9814d491a7aa8c1fb17'
OLD = Path('/home/liuyeqiang/WWW_jev_phase13_runtime/20261009_v2')
OUT = Path('/data1/liuyeqiang/WWW_jev_phase14_runtime/20261010_v1')
REPORTS = ROOT / 'reports/JEV_PHASE14'
TRAIN = [12, 13, 14, 16]
DEV = [17, 18, 19]
SEALED = [20, 21, 22]
PYTHON = '/home/liuyeqiang/anaconda3/envs/GMT/bin/python'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()

def save(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_suffix(p.suffix + '.tmp')
    t.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')
    t.replace(p)

def read(path):
    return json.loads(Path(path).read_text())

def binding():
    files = list((ROOT / 'reproduction_tools').glob('jev_phase14*'))
    files += list((ROOT / 'gtr/modeling/jev_stage2').glob('*.py'))
    files += list((ROOT / 'gtr/modeling/jev_phase14').glob('*.py'))
    return dict(source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                source_SHA256={str(p.relative_to(ROOT)): sha(p) for p in files if p.is_file()},
                frozen_evidence_SHA256=sha(REPORTS / 'PHASE13_FROZEN_EVIDENCE.json'),
                preregistration_SHA256=sha(REPORTS / 'PREREGISTRATION.json'),
                TRAIN=TRAIN, development=DEV, heldout='SEALED', Full24=False,
                official_TEST=False, CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'))

def protect():
    changed = subprocess.check_output(['git', 'diff', BASE, '--name-only', '--', 'reports', 'docs'], cwd=ROOT, text=True).splitlines()
    illegal = [p for p in changed if not (p.startswith('reports/JEV_PHASE14/') or p.startswith('docs/JEV_PHASE14_'))]
    assert not illegal, illegal
    frozen = read(REPORTS / 'PHASE13_FROZEN_EVIDENCE.json')
    for rel, checksum in frozen['protected_report_SHA256'].items():
        assert sha(ROOT / rel) == checksum, rel
    return True
