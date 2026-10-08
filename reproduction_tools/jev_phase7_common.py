"""Isolated Phase VII paths and immutable-anchor checks (TRAIN only)."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / 'reproduction_tools', ROOT / 'third_party/CenterNet2'):
    sys.path.insert(0, str(p))
RUN_ID = os.environ.get('JEV_PHASE7_RUN_ID', '20261008_p05_v1')
OUT = Path('/home/liuyeqiang/WWW_jev_phase7_runtime') / RUN_ID
REPORTS = ROOT / 'reports/JEV_PHASE7'
BASE = '40cbc0ecc22bc16c4e3602eefe5eb06e2d6e319e'
PHASE5 = 'a37083dac0a23eb3846cb252eeb975982aaaabc9'
B2 = Path('/home/liuyeqiang/WWW_jev_phase5_runtime/20261007/minimal_training/B2/calibration/model_calibrated.pth')
B2_SHA = 'f2aa3dd2b564d90bfbfb62dc0518b0b2107a931ed7134f8c0d19f5d152b94ed7'
CACHE = Path('/data1/liuyeqiang/WWW/outputs/research_final_v2/off/perception_cache_train')
ANNOTATIONS = Path('/data/DATASETS/TRACKING/JDE/VisionTrack/annotations/train.json')
PYTHON = '/home/liuyeqiang/anaconda3/envs/GMT/bin/python'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')
    tmp.replace(path)

def protect():
    assert sha(B2) == B2_SHA, 'protected B2 changed'
    for branch, expected in [('jev/www-jev-phase5-20261007', PHASE5),
                             ('jev/www-jev-lifecycle-phase6-20261008', BASE)]:
        actual = subprocess.check_output(['git','rev-parse',branch],cwd=ROOT,text=True).strip()
        assert actual == expected, (branch,actual,expected)

def isolate_phase6_imports():
    """Redirect reusable lab globals before constructing it; never old outputs."""
    protect(); OUT.mkdir(parents=True, exist_ok=True)
    import jev_phase6_common as common
    common.OUT = OUT
    import jev_phase6_rollouts as rollouts
    rollouts.OUT = OUT
    return rollouts

def binding():
    return {'run_id':RUN_ID,'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'git_worktree_dirty':bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()),
            'code_revision_scope':'HEAD plus exact recorded source hashes; dirty HEAD is not claimed to contain every executing file',
            'base_commit':BASE,'b2_sha256':sha(B2),'cache_index_sha256':sha(CACHE/'index.jsonl'),
            'annotations_sha256':sha(ANNOTATIONS),'seed':20261003,
            'official_test_read':False,'full24_authorized':False,
            'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'reproduction_tools').glob('*phase7*.py'))}}
