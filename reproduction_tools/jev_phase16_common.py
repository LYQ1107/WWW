"""Isolated Phase XVI provenance, scientific boundaries and storage guards."""
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
for p in [ROOT, ROOT/'reproduction_tools', ROOT/'third_party/CenterNet2']:
    sys.path.insert(0,str(p))
BASE = '3e16cbf914d70bbde0d999faabc85afc9b5ae43f'
BRANCH = 'jev/www-jev-phase16-evidence-recovery-20261011'
REPORTS = ROOT/'reports/JEV_PHASE16'
OUT = Path('/home/liuyeqiang/WWW_jev_phase16_runtime/20261011_v1')
XV = Path('/home/liuyeqiang/WWW_jev_phase15_runtime/20261010_v1')
XVROOT = Path('/home/liuyeqiang/WWW_jev_phase15')
PYTHON = '/home/liuyeqiang/anaconda3/envs/GMT/bin/python'
TRAIN = [12,13,14,16]
DEV = [17,18,19]
SEEDS = [20261008,20261009,20261010]
ANNOTATIONS = Path('/data/DATASETS/TRACKING/JDE/VisionTrack/annotations/train.json')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text())

def save(path,value):
    p=Path(path)
    assert p.resolve().is_relative_to(OUT.resolve()) or p.resolve().is_relative_to(REPORTS.resolve()),p
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.tmp.'+str(os.getpid()))
    tmp.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n');tmp.replace(p)

def ref(path):return dict(path=str(path),SHA256=sha(path))

def binding(seed=20261009,checkpoints=None,inputs=None,native_state=None,evaluator=None,scope=None):
    paths=list((ROOT/'reproduction_tools').glob('jev_phase16*.py'))
    for name in ['jev_stage2','jev_phase14','jev_phase15','jev_phase16']:
        paths+=list((ROOT/'gtr/modeling'/name).glob('*.py'))
    paths += [ROOT/'gtr/modeling/meta_arch/gtr_rcnn.py',ROOT/'gtr/modeling/jev_native_state.py',
              ROOT/'reproduction_tools/jev_phase13_runtime.py']
    environment=dict(python=sys.version,platform=platform.platform(),CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'))
    if 'torch' in sys.modules:
        t=sys.modules['torch'];environment.update(torch=str(t.__version__),torch_CUDA=t.version.cuda)
        if t.cuda.is_initialized():environment.update(GPU=t.cuda.get_device_name(),cudnn=t.backends.cudnn.version())
    return dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_root=str(ROOT),source_SHA256={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(paths)) if p.is_file()},
        dataset=ref(ANNOTATIONS),checkpoints=checkpoints,native_state=native_state,inputs=inputs,
        config_SHA256={p:sha(ROOT/p) for p in ['configs/VISION_stage1.yaml','configs/VISION_test.yaml']},
        preregistration_SHA256=sha(REPORTS/'PREREGISTRATION.json') if (REPORTS/'PREREGISTRATION.json').exists() else None,
        seed=seed,environment=environment,evaluator=evaluator,scope=scope,
        TRAIN=TRAIN,development=DEV,heldout_20_21_22='SEALED',official_TEST=False,Full24=False)

def protect():
    subprocess.run(['git','merge-base','--is-ancestor',BASE,'HEAD'],cwd=ROOT,check=True)
    names=subprocess.check_output(['git','diff',BASE,'--name-only'],cwd=ROOT,text=True).splitlines()
    allowed=('reports/JEV_PHASE16/','docs/JEV_PHASE16_','gtr/modeling/jev_phase16/','reproduction_tools/jev_phase16_')
    assert all(p.startswith(allowed) for p in names),names
    f=REPORTS/'FROZEN_PRIOR_EVIDENCE.json'
    if f.exists():
        for p,digest in read(f)['prior_report_SHA256'].items():
            assert sha(p)==digest,('prior report changed',p)

def storage_guard():
    assert shutil.disk_usage(OUT.parent).free>=30*2**30,'home reserve below30GiB'
    if OUT.exists():
        total=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file() and not p.is_symlink())
        assert total<=32*2**30,'Phase XVI runtime exceeds frozen32GiB budget'

def load_policy(name):
    import torch
    if name=='original':
        from jev_phase15_common import load_old_policy
        from gtr.modeling.jev_phase15.native_commit_adapter import FrozenOriginalPolicy
        policy,checkpoint,source=load_old_policy('multi_question',20261009)
        return FrozenOriginalPolicy(policy),checkpoint,source
    assert name in ['v1','v2','v3']
    from gtr.modeling.jev_phase15.model import PersistentIdentityPolicy
    path=XV/f'training_full_payload_{name}/F_full/seed20261009/pilot/RESULT.json'
    result=read(path);assert result['status']=='COMPLETE';checkpoint=result['checkpoint']
    assert sha(checkpoint['path'])==checkpoint['SHA256']
    weights=torch.load(checkpoint['path'],map_location='cpu')
    policy=PersistentIdentityPolicy('F_full').cuda().eval();policy.load_state_dict(weights['model'],strict=True)
    policy.posterior_feedback=weights.get('posterior_feedback','native')
    return policy,checkpoint,ref(path)

def duplicate_masked_labels(video,reader):
    import collections
    from build_dense_jev_stage2_dataset import OfflineLabels
    labels=OfflineLabels(video,reader)
    for key,image in labels.images.items():
        bad={gt for gt,n in collections.Counter(a['instance_id'] for a in labels.gt[image['id']]).items() if n>1}
        if bad:labels.aligned[key]=[None if gt in bad else gt for gt in labels.current(*key)]
    return labels
