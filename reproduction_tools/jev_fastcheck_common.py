"""Isolated bounded MATCH fastcheck; inherited research artifacts are read-only."""
import hashlib, json, os, pathlib, subprocess, time
ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = pathlib.Path('/data1/liuyeqiang/WWW_jev_fastcheck_runtime/20261014_v1')
REPORTS = ROOT / 'reports/JEV_FASTCHECK'
DOCS = ROOT / 'docs/JEV_FASTCHECK'
BRANCH = 'jev/closed-loop-fastcheck-20261014'
BASE = '2ab3d47e6ddef7aff9161c1e95dada9918cc01b9'
PYTHON = '/home/liuyeqiang/anaconda3/envs/GMT/bin/python'
RECOVERY_RUNTIME = pathlib.Path('/data1/liuyeqiang/WWW_jev_recovery_runtime/20261012_v1')
SEED = 20261009
def read(p): return json.loads(pathlib.Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(2**20),b''): h.update(chunk)
 return h.hexdigest()
def ref(p):
 p=pathlib.Path(p); return dict(path=str(p),SHA256=sha(p),bytes=p.stat().st_size)
def save(p,v):
 p=pathlib.Path(p)
 assert any(p.resolve().is_relative_to(r.resolve()) for r in (OUT,REPORTS)),p
 p.parent.mkdir(parents=True,exist_ok=True); t=p.with_name(p.name+'.tmp')
 t.write_text(json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n'); t.replace(p)
def configure_paths():
 # Historical modules import one another's names. Keep their source unchanged
 # and explicitly route all checkpoint reads back to the frozen runtime.
 import jev_recovery_learning as learning
 learning.OUT=RECOVERY_RUNTIME
 return learning
def load_policy(arm,trained=True):
 learning=configure_paths()
 if trained: return learning.load_model(arm,SEED,'pilot',3,'XIII')[:2]
 from jev_oneweek_online import FrozenPlanControl
 model,cp=learning.initialize('jev',SEED,3,'XIII')
 class Parent(FrozenPlanControl):
  def forward(self,x):
   return super().forward({k:v for k,v in x.items() if k not in ('anchor_visual','anchor_mask')})
 return Parent(model.identity,'parent').cuda().eval(),cp
def binding(scope):
 return dict(scope=scope,source_HEAD=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
  source_SHA256={str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'reproduction_tools').glob('jev_fastcheck_*.py'))},
  protocol=ref(REPORTS/'PREREGISTRATION.json'),CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),
  official_TEST=False,Full24=False,sealed20_21_22=True,GT_actor_inputs=False)
