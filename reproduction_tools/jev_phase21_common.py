"""Bounded existing-model execution audit; frozen scientific results remain read-only."""
import hashlib,json,os,subprocess,sys,time,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'reproduction_tools'))
OUT=Path(os.environ.get('JEV_PHASE21_OUT','/home/liuyeqiang/WWW_jev_phase21_runtime/20261011_v1'))
OUT.mkdir(parents=True,exist_ok=True)
REPORTS=ROOT/'reports/JEV_PHASE21';MASTER=ROOT/'reports/JEV_MASTER'
PRIOR=Path('/home/liuyeqiang/WWW_jev_phase20')
TEACHER=Path('/home/liuyeqiang/WWW_jev_phase17_runtime/20261011_v1')
BASE='eda4a9ba9166d67df1a708d8267e6721af90ba2e'
PYTHON=str(TEACHER/'visual_env/bin/python')
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def ref(p):return dict(path=str(p),SHA256=sha(p))
def save(p,x):
 p=Path(p);assert any(p.resolve().is_relative_to(d.resolve()) for d in [OUT,REPORTS,MASTER])
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp.'+str(os.getpid()))
 t.write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n');t.replace(p)
def protect():
 subprocess.run(['git','merge-base','--is-ancestor',BASE,'HEAD'],cwd=ROOT,check=True)
 paths=subprocess.check_output(['git','diff',BASE,'--name-only'],cwd=ROOT,text=True).splitlines()
 assert all(p.startswith(('reports/JEV_PHASE21/','reports/JEV_MASTER/','docs/PHASE21_','docs/WWW_','reproduction_tools/jev_phase21_')) or p=='reproduction_tools/jev_master_goal_controller.py' for p in paths),paths
 assert shutil.disk_usage(OUT).free>=30*2**30
 assert sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file() and not p.is_symlink())<=64*2**20
def binding():return dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
 source_SHA256={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'reproduction_tools').glob('jev_phase21_*.py')},
 preregistration=ref(REPORTS/'PREREGISTRATION.json'),frozen_prior=BASE,new_training=False,new_native_actions=0,
 sealed20_21_22_accessed=False,official_TEST=False,Full24=False)
