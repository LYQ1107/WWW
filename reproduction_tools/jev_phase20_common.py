"""Read-only past-evidence audit; no additional tracking actions or training."""
import hashlib,json,os,subprocess,sys,time,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'reproduction_tools'))
OUT=Path('/home/liuyeqiang/WWW_jev_phase20_runtime/20261011_v1')
REPORTS=ROOT/'reports/JEV_PHASE20';MASTER=ROOT/'reports/JEV_MASTER'
PRIOR=Path('/home/liuyeqiang/WWW_jev_phase19')
PRIOR_OUT=Path('/home/liuyeqiang/WWW_jev_phase19_runtime/20261011_v1')
BASE='863bababd02641f605891b2365cfd5c60fb3c79f'
PYTHON='/home/liuyeqiang/anaconda3/envs/GMT/bin/python'
SCOPE=list(range(2,12))+[15]
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
def size(p):return sum(f.stat().st_size for f in p.rglob('*') if f.is_file() and not f.is_symlink())
def protect():
 subprocess.run(['git','merge-base','--is-ancestor',BASE,'HEAD'],cwd=ROOT,check=True)
 paths=subprocess.check_output(['git','diff',BASE,'--name-only'],cwd=ROOT,text=True).splitlines()
 assert all(p.startswith(('reports/JEV_PHASE20/','reports/JEV_MASTER/','docs/PHASE20_','docs/WWW_','reproduction_tools/jev_phase20_')) or p=='reproduction_tools/jev_master_goal_controller.py' for p in paths),paths
 assert shutil.disk_usage(OUT).free>=30*2**30
 assert size(OUT)<=64*2**20 and size(OUT)+size(PRIOR_OUT)<=8*2**30
def census(video):
 assert video in SCOPE
 r=read(PRIOR/'reports/JEV_PHASE19/CENSUS_RESULTS_INDEX.json')['per_video'][str(video)]
 assert sha(r['path'])==r['SHA256'];return read(r['path'])
def binding():return dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
 source_SHA256={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'reproduction_tools').glob('jev_phase20_*.py')},
 preregistration=ref(REPORTS/'PREREGISTRATION.json'),frozen_prior=BASE,no_training=True,new_native_actions=0,
 sealed20_21_22_accessed=False,official_TEST=False,Full24=False)
