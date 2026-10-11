"""New bounded evidence round; all previous research results remain frozen."""
import hashlib,json,os,subprocess,sys,time,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
for p in [ROOT,ROOT/'reproduction_tools',ROOT/'third_party/CenterNet2']:sys.path.insert(0,str(p))
BASE='d44c30d22c40530a3fb2afa78e76ed049d502e82'
PRIOR=Path('/home/liuyeqiang/WWW_jev_phase17')
OUT=Path('/home/liuyeqiang/WWW_jev_phase18_runtime/20261011_v1');REPORTS=ROOT/'reports/JEV_PHASE18';MASTER=ROOT/'reports/JEV_MASTER'
PYTHON='/home/liuyeqiang/anaconda3/envs/GMT/bin/python';TRAIN=[12,13,16];DIAGNOSTIC=[14]
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def ref(p):return dict(path=str(p),SHA256=sha(p))
def save(p,x):
 p=Path(p);assert any(p.resolve().is_relative_to(d.resolve()) for d in [OUT,REPORTS,MASTER]),p
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp.'+str(os.getpid()));tmp.write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n');tmp.replace(p)
def protect():
 subprocess.run(['git','merge-base','--is-ancestor',BASE,'HEAD'],cwd=ROOT,check=True)
 names=subprocess.check_output(['git','diff',BASE,'--name-only'],cwd=ROOT,text=True).splitlines();allowed=('reports/JEV_PHASE18/','reports/JEV_MASTER/','docs/PHASE18_','docs/WWW_','gtr/modeling/jev_phase18/','reproduction_tools/jev_phase18_')
 assert all(p.startswith(allowed) for p in names),names
 assert shutil.disk_usage(OUT).free>=30*2**30,'home reserve below30GiB'
 assert sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file() and not p.is_symlink())<=3*2**30,'round storage exceeds3GiB'
def binding(**kw):
 return dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),source_SHA256={str(p.relative_to(ROOT)):sha(p) for p in list((ROOT/'reproduction_tools').glob('jev_phase18_*.py'))+list((ROOT/'gtr/modeling/jev_phase18').glob('*.py'))},preregistration=ref(REPORTS/'PREREGISTRATION.json'),frozen_prior=BASE,TRAIN=TRAIN,diagnostic14=True,sealed_20_21_22=True,official_TEST=False,Full24=False,seed=20261009,**kw)
