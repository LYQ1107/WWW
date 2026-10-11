"""Phase XVII bounded provenance; prior scientific paths are read-only."""
import hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
for p in [ROOT,ROOT/'reproduction_tools',ROOT/'third_party/CenterNet2']:sys.path.insert(0,str(p))
OUT=Path('/home/liuyeqiang/WWW_jev_phase17_runtime/20261011_v1')
REPORTS=ROOT/'reports/JEV_PHASE17';MASTER=ROOT/'reports/JEV_MASTER'
BASE='666e2b922eb676b3552db0c6dc8b67b1b3978450'
PYTHON='/home/liuyeqiang/anaconda3/envs/GMT/bin/python'
XVI=Path('/home/liuyeqiang/WWW_jev_phase16_runtime/20261011_v1')
XV=Path('/home/liuyeqiang/WWW_jev_phase15_runtime/20261010_v1')
TRAIN=[12,13,14,16];DEV=[17,18,19]
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def ref(p):return {'path':str(p),'SHA256':sha(p)}
def save(p,v):
 p=Path(p);assert any(p.resolve().is_relative_to(d.resolve()) for d in [OUT,REPORTS,MASTER]),p
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp.'+str(os.getpid()));tmp.write_text(json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n');tmp.replace(p)
def protect():
 subprocess.run(['git','merge-base','--is-ancestor',BASE,'HEAD'],cwd=ROOT,check=True)
 names=subprocess.check_output(['git','diff',BASE,'--name-only'],cwd=ROOT,text=True).splitlines()
 allowed=('reports/JEV_PHASE17/','reports/JEV_MASTER/','docs/PHASE17_','docs/WWW_','gtr/modeling/jev_phase17/','reproduction_tools/jev_phase17_','reproduction_tools/jev_master_goal_controller.py')
 assert all(n.startswith(allowed) for n in names),names
 assert shutil.disk_usage(OUT).free>=30*2**30,'home reserve below30GiB'
 total=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file() and not p.is_symlink());assert total<=32*2**30,'runtime cap32GiB'
def binding(**kw):
 return dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),source_SHA256={str(p.relative_to(ROOT)):sha(p) for p in list((ROOT/'reproduction_tools').glob('jev_phase17_*.py'))+list((ROOT/'gtr/modeling/jev_phase17').glob('*.py'))},preregistration=ref(REPORTS/'PREREGISTRATION.json'),TRAIN=TRAIN,reused_DEV=DEV,sealed_20_21_22=True,official_TEST=False,Full24=False,**kw)
