"""Independent recovery namespace, immutable inherited evidence, bounded storage."""
import os, sys, json, hashlib, pathlib, subprocess, shutil, time
ROOT=pathlib.Path(__file__).resolve().parents[1]
for p in (ROOT,ROOT/'reproduction_tools',ROOT/'third_party/CenterNet2'):sys.path.insert(0,str(p))
OUT=pathlib.Path('/data1/liuyeqiang/WWW_jev_recovery_runtime/20261012_v1')
REPORTS=ROOT/'reports/JEV_RECOVERY'
DOCS=ROOT/'docs/JEV_RECOVERY'
BASE='0d051bbcd31bd44f32f5e5f0005c0d8b1e77eac1'
BRANCH='jev/www-jev-recovery-20261012'
PYTHON='/home/liuyeqiang/anaconda3/envs/GMT/bin/python'
SEEDS=[20261008,20261009,20261010]
TRAIN=[12,13,14,16]
DEV=[17,18,19]
OLD_OUT=pathlib.Path('/data1/liuyeqiang/WWW_jev_oneweek_runtime/20261012_v1')
def read(p):return json.loads(pathlib.Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*2**20),b''):h.update(b)
 return h.hexdigest()
def ref(p):
 p=pathlib.Path(p);return dict(path=str(p),SHA256=sha(p),bytes=p.stat().st_size)
def save(p,v):
 p=pathlib.Path(p);assert any(p.resolve().is_relative_to(q.resolve()) for q in (OUT,REPORTS)),p
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp.'+str(os.getpid()))
 t.write_text(json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\n');t.replace(p)
def protect():
 assert subprocess.check_output(['git','symbolic-ref','--short','HEAD'],cwd=ROOT,text=True).strip()==BRANCH
 subprocess.run(['git','merge-base','--is-ancestor',BASE,'HEAD'],cwd=ROOT,check=True)
 names=subprocess.check_output(['git','diff',BASE,'--name-only'],cwd=ROOT,text=True).splitlines()
 assert all(n.startswith(('docs/JEV_RECOVERY/','reports/JEV_RECOVERY/','configs/JEV_RECOVERY/','gtr/modeling/jev_recovery/','reproduction_tools/jev_recovery_')) for n in names),names
 for p in ('/home','/data1'):assert shutil.disk_usage(p).free>=30*2**30,'30GiB reserve'
 size=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file());assert size<=8*2**30,'8GiB new retention cap'
def binding(scope):
 files=list((ROOT/'reproduction_tools').glob('jev_recovery_*.py'))+list((ROOT/'gtr/modeling/jev_recovery').glob('*.py'))
 files+=list((ROOT/'gtr/modeling/jev_oneweek').glob('*.py'))
 return dict(scope=scope,commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),sources={str(p.relative_to(ROOT)):sha(p) for p in files},CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),protocol=ref(REPORTS/'PREREGISTRATION.json'),official_TEST=False,sealed20_21_22=True)
def failure(stage,error):save(OUT/'failures'/f'{stage}_{time.time_ns()}.json',dict(status='FAIL',stage=stage,error=str(error),binding=binding(stage)))
