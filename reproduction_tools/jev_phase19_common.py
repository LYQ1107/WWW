"""Bounded authorized new scope; predecessors are immutable."""
import hashlib,json,os,subprocess,sys,time,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
for p in [ROOT,ROOT/'reproduction_tools',ROOT/'third_party/CenterNet2']:sys.path.insert(0,str(p))
BASE='efa231955c41d40ee9135a03a545b3e88ff028e8'
OUT=Path('/home/liuyeqiang/WWW_jev_phase19_runtime/20261011_v1')
REPORTS=ROOT/'reports/JEV_PHASE19';MASTER=ROOT/'reports/JEV_MASTER'
PYTHON='/home/liuyeqiang/anaconda3/envs/GMT/bin/python'
TRAIN=list(range(2,10));RESERVED=[10,11,15];SCOPE=TRAIN+RESERVED
ANNOTATIONS=Path('/data/DATASETS/TRACKING/JDE/VisionTrack/annotations/train.json')
IMAGES=Path('/data/DATASETS/TRACKING/JDE/VisionTrack/train')
STAGE1=Path('/data1/liuyeqiang/WWW/outputs/stage1_single_gpu/model_16000.pth')
STAGE1_SHA='143e84deb50bdf5379c8f4463f1b9b237132e9281726b9cff469aff8c9dbe64e'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def ref(p):return dict(path=str(p),SHA256=sha(p))
def save(p,x):
 p=Path(p);assert any(p.resolve().is_relative_to(d.resolve()) for d in [OUT,REPORTS,MASTER]),p
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp.'+str(os.getpid()));t.write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n');t.replace(p)
def allowed(video):assert video in SCOPE,('outside authorized scope',video)
def census_case(video):
 allowed(video);item=read(REPORTS/'CENSUS_RESULTS_INDEX.json')['per_video'][str(video)]
 assert sha(item['path'])==item['SHA256'];return Path(item['path']).parent
def protect():
 subprocess.run(['git','merge-base','--is-ancestor',BASE,'HEAD'],cwd=ROOT,check=True)
 names=subprocess.check_output(['git','diff',BASE,'--name-only'],cwd=ROOT,text=True).splitlines()
 allowed_names=('reports/JEV_PHASE19/','reports/JEV_MASTER/','docs/PHASE19_','docs/WWW_','gtr/modeling/jev_phase19/','reproduction_tools/jev_phase19_')
 assert all(p.startswith(allowed_names) or p=='reproduction_tools/jev_master_goal_controller.py' for p in names),names
 assert shutil.disk_usage(OUT).free>=30*2**30,'home reserve below30GiB'
 assert sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file() and not p.is_symlink())<=8*2**30,'round exceeds authorized8GiB'
def binding(**kw):
 return dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),source_SHA256={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'reproduction_tools').glob('jev_phase19_*.py')},preregistration=ref(REPORTS/'PREREGISTRATION.json'),frozen_prior=BASE,TRAIN=TRAIN,RESERVED=RESERVED,sealed20_21_22_accessed=False,official_TEST=False,Full24=False,no_training=True,**kw)

