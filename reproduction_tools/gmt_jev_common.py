"""Bounded full-GMT study. Historical data1 assets remain read-only."""
import hashlib,json,os,pathlib,shutil,subprocess,time
ROOT=pathlib.Path(__file__).resolve().parents[1]
REPORTS=ROOT/'reports/GMT_JEV';DOCS=ROOT/'docs/GMT_JEV'
OUT=pathlib.Path('/home/liuyeqiang/WWW_gmt_jev_runtime/20261014_v1')
BASE='04324354e9e9971eec6d937b961590a0cdbf13a9'
BRANCH='research/gmt-jev-risk-association-20261014'
PYTHON='/home/liuyeqiang/anaconda3/envs/GMT/bin/python';SEED=20261009
STAGE2=pathlib.Path('/data1/liuyeqiang/WWW/outputs/stage2_single_gpu/model_20000.pth')
STAGE2_SHA='cd72823824d16c86ed27c2dfc8323de610aa9f6c0c0b29249aa3de609deabce8'
ANNOTATIONS=pathlib.Path('/data/DATASETS/TRACKING/JDE/VisionTrack/annotations/train.json')
IMAGES=pathlib.Path('/data/DATASETS/TRACKING/JDE/VisionTrack/train')
def read(p):return json.loads(pathlib.Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for x in iter(lambda:f.read(2**20),b''):h.update(x)
 return h.hexdigest()
def ref(p):return dict(path=str(p),SHA256=sha(p),bytes=pathlib.Path(p).stat().st_size)
def retained():return sum(p.stat().st_size for r in [ROOT,OUT] for p in r.rglob('*') if p.is_file() and not p.is_symlink())
def reserve():
 assert shutil.disk_usage('/home/liuyeqiang').free/2**30>=30
 assert retained()<512*2**20
def save(p,d):
 p=pathlib.Path(p);assert p.resolve().is_relative_to(ROOT) or p.resolve().is_relative_to(OUT)
 reserve();p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+'.tmp')
 q.write_text(json.dumps(d,indent=2,sort_keys=True,allow_nan=False,ensure_ascii=False)+'\n');q.replace(p)
def binding():return dict(HEAD=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),GMT_checkpoint=ref(STAGE2) if STAGE2.exists() else None,
 CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),GT_actor_input=False,official_TEST=False,Full24=False,sealed20_21_22=True,
 source_SHA256={str(p.relative_to(ROOT)):sha(p) for p in list((ROOT/'reproduction_tools').glob('gmt_jev_*.py'))+list((ROOT/'gtr/modeling/gmt_jev').glob('*.py'))+[ROOT/n for n in ['gtr/modeling/meta_arch/gtr_rcnn.py','gtr/modeling/roi_heads/gtr_roi_heads.py','gtr/modeling/roi_heads/transformer.py','configs/VISION_test.yaml']]},goal=ref(REPORTS/'MASTER_GOAL.json'))
