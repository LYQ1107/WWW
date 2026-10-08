"""Phase X isolation, pinned sources and fail-closed scientific guards."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
for path in (ROOT,ROOT/'reproduction_tools',ROOT/'third_party/CenterNet2'):sys.path.insert(0,str(path))
OUT=Path('/home/liuyeqiang/WWW_jev_phase10_runtime/20261008_v1')
PREVIOUS=Path('/home/liuyeqiang/WWW_jev_phase9_runtime/20261008_v1')
REPORTS=ROOT/'reports/JEV_PHASE10'
BASE='af081a9371ff0634d384d6106aa2a542e5b69afa'
TRAIN=[12,13,14,16];VAL=[17,18,19];HELDOUT=[20,21,22]
CONFIG=ROOT/'configs/VISION_test.yaml'
FOUNDATION=Path('/data1/liuyeqiang/WWW/outputs/stage2_single_gpu/model_20000.pth')
CACHE=Path('/data1/liuyeqiang/WWW/outputs/research_final_v2/off/perception_cache_train')
B2=Path('/home/liuyeqiang/WWW_jev_phase5_runtime/20261007/minimal_training/B2/calibration/model_calibrated.pth')
B2_SHA='f2aa3dd2b564d90bfbfb62dc0518b0b2107a931ed7134f8c0d19f5d152b94ed7'
PYTHON='/home/liuyeqiang/anaconda3/envs/GMT/bin/python'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def save(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,sort_keys=True,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def protect():
 assert sha(B2)==B2_SHA
 refs={'jev/www-jev-phase5-20261007':'a37083dac0a23eb3846cb252eeb975982aaaabc9','jev/www-jev-lifecycle-phase6-20261008':'40cbc0ecc22bc16c4e3602eefe5eb06e2d6e319e','jev/www-jev-phase7-causal-structured-20261008':'ecc0e63f544cbbc272797c17ea285b0b8cce5f80','jev/www-jev-phase8-corrective-association-20261008':'4190b803ed83bde1a44a015d15d03ceff68e03d0','jev/www-jev-phase9-native-candidate-choice-20261008':BASE}
 for branch,expected in refs.items():assert subprocess.check_output(['git','rev-parse',branch],cwd=ROOT,text=True).strip()==expected
 assert not subprocess.check_output(['git','diff',BASE,'--name-only','--','reports/JEV_PHASE5','reports/JEV_PHASE6','reports/JEV_PHASE7','reports/JEV_PHASE8','reports/JEV_PHASE9'],cwd=ROOT,text=True).strip()
 assert not subprocess.check_output(['git','diff','--name-only','--','reports/JEV_PHASE9'],cwd=ROOT,text=True).strip()
def binding():
 return {'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'dirty':bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()),'source_root':str(ROOT),'config_SHA256':sha(CONFIG),'foundation_SHA256':sha(FOUNDATION),'cache_index_SHA256':sha(CACHE/'index.jsonl'),'B2_SHA256':sha(B2),'sources':{str(p.relative_to(ROOT)):sha(p)for p in sorted(list((ROOT/'reproduction_tools').glob('*phase10*.py'))+list((ROOT/'gtr/modeling').glob('jev_candidate*.py'))+list((ROOT/'gtr/modeling').glob('jev_native*.py'))+[ROOT/'gtr/modeling/meta_arch/gtr_rcnn.py'])},'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),'seed':20261008,'trajectory_master_seed':20261006,'Full24':False,'official_TEST':False}
def build_model(video):
 assert video in TRAIN+VAL,'heldout stays sealed before model freeze'
 from build_jev_counterfactual_v2 import build_formal_gmt_engine
 from gtr.modeling.jev_runtime import JEVRuntimePolicy
 from gtr.modeling.jev_perception_cache import FrozenPerceptionCache
 from gtr.modeling.jev_candidate_policy import CandidateValuePolicy
 e=build_formal_gmt_engine(config_file=CONFIG,checkpoint=FOUNDATION,device='cuda:0',view_num=2,history_limit=80)
 m=e.association_fn.model;m.to('cuda:0').eval();m.jev_enabled=True;m.jev_mode='off';m.jev_policy=JEVRuntimePolicy('off',None);m.jev_perception_cache_reader=FrozenPerceptionCache(CACHE);m.jev_perception_cache=None;m.jev_candidate_policy=CandidateValuePolicy('gmt_compat')
 return m
def inputs(video,frames):
 from gtr.modeling.jev_perception_cache import FrozenPerceptionCache
 cache=FrozenPerceptionCache(CACHE);values=[]
 for view in range(2):
  for frame in range(frames):
   p=cache.load(video,frame,view);h,w=p['image_size'];values.append({'video_id':video,'view_num':2,'height':int(h),'width':int(w),'image':None})
 return values
