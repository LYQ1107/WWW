"""Phase IX isolated provenance and no-heldout/no-history-mutation guards."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
for p in (ROOT,ROOT/'reproduction_tools',ROOT/'third_party/CenterNet2'):sys.path.insert(0,str(p))
OUT=Path('/home/liuyeqiang/WWW_jev_phase9_runtime/20261008_v1')
PREVIOUS=Path('/home/liuyeqiang/WWW_jev_phase8_runtime/20261008_v1')
REPORTS=ROOT/'reports/JEV_PHASE9'
BASE='4190b803ed83bde1a44a015d15d03ceff68e03d0'
B2=Path('/home/liuyeqiang/WWW_jev_phase5_runtime/20261007/minimal_training/B2/calibration/model_calibrated.pth')
B2_SHA='f2aa3dd2b564d90bfbfb62dc0518b0b2107a931ed7134f8c0d19f5d152b94ed7'
PYTHON='/home/liuyeqiang/anaconda3/envs/GMT/bin/python'
TRAIN=[12,13,14,16];VAL=[17,18,19]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def save(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
 tmp.write_text(json.dumps(obj,sort_keys=True,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def protect():
 assert sha(B2)==B2_SHA
 refs={'jev/www-jev-phase5-20261007':'a37083dac0a23eb3846cb252eeb975982aaaabc9','jev/www-jev-lifecycle-phase6-20261008':'40cbc0ecc22bc16c4e3602eefe5eb06e2d6e319e','jev/www-jev-phase7-causal-structured-20261008':'ecc0e63f544cbbc272797c17ea285b0b8cce5f80','jev/www-jev-phase8-corrective-association-20261008':BASE}
 for b,e in refs.items():assert subprocess.check_output(['git','rev-parse',b],cwd=ROOT,text=True).strip()==e
 assert not subprocess.check_output(['git','diff',BASE,'--name-only','--','reports/JEV_PHASE5','reports/JEV_PHASE6','reports/JEV_PHASE7','reports/JEV_PHASE8'],cwd=ROOT,text=True).strip()
 assert not subprocess.check_output(['git','diff','--name-only','--','reports/JEV_PHASE8'],cwd=ROOT,text=True).strip()
def binding():
 return {'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'dirty':bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()),'B2_sha256':sha(B2),'sources':{str(p.relative_to(ROOT)):sha(p)for p in sorted((ROOT/'reproduction_tools').glob('*phase9*.py'))},'candidate_production_sources':{str(p.relative_to(ROOT)):sha(p)for p in sorted((ROOT/'gtr/modeling').glob('jev_candidate*.py'))},'gtr_rcnn_sha256':sha(ROOT/'gtr/modeling/meta_arch/gtr_rcnn.py'),'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),'Full24':False,'official_TEST':False,'heldout_sealed':True}
def native_lab(video,device='cuda:0'):
 assert video in TRAIN+VAL,'heldout sealed'
 protect();OUT.mkdir(parents=True,exist_ok=True)
 import jev_phase6_common as c;c.OUT=OUT
 import jev_phase6_rollouts as r;r.OUT=OUT
 import jev_phase7_common as c7;c7.OUT=OUT
 return r.NativeReplayLab(video,device,compact_context=True)
