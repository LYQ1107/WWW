"""Isolated Phase VIII provenance, guards and compatible native lab imports."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
for p in(ROOT,ROOT/'reproduction_tools',ROOT/'third_party/CenterNet2'):sys.path.insert(0,str(p))
OUT=Path('/home/liuyeqiang/WWW_jev_phase8_runtime/20261008_v1')
REPORTS=ROOT/'reports/JEV_PHASE8';PYTHON='/home/liuyeqiang/anaconda3/envs/GMT/bin/python'
PREREG=REPORTS/'PREREGISTRATION.json'
PREREG_SHA='6b4c03d07261ca3b0113a1c2654ab8cfe77ccbff216ed8f0de8d854eb1d78be4'
ANCHOR_AMENDMENT_SHA='7217e47292193e3dcfc27b75a4fc84d7779fa9c4b992bb21861a80635160cb41'
B2=Path('/home/liuyeqiang/WWW_jev_phase5_runtime/20261007/minimal_training/B2/calibration/model_calibrated.pth')
B2_SHA='f2aa3dd2b564d90bfbfb62dc0518b0b2107a931ed7134f8c0d19f5d152b94ed7'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb')as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n');t.replace(p)
def protect():
    assert sha(PREREG)==PREREG_SHA,'preregistration changed'
    assert sha(REPORTS/'ANCHOR_CONTRACT_AMENDMENT.json')==ANCHOR_AMENDMENT_SHA,'anchor amendment changed'
    assert sha(B2)==B2_SHA,'protected B2 changed'
    for branch,expected in [('jev/www-jev-phase5-20261007','a37083dac0a23eb3846cb252eeb975982aaaabc9'),('jev/www-jev-lifecycle-phase6-20261008','40cbc0ecc22bc16c4e3602eefe5eb06e2d6e319e'),('jev/www-jev-phase7-causal-structured-20261008','ecc0e63f544cbbc272797c17ea285b0b8cce5f80')]:
        assert subprocess.check_output(['git','rev-parse',branch],cwd=ROOT,text=True).strip()==expected
    assert not subprocess.check_output(['git','diff','ecc0e63f544cbbc272797c17ea285b0b8cce5f80','--name-only','--','gtr'],cwd=ROOT,text=True).strip(),'unregistered production source change'
def binding():
    return {'start_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
       'worktree_dirty':bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()),
       'phase8_source_sha256':{str(p.relative_to(ROOT)):sha(p)for p in sorted((ROOT/'reproduction_tools').glob('*phase8*.py'))},
       'frozen_production_and_adapter_source_sha256':json.loads((REPORTS/'CODE_CONTRACT_AUDIT.json').read_text())['frozen_source_sha256'],
       'config_sha256':sha(ROOT/'configs/VISION_test.yaml'),'preregistration_sha256':sha(PREREG),'anchor_amendment_sha256':ANCHOR_AMENDMENT_SHA,'B2_sha256':sha(B2),
       'foundation_sha256':json.loads((REPORTS/'STORAGE_BEFORE.json').read_text())['foundation_sha256'],'seed':20261008,
       'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),'official_TEST_read':False,'Full24_started':False}
def native_lab(video,device='cuda:0'):
    protect();OUT.mkdir(parents=True,exist_ok=True)
    import jev_phase6_common as c;c.OUT=OUT
    import jev_phase6_rollouts as r;r.OUT=OUT
    import jev_phase7_common as c7;c7.OUT=OUT
    return r.NativeReplayLab(video,device,compact_context=True)

class FrozenOFFActor:
    """Current and future inference uses online OFF semantics, never GT labels."""
    def decide(self,feature,question,legal,*,off_action=None,context=None):
        from types import SimpleNamespace
        assert off_action in legal
        return SimpleNamespace(committed_action=off_action,probabilities={a:float(a==off_action)for a in legal})
