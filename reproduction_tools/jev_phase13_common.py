"""Independent Phase XIII provenance, checkpoint truth and sealed runtime inputs."""
import os,json,hashlib,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
for p in [ROOT,ROOT/'reproduction_tools',ROOT/'third_party/CenterNet2']:sys.path.insert(0,str(p))
OUT=Path('/home/liuyeqiang/WWW_jev_phase13_runtime/20261009_v1')
REPORTS=ROOT/'reports/JEV_PHASE13'
BASE='fffd0a1b7c04a19513f0b8fa07a326572533f0ae'
TRAIN=[12,13,14,16];VAL=[17,18,19];SEALED=[20,21,22]
PYTHON='/home/liuyeqiang/anaconda3/envs/GMT/bin/python'
STAGE1=Path('/data1/liuyeqiang/WWW/outputs/stage1_single_gpu/model_16000.pth')
STAGE2=Path('/data1/liuyeqiang/WWW/outputs/stage2_single_gpu/model_20000.pth')
ANNOTATIONS=Path('/data/DATASETS/TRACKING/JDE/VisionTrack/annotations/train.json')
IMAGES=Path('/data/DATASETS/TRACKING/JDE/VisionTrack/train')
STAGE1_SHA='143e84deb50bdf5379c8f4463f1b9b237132e9281726b9cff469aff8c9dbe64e'
STAGE2_SHA='cd72823824d16c86ed27c2dfc8323de610aa9f6c0c0b29249aa3de609deabce8'
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for x in iter(lambda:f.read(1048576),b''):h.update(x)
    return h.hexdigest()
def save(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n');t.replace(p)
def binding():
    files=list((ROOT/'gtr/modeling/jev_stage2').glob('*.py'))+list((ROOT/'reproduction_tools').glob('*phase13*.py'))+list((ROOT/'reproduction_tools').glob('*jev_stage2*.py'))+[ROOT/'gtr/modeling/meta_arch/gtr_rcnn.py']
    return {'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'source_root':str(ROOT),'dirty':bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()),'source_SHA256':{str(p.relative_to(ROOT)):sha(p) for p in sorted(set(files))},'Stage1_SHA256':STAGE1_SHA,'Stage2_original_SHA256':STAGE2_SHA,'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),'TRAIN':TRAIN,'VAL':VAL,'heldout':'SEALED','Full24':False,'official_TEST':False}
def protect():
    from jev_phase12_common import protect as old
    old()
    for name in ['reports/JEV_PHASE12','docs/JEV_PHASE12']:
        changed=subprocess.check_output(['git','diff',BASE,'--name-only','--',name],cwd=ROOT,text=True).strip();assert not changed,changed
    assert sha(STAGE1)==STAGE1_SHA and sha(STAGE2)==STAGE2_SHA
def allowed(video):
    if video not in TRAIN+VAL:raise ValueError('Phase XIII video sealed or outside frozen partition')
