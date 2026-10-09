"""Independent Phase XII outputs and immutable scientific provenance."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
for path in (ROOT,ROOT/'reproduction_tools',ROOT/'third_party/CenterNet2'):sys.path.insert(0,str(path))
OUT=Path('/home/liuyeqiang/WWW_jev_phase12_runtime/20261009_v1')
PREVIOUS=Path('/home/liuyeqiang/WWW_jev_phase10_runtime/20261008_v1')
REPORTS=ROOT/'reports/JEV_PHASE12'
BASE='957644fa2e300388a603fb352f73917e385bcd24'
TRAIN=[12,13,14,16];VAL=[17,18,19];HELDOUT=[20,21,22]
PYTHON='/home/liuyeqiang/anaconda3/envs/GMT/bin/python'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()

def save(path,data):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(data,indent=2,sort_keys=True,allow_nan=False)+'\n');temp.replace(p)

def protect():
    from jev_phase10_common import protect as earlier_protect
    earlier_protect()
    frozen=json.loads((REPORTS/'PROTECTED_BASELINE.json').read_text())
    for name,digest in frozen['historical_report_SHA256'].items():assert sha(ROOT/name)==digest,name
    assert subprocess.check_output(['git','rev-parse','jev/www-jev-phase10-native-gallery-repair-20261008'],cwd=ROOT,text=True).strip()==BASE
    assert sha(PREVIOUS/'dataset_v3/DATASET.pth')==frozen['PhaseX_dataset_SHA256']
    freeze=json.loads((REPORTS/'ARCHITECTURE_FREEZE.json').read_text())
    for name,digest in freeze['files_SHA256'].items():assert sha(ROOT/name)==digest,name

def binding():
    files=list((ROOT/'gtr/modeling/visual_jev_mcmot').glob('*.py'))+list((ROOT/'reproduction_tools').glob('*phase12*.py'))+[ROOT/'gtr/modeling/meta_arch/gtr_rcnn.py',ROOT/'gtr/config.py']
    return {'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'source_root':str(ROOT),'dirty':bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()),'source_SHA256':{str(p.relative_to(ROOT)):sha(p) for p in sorted(files)},'architecture_freeze_SHA256':sha(REPORTS/'ARCHITECTURE_FREEZE.json'),'dataset_SHA256':sha(PREVIOUS/'dataset_v3/DATASET.pth'),'config_SHA256':sha(ROOT/'configs/VISION_test.yaml'),'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),'heldout_status':'SEALED','Full24':False,'official_TEST':False}

def build_model(video):
    if video not in TRAIN+VAL:raise ValueError('Phase XII heldout sealed')
    from jev_phase10_common import build_model as native_builder,configure_candidate_torchscript
    configure_candidate_torchscript()
    return native_builder(video)

def inputs(video,frames):
    if video not in TRAIN+VAL:raise ValueError('Phase XII heldout sealed')
    from jev_phase10_common import inputs as native_inputs
    return native_inputs(video,frames)

def parameters(model):
    return sum(p.numel() for p in model.parameters())

def profile_macs(model,args):
    import torch
    totals={'linear':0,'attention':0};handles=[]
    def linear(module,values,result):
        totals['linear']+=values[0].numel()//module.in_features*module.in_features*module.out_features
    def attention(module,values,result):
        q,k,v=values[:3];b,l,d=q.shape;s=k.shape[1]
        totals['attention']+=b*((l+2*s)*d*d+l*d*d+2*l*s*d)
    for module in model.modules():
        if isinstance(module,torch.nn.Linear):handles.append(module.register_forward_hook(linear))
        if isinstance(module,torch.nn.MultiheadAttention):handles.append(module.register_forward_hook(attention))
    with torch.no_grad():model(*args)
    for handle in handles:handle.remove()
    return {'MAC':sum(totals.values()),'FLOPs_matmul':2*sum(totals.values()),'components':totals,'excludes':'normalization/softmax/GELU/memory movement','params':parameters(model)}
