"""Strictly validate the real finalStage2 state against the official test model."""
from pathlib import Path
import hashlib,json,os,sys
R=Path(__file__).resolve().parents[1];repo=R/'code/GMT';os.chdir(repo)
sys.path[:0]=[str(repo),str(repo/'third_party/CenterNet2')]
import torch,gtr
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from centernet.config import add_centernet_config
from gtr.config import add_gtr_config
record=json.loads((R/'manifests/stage2_complete.json').read_text());assert record['iterations']==20000 and record['checkpoint_reload']=='PASS'
path=Path(record['checkpoint']);digest=hashlib.sha256()
with path.open('rb') as f:
 for block in iter(lambda:f.read(8*1024*1024),b''):digest.update(block)
assert digest.hexdigest()==record['sha256']
checkpoint=torch.load(path,map_location='cpu');state=checkpoint['model']
cfg=get_cfg();add_centernet_config(cfg);add_gtr_config(cfg);cfg.merge_from_file(str(repo/'configs/VISION_test.yaml'));cfg.MODEL.WEIGHTS=str(path);cfg.freeze()
model=build_model(cfg);expected=model.state_dict()
assert set(expected)==set(state),{'missing':sorted(set(expected)-set(state)),'unexpected':sorted(set(state)-set(expected))}
for key,tensor in state.items():
 assert expected[key].shape==tensor.shape,key
 if tensor.is_floating_point() or tensor.is_complex():assert bool(torch.isfinite(tensor).all()),key
model.load_state_dict(state,strict=True)
report={'status':'PASS','checkpoint':str(path),'checkpoint_sha256':digest.hexdigest(),'test_config_sha256':hashlib.sha256((repo/'configs/VISION_test.yaml').read_bytes()).hexdigest(),'state_keys':len(state),'missing_keys':[],'unexpected_keys':[],'shape_mismatches':[],'parameter_finiteness':'PASS','inference_executed':False}
(R/'manifests/final_inference_checkpoint_preflight.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
