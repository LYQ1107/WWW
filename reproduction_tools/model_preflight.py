"""Architecture/checkpoint compatibility only; no data, training or saved model."""
from pathlib import Path
import os,sys,json,gc
r=Path('/data3/liuyeqiang/GMT_VisionTrack_repro');repo=r/'code/GMT'
os.chdir(repo);sys.path[:0]=[str(repo),str(repo/'third_party/CenterNet2')]
os.environ['CUDA_VISIBLE_DEVICES']='0'
for key in ['HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy']:os.environ.pop(key,None)
import torch,gtr
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from centernet.config import add_centernet_config
from gtr.config import add_gtr_config
summaries={};keys={}
for stage in [1,2]:
 cfg=get_cfg();add_centernet_config(cfg);add_gtr_config(cfg)
 cfg.merge_from_file(str(repo/f'configs/VISION_stage{stage}.yaml'))
 cfg.SOLVER.IMS_PER_BATCH=10
 model=build_model(cfg)
 shapes={k:list(v.shape) for k,v in model.state_dict().items()};keys[stage]=shapes
 summaries[str(stage)]={'parameters':sum(p.numel() for p in model.parameters()),'state_keys':len(shapes)}
 if stage==1:
  class InspectCheckpointer(DetectionCheckpointer):
   def _load_model(self, checkpoint):
    self.incompatible=super()._load_model(checkpoint)
    return self.incompatible
  checkpointer=InspectCheckpointer(model)
  checkpointer.load(str(r/'checkpoints/backbone/CH_FPN_1x.pth'))
  incompatible=checkpointer.incompatible
  summaries['backbone_load']={'missing_keys':incompatible.missing_keys,'unexpected_keys':incompatible.unexpected_keys,'incorrect_shapes':getattr(incompatible,'incorrect_shapes',[])}
 del model;gc.collect();torch.cuda.empty_cache()
summaries['stage1_to_stage2_structure']={'only_stage1':sorted(keys[1].keys()-keys[2].keys()),'only_stage2':sorted(keys[2].keys()-keys[1].keys()),'shape_mismatches':{k:[keys[1][k],keys[2][k]] for k in keys[1].keys()&keys[2].keys() if keys[1][k]!=keys[2][k]}}
(r/'reports/model_preflight.json').write_text(json.dumps(summaries,indent=2))
print(json.dumps({k:v for k,v in summaries.items() if k!='backbone_load'},indent=2))
print('backbone missing',len(summaries['backbone_load']['missing_keys']),'unexpected',len(summaries['backbone_load']['unexpected_keys']),'shape_mismatches',len(summaries['backbone_load']['incorrect_shapes']))
print('No training or model checkpoint saved; full actual Stage1→Stage2 load remains a later gate.')
