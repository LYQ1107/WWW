"""Compare original loading with two-key name adaptation on the same real training clip."""
from pathlib import Path
import os,sys,json,copy
r=Path(__file__).resolve().parents[1];repo=r/'code/GMT';os.chdir(repo);sys.path[:0]=[str(repo),str(repo/'third_party/CenterNet2')]
os.environ['CUDA_VISIBLE_DEVICES']='0'
import torch,train_net
from detectron2.engine import default_argument_parser
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.events import EventStorage
from runtime_checkpoint import install
args=default_argument_parser().parse_args(['--config-file',str(repo/'configs/VISION_stage1.yaml'),'SEED','32995656','DATALOADER.NUM_WORKERS','0','MODEL.WEIGHTS',str(r/'checkpoints/backbone/CH_FPN_1x.pth'),'OUTPUT_DIR',str(r/'outputs/diagnose_weight_keys')])
cfg=train_net.setup(args);model=train_net.build_model(cfg);install(model);DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS);model.train()
mapper=train_net.GMTDatasetMapper(cfg,True,augmentations=train_net.build_custom_augmentation(cfg,True));data=next(iter(train_net.build_gtr_train_loader(cfg,mapper=mapper)))
initial=copy.deepcopy(model.state_dict());rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state();records=[]
current={}
def proposal(module,args,result):
 current['proposals']=[{'count':len(x),'above_threshold':int((x.objectness_logits>0.3).sum()),'max_score':float(x.objectness_logits.max()) if len(x) else None} for x in result[0]]
def classify(module,args):current.setdefault('reid_shapes',[]).append(list(args[0].shape))
model.proposal_generator.register_forward_hook(proposal);model.roi_heads.classify_head.register_forward_pre_hook(classify)
weights=torch.load(cfg.MODEL.WEIGHTS,map_location='cpu')['model']
for adapted in [False,True]:
 model.load_state_dict(initial);torch.set_rng_state(rng);torch.cuda.set_rng_state(cuda_rng)
 if adapted:
  with torch.no_grad():
   for suffix in ['weight','bias']:
    getattr(model.proposal_generator.centernet_head.bbox_tower[9].conv,suffix).copy_(weights['proposal_generator.centernet_head.bbox_tower.9.'+suffix])
 current={'adapted_two_conv_keys':adapted,'frames':[x['file_name'] for x in data]}
 with torch.no_grad(),EventStorage():
  losses=model(data)
 current['losses']={k:float(v) for k,v in losses.items()};records.append(current)
 (r/'manifests/backbone_key_diagnostic.json').write_text(json.dumps(records,indent=2));print('DIAGNOSTIC',adapted,current['losses'],current['reid_shapes'],flush=True)
