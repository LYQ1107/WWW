"""Phase XIII native semantics with explicit authorized Phase XIX cache roots."""
from jev_phase19_common import *
import torch
from gtr.modeling.jev_perception_cache import FrozenPerceptionCache
class DimensionPreservingCache(FrozenPerceptionCache):
 """Writer loses the feature width for zero rows; restore the known empty schema."""
 def load(self,*args):
  p=super().load(*args)
  if p['reid_features'].shape==(0,0):
   assert p['pred_boxes'].shape[0]==0 and p['metadata']['feature_dim']==1024
   p=dict(p,reid_features=torch.empty((0,1024),dtype=p['reid_features'].dtype))
  return p
def build_tracker(video,policy):
 allowed(video)
 from detectron2.config import get_cfg
 from detectron2.modeling import build_model
 from centernet.config import add_centernet_config
 from gtr.config import add_gtr_config
 from gtr.modeling.jev_perception_cache import FrozenPerceptionCache
 from gtr.modeling.jev_runtime import JEVRuntimePolicy
 from gtr.modeling.jev_stage2.native import NativeDirectExecutor
 cfg=get_cfg();add_centernet_config(cfg);add_gtr_config(cfg);cfg.merge_from_file(str(ROOT/'configs/VISION_test.yaml'));cfg.defrost();cfg.MODEL.JEV.ENABLED=False;cfg.freeze()
 model=build_model(cfg).cuda().eval();model.load_state_dict(torch.load(STAGE1,map_location='cpu')['model'],strict=True)
 for p in model.parameters():p.requires_grad_(False)
 model.jev_enabled=True;model.jev_mode='off';model.jev_policy=JEVRuntimePolicy('off',None);model.jev_candidate_policy=None;model.visual_jev_enabled=False
 model.jev_perception_cache_reader=DimensionPreservingCache(OUT/'stage1_cache_v1'/f'video{video:02d}')
 model.jev_stage2_executor=NativeDirectExecutor(policy,'JEV_DIRECT',(1.,1.),'full',False)
 def forbidden(*a,**kw):raise AssertionError('frozen JEV_DIRECT forbids GTA/RPCE scoring')
 model.get_asso=forbidden;model.roi_heads._forward_transformer=forbidden;model.roi_heads.asso_predictor.forward=forbidden;model.roi_heads._activate_asso=forbidden;model.roi_heads.s_t_head.forward=forbidden
 return model
def cache_inputs(video):
 allowed(video)
 from gtr.modeling.jev_perception_cache import FrozenPerceptionCache
 reader=DimensionPreservingCache(OUT/'stage1_cache_v1'/f'video{video:02d}');cache=read(reader.root/'RESULT.json');assert cache['status']=='COMPLETE' and cache['Stage1_SHA256']==STAGE1_SHA
 keys=sorted(reader.keys());frames=max(k[1] for k in keys)+1;assert set(keys)=={(video,f,v) for f in range(frames) for v in range(2)}
 metadata={(i['frame_id']-1,i['view_id']-1):i for i in read(ANNOTATIONS)['images'] if i['video_id']==video}
 values=[dict(video_id=video,view_num=2,height=int(metadata[f,v]['height']),width=int(metadata[f,v]['width']),image=None,phase13_image_path=str(IMAGES/metadata[f,v]['file_name'])) for v in range(2) for f in range(frames)]
 return values,frames,reader
def run(model,values,frames,stop=None,prefix=None):
 return model.sliding_inference_GMT(values,2,[0,frames-1,list(range(frames))*2],native_raw=True,native_prefix=prefix,native_stop_frame=stop)

