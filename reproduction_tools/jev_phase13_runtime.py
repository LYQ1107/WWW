"""Perception-cache backed actual native GMT loop with independent Stage2 executor."""
import torch
from jev_phase13_common import *
def build_tracker(video,policy=None,variant='full',mode='JEV_DIRECT',temperature=(1.,1.),react_learned=True):
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
    model.jev_perception_cache_reader=FrozenPerceptionCache(OUT/'stage1_cache_v1'/f'video{video:02d}')
    model.jev_stage2_executor=NativeDirectExecutor(policy,mode,temperature,variant,react_learned)
    def forbidden(*a,**kw):raise AssertionError('JEV_DIRECT may not invoke GTA scoring or activated/traj_score pipeline')
    if mode=='JEV_DIRECT':
        model.get_asso=forbidden;model.roi_heads._forward_transformer=forbidden;model.roi_heads.asso_predictor.forward=forbidden;model.roi_heads._activate_asso=forbidden;model.roi_heads.s_t_head.forward=forbidden
    return model
def cache_inputs(video):
    allowed(video);from gtr.modeling.jev_perception_cache import FrozenPerceptionCache
    reader=FrozenPerceptionCache(OUT/'stage1_cache_v1'/f'video{video:02d}');keys=sorted(reader.keys());frames=max(k[1] for k in keys)+1
    assert len(keys)==2*frames
    values=[]
    for view in range(2):
        for frame in range(frames):
            p=reader.load(video,frame,view);h,w=p['image_size'];values.append({'video_id':video,'view_num':2,'height':int(h),'width':int(w),'image':None})
    return values,frames,reader
def run(model,values,frames,stop=None,prefix=None):
    return model.sliding_inference_GMT(values,2,[0,frames-1,list(range(frames))*2],native_raw=True,native_prefix=prefix,native_stop_frame=stop)
