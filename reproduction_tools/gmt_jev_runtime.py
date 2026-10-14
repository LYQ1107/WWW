"""Complete frozen Stage2 GMT, original source snapshot and actual image IO."""
import gc,importlib.util,random,sys,torch,numpy as np
from gmt_jev_common import *
for p in [ROOT,ROOT/'third_party/CenterNet2']:sys.path.insert(0,str(p))
def seed():
 random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
def metadata(video):
 assert video in [12,13,14,16],video
 images={(i['frame_id']-1,i['view_id']-1):i for i in read(ANNOTATIONS)['images'] if i['video_id']==video}
 frames=max(f for f,v in images)+1;assert len(images)==2*frames
 values=[dict(video_id=video,view_num=2,height=int(images[f,v]['height']),width=int(images[f,v]['width']),image=None,phase13_image_path=str(IMAGES/images[f,v]['file_name'])) for v in range(2) for f in range(frames)]
 return values,frames,images
def original_class():
 import gtr
 from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
 path=OUT/'source'/('original_gtr_rcnn_'+BASE[:8]+'.py');path.parent.mkdir(parents=True,exist_ok=True)
 source=subprocess.check_output(['git','show',BASE+':gtr/modeling/meta_arch/gtr_rcnn.py'],cwd=ROOT)
 if path.exists():assert path.read_bytes()==source
 else:path.write_bytes(source)
 name='gtr.modeling.meta_arch.gmt_jev_original_snapshot';spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m
 old=META_ARCH_REGISTRY._obj_map.pop('GTRRCNN')
 try:spec.loader.exec_module(m);cls=m.GTRRCNN
 finally:META_ARCH_REGISTRY._obj_map['GTRRCNN']=old
 return cls,path
def model(original=False):
 import gtr
 from detectron2.config import get_cfg
 from detectron2.modeling import build_model
 from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
 from centernet.config import add_centernet_config
 from gtr.config import add_gtr_config
 cfg=get_cfg();add_centernet_config(cfg);add_gtr_config(cfg);cfg.merge_from_file(str(ROOT/'configs/VISION_test.yaml'));cfg.defrost()
 cfg.MODEL.JEV.ENABLED=False;cfg.MODEL.JEV.CANDIDATE_ENABLED=False;cfg.MODEL.JEV.VISUAL_JEV_ENABLED=False
 cfg.MODEL.WEIGHTS=str(STAGE2);cfg.freeze();assert sha(STAGE2)==STAGE2_SHA
 cls,path=original_class() if original else (None,None)
 old=META_ARCH_REGISTRY._obj_map['GTRRCNN']
 if cls:META_ARCH_REGISTRY._obj_map['GTRRCNN']=cls
 try:m=build_model(cfg).cuda().eval()
 finally:META_ARCH_REGISTRY._obj_map['GTRRCNN']=old
 m.load_state_dict(torch.load(STAGE2,map_location='cpu')['model'],strict=True)
 assert m.roi_heads.concat_emb and m.roi_heads.cat_dim==128
 assert m.jev_policy is None and m.jev_candidate_policy is None
 assert getattr(m,'jev_stage2_executor',None) is None and m.jev_perception_cache_reader is None
 for p in m.parameters():p.requires_grad_(False)
 from PIL import Image
 from gtr.data.transforms.custom_augmentation_impl import EfficientDetResizeCrop
 infer=m.inference
 def live(payloads,*args,**kwargs):
  ready=[]
  for p in payloads:
   rgb=np.array(Image.open(p['phase13_image_path']).convert('RGB'))
   rgb=EfficientDetResizeCrop(cfg.INPUT.TEST_SIZE,(1.,1.)).get_transform(rgb).apply_image(rgb)
   ready.append(dict(p,image=torch.as_tensor(np.ascontiguousarray(rgb.transpose(2,0,1)))))
  return infer(ready,*args,**kwargs)
 m.inference=live
 return m,cfg,path
def run(m,values,frames,stop=None):
 return m.sliding_inference_GMT(values,2,[0,frames-1,list(range(frames))*2],native_raw=True,native_stop_frame=stop)
