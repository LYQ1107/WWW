"""Fresh detector and VFCE-only Stage1 cache, no GTA/RPCE, streaming resumable records."""
import argparse,time,random
import numpy as np,torch
from PIL import Image
from jev_phase13_common import *

def build_frontend():
    from detectron2.config import get_cfg
    from detectron2.modeling import build_model
    from centernet.config import add_centernet_config
    from gtr.config import add_gtr_config
    cfg=get_cfg();add_centernet_config(cfg);add_gtr_config(cfg);cfg.merge_from_file(str(ROOT/'configs/VISION_stage1.yaml'));cfg.defrost();cfg.MODEL.DEVICE='cuda';cfg.MODEL.WEIGHTS=str(STAGE1);cfg.freeze()
    model=build_model(cfg).cuda().eval();checkpoint=torch.load(STAGE1,map_location='cpu');model.load_state_dict(checkpoint['model'],strict=True)
    for p in model.parameters():p.requires_grad_(False)
    def forbidden(*a,**kw):raise AssertionError('GTA/RPCE forbidden in true Stage1 frontend')
    model.roi_heads._forward_transformer=forbidden;model.roi_heads.asso_predictor.forward=forbidden;model.roi_heads.s_t_head.forward=forbidden
    return model,cfg

def extract(model,cfg,image):
    from detectron2.structures import Instances
    from gtr.data.transforms.custom_augmentation_impl import EfficientDetResizeCrop
    rgb=np.array(Image.open(image).convert('RGB'));rgb=EfficientDetResizeCrop(cfg.INPUT.TEST_SIZE,(1.,1.)).get_transform(rgb).apply_image(rgb)
    tensor=torch.as_tensor(np.ascontiguousarray(rgb.transpose(2,0,1)));images=model.preprocess_image([{'image':tensor}])
    features=model.backbone(images.tensor);proposals,_=model.proposal_generator(images,features,None)
    proposals=[p[p.objectness_logits>model.roi_heads.asso_thresh_test] for p in proposals]
    pooled=model.roi_heads.asso_pooler([features[f] for f in model.roi_heads.asso_in_features],[p.proposal_boxes for p in proposals]);vfce=model.roi_heads.asso_head(pooled)
    p=proposals[0];inst=Instances(p.image_size);inst.pred_boxes=p.proposal_boxes;inst.scores=p.objectness_logits;inst.pred_classes=torch.zeros(len(p),dtype=torch.long,device=model.device);inst.reid_features=vfce
    assert vfce.shape==(len(p),1024) and torch.isfinite(vfce).all()
    return inst

def main(video,limit):
    allowed(video);protect();torch.set_num_threads(1);torch.manual_seed(20261009);np.random.seed(20261009);random.seed(20261009)
    model,cfg=build_frontend();from gtr.modeling.jev_perception_cache import FrozenPerceptionCacheWriter
    records=[i for i in json.loads(ANNOTATIONS.read_text())['images'] if i['video_id']==video];records.sort(key=lambda i:(i['frame_id'],i['view_id']))
    if limit is not None:records=[i for i in records if i['frame_id']<=limit]
    out=OUT/('stage1_smoke_v1' if limit is not None else 'stage1_cache_v1')/f'video{video:02d}';out.mkdir(parents=True,exist_ok=True)
    assert not (out/'RESULT.json').exists(),'completed cache immutable';writer=FrozenPerceptionCacheWriter(out);times=[];start=time.monotonic();detections=0
    for i,image in enumerate(records):
        key=(video,image['frame_id']-1,image['view_id']-1)
        if key in writer._seen:continue
        torch.cuda.synchronize();t=time.perf_counter()
        with torch.no_grad():inst=extract(model,cfg,IMAGES/image['file_name'])
        torch.cuda.synchronize();times.append((time.perf_counter()-t)*1000);detections+=len(inst)
        writer.write(video_id=video,frame=key[1],view=key[2],instances=inst,metadata={'file_name':image['file_name'],'Stage1_SHA256':STAGE1_SHA,'image_id':image['id'],'feature_dim':1024,'GTA':False,'RPCE':False})
        writer.handle.flush()
        if (i+1)%64==0:save(out/'PROGRESS.json',{'status':'RUNNING','payloads':len(writer._seen),'total':len(records),'last_key':key,'elapsed_seconds':time.monotonic()-start});print('STAGE1_CACHE',video,i+1,len(records),flush=True)
    writer.handle.close();r={'status':'COMPLETE','binding':binding(),'video':video,'payloads':len(records),'cache_index_SHA256':sha(out/'index.jsonl'),'feature_dim':1024,'live_GTA_and_RPCE_throw_mock':True,'detections_this_attempt':detections,'latency_this_attempt_ms':{'count':len(times),'p50':float(np.percentile(times,50)) if times else None,'p95':float(np.percentile(times,95)) if times else None,'sum':sum(times)},'elapsed_seconds':time.monotonic()-start,'cache_path':str(out),'scope':'real Stage1 detector + same VFCE ROI pooling/head as REID training, no eval jitter or untrained RPCE. Per-image preprocessing/file I/O included in measured frontend time; camera payloads independent. Resumed earlier chunks retained.'};save(out/'RESULT.json',r);print('STAGE1_CACHE_COMPLETE',video,len(records),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--video',type=int,required=True);p.add_argument('--frames',type=int);a=p.parse_args();main(a.video,a.frames)
