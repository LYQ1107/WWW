"""Two-view real external prefix format and fresh Stage1-only perception cache."""
import argparse
import collections
import time
import torch
from PIL import Image
from jev_phase14_common import *
from jev_phase13_common import STAGE1,STAGE1_SHA
from cache_jev_phase13_stage1 import build_frontend,extract
from gtr.modeling.jev_perception_cache import FrozenPerceptionCacheWriter,FrozenPerceptionCache

SCENE=100001
DATA=OUT/'external_wildtrack_v1'
EXTERNAL=OUT/'external_eval_v1'

def prepare():
    protect();result=read(DATA/'RESULT.json');assert result['status']=='DATA_READY'
    EXTERNAL.mkdir(parents=True,exist_ok=True);protocol=read(REPORTS/'EXTERNAL_EVALUATION_FORMAT.json')
    images={};anns=[];clipped=[];counts=collections.Counter();manifest=[]
    annotations=sorted([m for m in result['manifests'] if '/annotations_positions/' in m['archive_member']],key=lambda m:m['archive_member']);assert len(annotations)==320
    members={m['archive_member']:m for m in result['manifests']}
    for frame,record in enumerate(annotations):
        assert sha(record['path'])==record['SHA256'];stamp=Path(record['path']).stem;original=read(record['path']);manifest.append(record)
        for view,camera in enumerate(['C1','C2']):
            item=next(m for m in result['manifests'] if f'/Image_subsets/{camera}/' in m['archive_member'] and Path(m['archive_member']).stem==stamp)
            assert sha(item['path'])==item['SHA256']
            with Image.open(item['path']) as image:assert image.size==(1920,1080)
            image_id=2*frame+view+1
            images[frame,view]=dict(id=image_id,frame_id=frame+1,view_id=view+1,video_id=SCENE,height=1080,width=1920,file_name=item['path'],source_stamp=int(stamp),image_SHA256=item['SHA256']);manifest.append(item)
            seen=set()
            for person in original:
                projections=[v for v in person['views'] if v['viewNum']==view];assert len(projections)==1
                b=projections[0];box=[b['xmin'],b['ymin'],b['xmax'],b['ymax']]
                if -1 in box:counts['not_visible_sentinel']+=1;continue
                x,y,x2,y2=box
                if x2<=x or y2<=y or min(x2,1920)<=max(x,0) or min(y2,1080)<=max(y,0):counts['invalid_or_outside']+=1;continue
                gt=int(person['personID']);assert gt not in seen;seen.add(gt)
                a=dict(id=len(anns)+1,image_id=image_id,instance_id=gt,bbox=[x,y,x2-x,y2-y],category_id=1,conf=1.,iscrowd=0,view_id=view+1,area=(x2-x)*(y2-y));anns.append(a)
                bx,by,bx2,by2=max(x,0),max(y,0),min(x2,1920),min(y2,1080)
                clipped.append(dict(a,bbox=[bx,by,bx2-bx,by2-by],area=(bx2-bx)*(by2-by)));counts['included']+=1;counts['partially_outside_retained']+=int(box!=[bx,by,bx2,by2])
    value=dict(images=list(images.values()),annotations=anns,categories=[dict(id=1,name='person')],videos=[dict(id=SCENE,file_name='WILDTRACK',view_num=2)])
    save(EXTERNAL/'ANNOTATIONS_RAW.json',value);save(EXTERNAL/'ANNOTATIONS_CLIPPED.json',dict(value,annotations=clipped))
    save(EXTERNAL/'DATA_MANIFEST.json',dict(status='COMPLETE',binding=binding(),format_protocol_SHA256=sha(REPORTS/'EXTERNAL_EVALUATION_FORMAT.json'),download_result_SHA256=sha(DATA/'RESULT.json'),counts=dict(counts),members=manifest,annotations=[dict(path=str(EXTERNAL/p),SHA256=sha(EXTERNAL/p)) for p in ['ANNOTATIONS_RAW.json','ANNOTATIONS_CLIPPED.json']],GT_actor_inputs=False,model_selection=False))
    print('PHASE14_EXTERNAL_REAL_FORMAT_READY',dict(counts),flush=True)

def inputs():
    annotation=read(EXTERNAL/'ANNOTATIONS_RAW.json');images={(i['frame_id']-1,i['view_id']-1):i for i in annotation['images']};values=[]
    for view in range(2):
        for frame in range(320):
            image=images[frame,view];values.append(dict(video_id=SCENE,view_num=2,height=1080,width=1920,image=None,phase13_image_path=image['file_name']))
    return values,320,images

def cache():
    protect();assert sha(STAGE1)==STAGE1_SHA;manifest=read(EXTERNAL/'DATA_MANIFEST.json');assert manifest['status']=='COMPLETE';torch.set_num_threads(1);torch.manual_seed(20261009);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    source=binding();model,cfg=build_frontend();out=EXTERNAL/'stage1_cache_v1';out.mkdir(parents=True,exist_ok=True);assert not (out/'RESULT.json').exists()
    writer=FrozenPerceptionCacheWriter(out);values,frames,images=inputs();begin=time.monotonic();counts=collections.Counter()
    for frame in range(frames):
        for view in range(2):
            if (SCENE,frame,view) in writer._seen:continue
            image=images[frame,view];assert sha(image['file_name'])==image['image_SHA256']
            with torch.no_grad():instance=extract(model,cfg,image['file_name'])
            writer.write(video_id=SCENE,frame=frame,view=view,instances=instance,metadata=dict(Stage1_SHA256=STAGE1_SHA,image_id=image['id'],file_name=image['file_name'],source_image_SHA256=image['image_SHA256'],GT=False,GTA=False,RPCE=False));writer.handle.flush();counts['detections']+=len(instance)
        if (frame+1)%32==0:
            save(out/'PROGRESS.json',dict(status='RUNNING',scene_frames=frame+1,total=320,seconds=time.monotonic()-begin));print('PHASE14_EXTERNAL_ACTUAL_FRONTEND',frame+1,320,flush=True)
    writer.handle.close();reader=FrozenPerceptionCache(out);assert len(reader.keys())==640
    save(out/'RESULT.json',dict(status='COMPLETE',binding=source,Stage1=dict(path=str(STAGE1),SHA256=STAGE1_SHA),config=dict(path=str(ROOT/'configs/VISION_stage1.yaml'),SHA256=sha(ROOT/'configs/VISION_stage1.yaml')),index_SHA256=sha(out/'index.jsonl'),payloads=640,counts_this_attempt=dict(counts),data_manifest_SHA256=sha(EXTERNAL/'DATA_MANIFEST.json'),GTA_RPCE_throw=True,GT_actor_inputs=False,seconds=time.monotonic()-begin))
    print('PHASE14_EXTERNAL_FROZEN_FRONTEND_COMPLETE',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['prepare','cache'],required=True);a=p.parse_args();prepare() if a.mode=='prepare' else cache()
