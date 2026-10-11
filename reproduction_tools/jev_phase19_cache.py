"""Same frozen Stage1 detector/VFCE; immutable records, observable bounded writes."""
from jev_phase19_common import *
import argparse,random,numpy as np,torch
from cache_jev_phase13_stage1 import build_frontend,extract
from gtr.modeling.jev_perception_cache import FrozenPerceptionCacheWriter

def main(video):
 allowed(video);protect();assert sha(STAGE1)==STAGE1_SHA
 torch.set_num_threads(1);torch.manual_seed(20261009);np.random.seed(20261009);random.seed(20261009)
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 records=[i for i in read(ANNOTATIONS)['images'] if i['video_id']==video];records.sort(key=lambda i:(i['frame_id'],i['view_id']))
 model,cfg=build_frontend();folder=OUT/'stage1_cache_v1'/f'video{video:02d}';folder.mkdir(parents=True,exist_ok=True);assert not (folder/'RESULT.json').exists()
 writer=FrozenPerceptionCacheWriter(folder);existing=sum(p.stat().st_size for p in folder.rglob('*') if p.is_file());cap=(1024 if video==15 else 512)*2**20;times=[];detections=0;start=time.monotonic()
 for n,i in enumerate(records):
  key=(video,i['frame_id']-1,i['view_id']-1)
  if key in writer._seen:continue
  torch.cuda.synchronize();t=time.perf_counter()
  with torch.no_grad():inst=extract(model,cfg,IMAGES/i['file_name'])
  torch.cuda.synchronize();times.append((time.perf_counter()-t)*1000)
  estimate=(1024*4+64)*len(inst)+32768
  if existing+estimate>cap or shutil.disk_usage(OUT).free<30*2**30+estimate:
   writer.close();save(folder/'RESOURCE_STOP.json',dict(status='RESOURCE_STOP',video=video,retained_bytes=existing,cap_bytes=cap,next_payload=key,estimated_next_bytes=estimate,semantic_truncation=False));raise RuntimeError('authorized cache reservation exhausted; retained partial records, no altered detections')
  record=writer.write(video_id=video,frame=key[1],view=key[2],instances=inst,metadata=dict(file_name=i['file_name'],Stage1_SHA256=STAGE1_SHA,image_id=i['id'],feature_dim=1024,GTA=False,RPCE=False))
  existing+=(folder/record['record']).stat().st_size+len(json.dumps(record))+1;detections+=len(inst)
  if (n+1)%64==0:
   save(folder/'PROGRESS.json',dict(status='RUNNING',video=video,payloads=len(writer._seen),total=len(records),last_key=key,pid=os.getpid(),seconds=time.monotonic()-start,bytes=existing));print('PHASE19_CACHE_PROGRESS',video,n+1,len(records),flush=True)
 writer.close();assert len(writer._seen)==len(records);protect()
 save(folder/'RESULT.json',dict(status='COMPLETE',binding=binding(),video=video,payloads=len(records),cache_index=ref(folder/'index.jsonl'),feature_dim=1024,Stage1_SHA256=STAGE1_SHA,extractor=ref(ROOT/'reproduction_tools/cache_jev_phase13_stage1.py'),same_frozen_detector_VFCE=True,GT_actor_input=False,GTA=False,RPCE=False,seconds=time.monotonic()-start,bytes=existing,detector_rows_this_attempt=detections,latency_camera_payload_ms=dict(count=len(times),p50=float(np.percentile(times,50)),p95=float(np.percentile(times,95))),scope='fresh frozen perception extraction only; no tracking gain or total online FPS claim'))
 save(folder/'PROGRESS.json',dict(status='COMPLETE',payloads=len(records),bytes=existing));print('PHASE19_CACHE_COMPLETE',video,len(records),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--video',type=int,required=True);main(p.parse_args().video)

