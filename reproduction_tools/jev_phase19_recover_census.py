"""Recover completed native trajectories from exact commits, not another GPU replay."""
from jev_phase19_common import *
import argparse,gzip,collections,numpy as np
from jev_phase19_runtime import cache_inputs
from jev_phase19_eval import prepare
from jev_phase16_memory_risk import MemoryRisk
def rows(path):
 with gzip.open(path,'rt') as f:return [json.loads(line) for line in f]
def main(video):
 allowed(video);protect();source=OUT/'census_v1'/f'video{video:02d}';target=OUT/'census_recovered_v1'/f'video{video:02d}';target.mkdir(parents=True,exist_ok=True);assert not (target/'RESULT.json').exists()
 assert (source/'OWNER_TRACE.jsonl.gz').exists(),'incomplete native trajectory is never recovered'
 values,frames,reader=cache_inputs(video);commits=rows(source/'COMMITS.jsonl.gz');events=rows(source/'EVENTS.jsonl.gz');trace=rows(source/'OWNER_TRACE.jsonl.gz');queries=rows(source/'QUERIES.jsonl.gz');ids={tuple(e['key']):e['ids'] for e in commits}
 all_keys={(f,v) for f in range(frames) for v in range(2)};missing=all_keys-set(ids)
 counts0=[len(reader.load(video,0,v)['pred_boxes']) for v in range(2)];bootstrap=int(np.argsort(counts0)[::-1][0])
 assert missing=={(0,bootstrap)},('not a completed actual native trajectory',missing)
 # This exact source-defined initialization was executed but has no commit observer.
 ids[0,bootstrap]=list(range(1,counts0[bootstrap]+1))
 first=next(q for q in queries if q['task']==0 and q['key'][0]==0);assert first['refs']==ids[0,bootstrap]
 assert len(events)==len(trace)==sum(len(e['ids']) for e in commits)
 for e in events:assert ids[tuple(e['event_key'][1:3])][e['event_key'][3]]==e['actual_committed_ref']
 images={(i['frame_id']-1,i['view_id']-1):i for i in read(ANNOTATIONS)['images'] if i['video_id']==video};predictions=[]
 for f,v in sorted(all_keys):
  p=reader.load(video,f,v);im=images[f,v];h,w=p['image_size'];sx=im['width']/w;sy=im['height']/h
  assert len(p['pred_boxes'])==len(ids[f,v]) and len(ids[f,v])==len(set(ids[f,v]))
  for box,t,score in zip(p['pred_boxes'].tolist(),ids[f,v],p['detection_scores'].tolist()):
   x,y,x2,y2=box;predictions.append(dict(image_id=im['id'],track_id=int(t),bbox=[x*sx,y*sy,max(0.,(x2-x)*sx),max(0.,(y2-y)*sy)],score=float(score),category_id=1))
 path=target/'RAW_PREDICTIONS.json';save(path,predictions);evaluation=prepare(target,video,path)
 c=collections.Counter(e['category'] for e in events);c['events']=len(events);c['legal_K_gt16']=sum(sum(e['legal'])>16 for e in events)
 risk=MemoryRisk.__new__(MemoryRisk);risk.strong_trace=trace
 training=read('/home/liuyeqiang/WWW_jev_phase15_runtime/20261010_v1/training_full_payload_v3/F_full/seed20261009/pilot/RESULT.json')
 result=dict(status='COMPLETE',video=video,frames=frames,counts=dict(c),queries=ref(source/'QUERIES.jsonl.gz'),commits=ref(source/'COMMITS.jsonl.gz'),events=ref(source/'EVENTS.jsonl.gz'),owner_trace=ref(source/'OWNER_TRACE.jsonl.gz'),raw_predictions=ref(path),strict_evaluation=evaluation,owner_propagation=risk.horizon((0,0),frames),GT_actor_inputs=False,actual_native_mutated_state=True,recovery_binding=binding(),binding=dict(source_commit='07ac1d5958c21cd35b50660edd551033cc41fc9b',actor_root=str(OUT/'source_census_v1'),checkpoint=training['checkpoint'],new_training=False),recovery=dict(status='EXACT_RECORDED_NATIVE_COMMITS_PLUS_SOURCE_DEFINED_UNOBSERVED_BOOTSTRAP',source_NATIVE_forward_complete=True,all_observed_rows_IDs_exact=True,bootstrap=dict(key=[0,bootstrap],IDs=ids[0,bootstrap],source=ref(ROOT/'gtr/modeling/meta_arch/gtr_rcnn.py'),largest_camera_sort='np.argsort(det_counts).tolist()[::-1]',first_query_refs_verified=True),cached_boxes_and_scores_unmodified=True,GT_not_used_to_create_predictions=True,no_GPU_replay=True,original_failure_preserved=True))
 save(target/'RESULT.json',result);print('RECOVERED_NATIVE_CENSUS',video,len(events),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--video',type=int,required=True);main(p.parse_args().video)

