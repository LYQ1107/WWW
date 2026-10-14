"""Full-video fixed GT-person mapping time diagnostics, separate from certificates."""
import collections,numpy as np,torch
from jev_fastcheck_common import *
def main():
 from jev_phase7_offline import IdentityEvaluator
 torch.set_num_threads(1);evaluator=IdentityEvaluator(16);sources=[]
 for p in sorted(REPORTS.glob('ONLINE_*_STEPS1000.json')):
  d=read(p);sources.append((d['tag'],d['raw_predictions'],ref(p)))
 for d in read(REPORTS/'FROZEN_TRAIN16_REFERENCES.json')['cases']:
  sources.append(('reference_'+d['arm'],d['raw_predictions'],d['native_tracking_source']))
 cases=[]
 for tag,cp,source in sources:
  assert sha(cp['path'])==cp['SHA256'];rows=evaluator.align(read(cp['path']));mapping,counts=evaluator.mapping(rows)
  timelines=collections.defaultdict(dict);summary=collections.Counter()
  for (f,v,row),r in sorted(rows.items()):
   g=r['gt']
   if g is None:summary['UNKNOWN_GT_observations']+=1;continue
   unmapped=r['id'] not in mapping;bad=mapping.get(r['id'])!=g
   summary['known_GT_observations']+=1;summary['identity_inconsistent_observations']+=bad;summary['unmapped_fragment_observations']+=unmapped;summary['mapped_wrong_person_observations']+=bad and not unmapped
   timelines[g,v][f]=bad or timelines[g,v].get(f,False)
  episodes=[]
  for (person,camera),timeline in sorted(timelines.items()):
   first=None;last=None
   for f,bad in sorted(timeline.items()):
    if first is not None and (not bad or f!=last+1):episodes.append(dict(person=person,camera=camera,start=first,end=last,duration_camera_frames=last-first+1,right_censored=f!=last+1));first=None
    if bad and first is None:first=f
    last=f
   if first is not None:episodes.append(dict(person=person,camera=camera,start=first,end=last,duration_camera_frames=last-first+1,right_censored=True))
  lengths=[r['duration_camera_frames'] for r in episodes]
  case=dict(tag=tag,counts=dict(summary),full_video_identity_inconsistency_duration_camera_frames=sum(lengths),duration_p50_p95_max=list(map(float,np.quantile(lengths,[.5,.95,1]))) if lengths else [0.,0.,0.],episodes=episodes,source=source,predictions=cp)
  cases.append(case);print('GLOBAL_ID_DURATION',tag,sum(lengths),summary['mapped_wrong_person_observations'],summary['unmapped_fragment_observations'],flush=True)
 save(REPORTS/'GLOBAL_ID_ERROR_DURATION.json',dict(status='COMPLETE_FROZEN_OFFLINE_GLOBAL_MAP_DIAGNOSTIC',cases=cases,
  definition='IoU>=.5 per-image matching with the existing IdentityEvaluator; one maximum-count one-to-one native-ID/person map pooled over the complete two-camera video. Error includes wrong mapped person and unmapped fragments, reported separately. Consecutive observed person/camera error frames form episodes; absent observations censor.',
  GT_used_only_after_tracking=True,full_video_mapping_not_online_birth_owner=True,causal_propagation_not_claimed=True,certified_past_owner_duration_separate=True,
  official_HOTA_CVIDF1_evaluator_unchanged=True,scope='descriptive identity-consistency time, not independent validation or a replacement for official metrics'))
if __name__=='__main__':main()
