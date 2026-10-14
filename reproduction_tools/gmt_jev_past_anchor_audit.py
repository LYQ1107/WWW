"""Post-hoc GT-only temporal audit using certified strictly earlier-scene anchors."""
import collections,gzip,numpy as np
from gmt_jev_common import *

def load(ref):
 assert sha(ref['path'])==ref['SHA256']
 with gzip.open(ref['path'],'rt') as f:return json.load(f)

def audit(pred,labels,images):
 by_frame=collections.defaultdict(list);row_index=collections.Counter()
 for p in pred:
  image=images[p['image_id']];f,v=image['frame_id']-1,image['view_id']-1
  row=row_index[p['image_id']];row_index[p['image_id']]+=1
  by_frame[f].append((v,int(p['track_id']),labels[f,v,row]))
 anchors={};totals=collections.Counter();votes=collections.defaultdict(collections.Counter)
 counts=collections.Counter();episodes=[];active={}
 def close(key,end_reason):
  event=active.pop(key);event['termination']=end_reason
  event['right_censored']=end_reason!='OBSERVED_CERTIFIED_SAME_PERSON_REPAIR'
  event['duration_camera_frames']=event['end']-event['start']+1;episodes.append(event)
 for f,observations in sorted(by_frame.items()):
  # No current-scene GT is used to create an anchor before classification.
  current={}
  for v,t,g in observations:
   counts['observations']+=1
   if g is None:counts['UNKNOWN_GT_observations']+=1;continue
   owner=anchors.get(t);counts['known_GT_observations']+=1
   status='UNKNOWN_UNCERTIFIED_OWNER' if owner is None else 'WRONG_PERSON' if owner['person']!=g else 'SAME_PERSON'
   counts[status+'_observations']+=1
   key=(g,v);assert key not in current,'Duplicated GT person/view must be masked'
   current[key]=(status,t,owner)
  for key in list(active):
   if key not in current:close(key,'GAP_OR_UNKNOWN_GT_CENSORING')
  for key,(status,t,owner) in current.items():
   if key in active and f!=active[key]['end']+1:close(key,'NONCONSECUTIVE_OBSERVATION_CENSORING')
   if status=='WRONG_PERSON':
    if key not in active:
     active[key]=dict(person=key[0],view=key[1],start=f,end=f,
      first_native_ID=t,first_wrong_anchor_person=owner['person'],
      first_anchor_certified_after_scene=owner['after_scene'],
      certified_ID_changes_inside_episode=0,last_native_ID=t)
    else:
     event=active[key];event['end']=f;event['certified_ID_changes_inside_episode']+=t!=event['last_native_ID'];event['last_native_ID']=t
   elif key in active:
    close(key,'OBSERVED_CERTIFIED_SAME_PERSON_REPAIR' if status=='SAME_PERSON' else 'UNCERTIFIED_OWNER_CENSORING')
  for v,t,g in observations:
   totals[t]+=1
   if g is not None:votes[t][g]+=1
  for t in {t for v,t,g in observations}:
   if t not in anchors and votes[t]:
    g,n=votes[t].most_common(1)[0]
    if n>=3 and n/totals[t]>=.8:anchors[t]=dict(person=g,after_scene=f,known_support=n,total_support=totals[t])
 for key in list(active):close(key,'VIDEO_END_CENSORING')
 durations=[e['duration_camera_frames'] for e in episodes]
 counts['certified_historical_anchors']=len(anchors)
 counts['wrong_person_episodes']=len(episodes)
 counts['observed_repaired_episodes']=sum(not e['right_censored'] for e in episodes)
 counts['censored_episodes']=sum(e['right_censored'] for e in episodes)
 assert sum(counts[k] for k in ['WRONG_PERSON_observations','SAME_PERSON_observations','UNKNOWN_UNCERTIFIED_OWNER_observations'])==counts['known_GT_observations']
 assert sum(durations)==counts['WRONG_PERSON_observations']
 return dict(counts=dict(counts),wrong_person_observed_duration_camera_frames=sum(durations),
  episode_p50_p95_max_camera_frames=list(map(float,np.quantile(durations,[.5,.95,1]))) if durations else [0.,0.,0.],
  longest_20_episodes=sorted(episodes,key=lambda e:e['duration_camera_frames'],reverse=True)[:20])

def synthetic_checks():
 images={f+1:dict(frame_id=f+1,view_id=1) for f in range(8)}
 pred=[dict(image_id=f+1,track_id=9) for f in range(8)]
 labels={(f,0,0):100 if f<3 else 200 for f in range(8)}
 r=audit(pred,labels,images)
 assert r['counts']['UNKNOWN_UNCERTIFIED_OWNER_observations']==3
 assert r['wrong_person_observed_duration_camera_frames']==5
 assert r['longest_20_episodes'][0]['first_anchor_certified_after_scene']==2
 assert r['longest_20_episodes'][0]['first_wrong_anchor_person']==100
 labels[5,0,0]=None;r=audit(pred,labels,images)
 assert r['wrong_person_observed_duration_camera_frames']==4
 assert r['counts']['wrong_person_episodes']==2 and r['counts']['observed_repaired_episodes']==0
 # Disappearance of person200 followed by person100 is censoring, not repair.
 labels={(f,0,0):100 if f<3 or f>=6 else 200 for f in range(8)}
 r=audit(pred,labels,images)
 assert r['wrong_person_observed_duration_camera_frames']==3 and r['counts']['observed_repaired_episodes']==0
 # Actual repair: person200 returns to its previously certified ID20.
 pred=[];labels={}
 for f in range(8):
  pairs=[(9,100),(20,200)] if f<3 else [(9 if f<6 else 20,200)]
  for row,(t,g) in enumerate(pairs):pred.append(dict(image_id=f+1,track_id=t));labels[f,0,row]=g
 r=audit(pred,labels,images)
 assert r['wrong_person_observed_duration_camera_frames']==3 and r['counts']['observed_repaired_episodes']==1
 assert not r['longest_20_episodes'][0]['right_censored']
 return ['no_same_scene_certification','no_future_majority_relabeling','UNKNOWN_censoring','person_disappearance_is_not_repair','observed_same_person_certified_repair']

def main():
 from jev_phase7_offline import IdentityEvaluator
 import jev_phase7_offline
 assert jev_phase7_offline.ANNOTATIONS.resolve()==ANNOTATIONS.resolve()
 checks=synthetic_checks()
 original=read(REPORTS/'ORIGINAL_GMT_METRICS.json')['predictions'];base=load(original);e=IdentityEvaluator(16)
 aligned=e.align(base);labels={key:r['gt'] for key,r in aligned.items()}
 duplicates={image_id:{g for g,n in collections.Counter(int(a['instance_id']) for a in gt).items() if n>1} for image_id,gt in e.gt.items()}
 image_by_key={(int(i['frame_id'])-1,int(i['view_id'])-1):image_id for image_id,i in e.images.items()}
 masked=0
 for key,g in labels.items():
  if g in duplicates[image_by_key[key[:2]]]:labels[key]=None;masked+=1
 sources={'original':original,**{k:read(REPORTS/f'ONLINE_{k}.json')['raw_predictions'] for k in ['threshold','rule','confidence']}}
 geometry=lambda rows:[{k:v for k,v in p.items() if k!='track_id'} for p in rows]
 cases={}
 for kind,source in sources.items():
  pred=base if kind=='original' else load(source);assert geometry(pred)==geometry(base)
  cases[kind]=dict(audit(pred,labels,e.images),predictions=source)
 denominator=cases['original']['wrong_person_observed_duration_camera_frames']
 for kind,r in cases.items():r['duration_ratio_to_original']=r['wrong_person_observed_duration_camera_frames']/denominator if denominator else None
 result=dict(status='COMPLETE_POST_HOC_TEMPORAL_DIAGNOSTIC',video=16,primary_protocol='NATIVE_RAW',methods=cases,
  annotations=ref(ANNOTATIONS),duplicate_GT_person_observations_masked=masked,no_new_GMT_run=True,no_gradient_updates=True,
  anchor='Immutable first historical native-ID/person anchor, certified only after >=3 known observations and >=80% coverage; classification uses anchors from strictly earlier scene frames. Later contamination does not rewrite the historical anchor.',
  episodes='Consecutive observed GT-person/camera frames carrying an ID certified earlier to a different person. Correct certified person ends an observed repair; gaps, UNKNOWN GT/owner and video end right-censor. Unknown frames are not inferred or counted.',
  limitations=['Post-hoc diagnostic added after development metrics, not a tuning or selection criterion','Historical anchors do not establish a unique canonical identity among aliases','Observed persistence is not an intervention-causal propagation effect','Uncertified births/fragments are UNKNOWN here; TrackEval IDSW/Frag and prior fragment diagnostics remain separate','GT alignment uses unchanged IoU>=.5 Hungarian; duplicate GT persons per image are masked','Camera frames, not seconds; no confidence interval from treating correlated frames as IID'],
  synthetic_semantic_checks=dict(status='PASS',checks=checks),scientific_success=False,source=binding())
 save(REPORTS/'PAST_ANCHOR_TEMPORAL_AUDIT.json',result)
 lines=['# 仅使用历史认证的错误人物关联持续量','','这是结果后诊断，不能用于调参、方法选择或替代预注册主要指标。','',
  '| 方法 | 认证错误人物观测/持续相机帧 | 相对 GMT | 区间数 | 观测到修复 | 删失区间 | 最长相机帧 |', '|---|---:|---:|---:|---:|---:|---:|']
 for kind,r in cases.items():
  c=r['counts'];ratio=r['duration_ratio_to_original'];ratio_text=f'{ratio:.4f}' if ratio is not None else 'UNDEFINED'
  lines.append(f"| {kind} | {r['wrong_person_observed_duration_camera_frames']} | {ratio_text} | {c['wrong_person_episodes']} | {c['observed_repaired_episodes']} | {c['censored_episodes']} | {r['episode_p50_p95_max_camera_frames'][2]:.0f} |")
 lines+=['','Native ID 的首个历史人物锚需至少3次已知观测且覆盖率>=80%；当前场景仅使用更早场景已认证的锚，后续污染不会用未来多数重写它。GT只用于本离线审计，没有进入 actor。重复 GT 人物标注被屏蔽。',
  '', '连续错误关联直到观测到认证的正确人物归属才记为修复；遮挡/缺观测、未知标注、未认证新 ID 或视频结束均为右删失，不算修复，也不推断未知时段。短轨出生和碎片可能在此指标中变成 UNKNOWN，所以必须结合 IDSW、Frag、额外 ID 和已知/未知覆盖量阅读。',
  '', '此量证明的是观测到的、相对于历史锚的错误人物关联持续区间；不证明错误干预造成的因果传播时间，也不指定别名中唯一正确的 canonical ID。原有整段回顾性映射统计保留，两者不能混用。JSON 记录各项覆盖计数、最长20区间、输入SHA和实现。']
 (DOCS/'PAST_ANCHOR_TEMPORAL_AUDIT_ZH.md').write_text('\n'.join(lines)+'\n')
 print('PAST_ANCHOR_AUDIT_COMPLETE',{k:r['wrong_person_observed_duration_camera_frames'] for k,r in cases.items()},flush=True)

if __name__=='__main__':main()
