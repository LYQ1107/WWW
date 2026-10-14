"""Actual original GMT states; unique first-past certificates, offline GT only."""
import argparse,collections,torch,numpy as np
from scipy.optimize import linear_sum_assignment
from gmt_jev_runtime import *

class OfflineCertificates:
 def __init__(self,video,images):
  self.images=images;self.gt=collections.defaultdict(list);ids={i['id'] for i in images.values()}
  for a in read(ANNOTATIONS)['annotations']:
   if a['image_id'] in ids and a.get('conf',1)>0 and a.get('category_id',1)==1:self.gt[a['image_id']].append(a)
  self.total=collections.Counter();self.votes=collections.defaultdict(collections.Counter);self.owner={}
 def labels(self,inst,frame,view):
  from audit_jev_phase5_decisions import overlaps,xyxy
  image=self.images[frame,view];gt=self.gt[image['id']];duplicates={g for g,n in collections.Counter(a['instance_id'] for a in gt).items() if n>1}
  h,w=inst.image_size;boxes=inst.pred_boxes.tensor.detach().cpu().numpy().copy();boxes[:,[0,2]]*=image['width']/w;boxes[:,[1,3]]*=image['height']/h
  labels=[None]*len(inst)
  if len(inst) and gt:
   iou=overlaps(boxes,np.array([xyxy(a['bbox']) for a in gt]));rr,cc=linear_sum_assignment(-((iou>=.5)*1000+iou))
   for r,c in zip(rr,cc):
    g=int(gt[c]['instance_id'])
    if iou[r,c]>=.5 and g not in duplicates:labels[int(r)]=g
  return labels
 def update(self,inst,frame,view):
  for t,g in zip(inst.track_ids.tolist(),self.labels(inst,frame,view)):
   self.total[t]+=1
   if g is not None:self.votes[t][g]+=1
   if t not in self.owner and self.votes[t]:
    g,n=self.votes[t].most_common(1)[0]
    if n>=3 and n/self.total[t]>=.8:self.owner[t]=g
 def clean(self,t):
  g=self.owner.get(int(t))
  return g if g is not None and self.votes[int(t)][g]/self.total[int(t)]>=.8 else None
 def targets(self,labels,refs):
  compatible=collections.defaultdict(list)
  for t in self.owner:
   g=self.clean(t)
   if g is not None:compatible[g].append(t)
  targets=[];categories=[]
  for g in labels:
   ids=compatible.get(g,[]) if g is not None else []
   if g is None:category='UNKNOWN_CURRENT';target=-1
   elif len(ids)>1:category='UNKNOWN_ALIASED_NATIVE_IDS';target=-1
   elif not ids:category='NO_CERTIFIED_UNIQUE_ID';target=-1
   elif ids[0] not in refs:category='UNIQUE_ID_NOT_ACTIVE';target=-1
   else:category='CERTIFIED_UNIQUE_ACTIVE';target=refs.index(ids[0])
   targets.append(target);categories.append(category)
  return targets,categories

def native_assignment(scores,lengths,threshold=.1):
 q,k=scores.shape;out=[-1]*q
 if q and k:
  rr,cc=linear_sum_assignment(-scores.detach().cpu().numpy())
  for r,c in zip(rr,cc):
   if scores[r,c]>threshold*lengths[c]:out[int(r)]=int(c)
 return out

def collect(video,kind='observe',frames=256,records=True,report=True):
 from gtr.modeling.gmt_jev import IdentityCorrection
 values,total,images=metadata(video);frames=min(frames,total);seed();m,cfg,_=model();actor=IdentityCorrection(kind).cuda().eval();m.gmt_identity_correction=actor;seed()
 labeler=OfflineCertificates(video,images);samples=[];stats=collections.Counter();groups=set();pending={};boot=[False];last=[0.];begin=time.monotonic()
 def before_native(**d):
  if d['first'] and not boot[0]:
   other=1-d['view'];inst=d['instances'][other]
   labeler.update(inst,0,other);actor.committed(inst,other,0);boot[0]=True
 def observed(packet,current,galleries):
  f,v=packet['frame'],packet['view'];gt=labeler.labels(current,f,v);refs=packet['ids'].tolist();y,categories=labeler.targets(gt,refs)
  stats.update(categories);stats['rows']+=len(gt)
  for g in gt:
   if g is not None:stats['known_current_GT']+=1
  sample=dict(key=[video,f,v],features=packet['features'].detach().cpu().clone(),eligible=packet['eligible'].detach().cpu().clone(),raw_scores=packet['base_scores'].detach().cpu().clone(),lengths=packet['lengths'].detach().cpu().clone(),refs=refs,
   targets=torch.tensor(y),categories=categories,GT_labels_OFFLINE_ONLY=gt,owners={t:labeler.clean(t) for t in refs},audit_block=(f//64)%5==4)
  pending[f,v]=sample
 actor.observer=observed;m.jev_native_prefix_observer=before_native
 def after(**d):
  f,v=d['frame'],d['view'];inst=d['instances'][-1];ids=inst.track_ids.tolist();assert len(ids)==len(set(ids))
  s=pending.pop((f,v));s['native_ids']=ids;s['baseline_accepted_cols']=native_assignment(s['raw_scores'],s['lengths'],m.overlap_thresh)
  x=s['features'];eligible=s['eligible'];rule=s['raw_scores']+.05*eligible*(x[...,13]/.2).clamp(-1,1)*s['lengths'][None]
  alternative=native_assignment(rule,s['lengths'],m.overlap_thresh);s['GMT_current_correct']=torch.zeros(len(ids),dtype=torch.bool)
  for row,(t,target,g,alt) in enumerate(zip(ids,s['targets'].tolist(),s['GT_labels_OFFLINE_ONLY'],alternative)):
   if target<0:continue
   target_id=s['refs'][target];base_correct=t==target_id;s['GMT_current_correct'][row]=base_correct
   stats['unique_active_rows']+=1;stats['GMT_correct_unique_active']+=base_correct;stats['GMT_wrong_unique_active']+=not base_correct
   if not base_correct:groups.add((video,g,target_id,v,f//64))
   other_support=float(x[row,target,10])>0;stats['wrong_with_other_camera_history']+=not base_correct and other_support
   new=s['refs'][alt] if alt>=0 else None
   if new!=t:
    stats['rule_changed_unique_rows']+=1
    if alt<0:stats['rule_reject_transition_UNKNOWN_bank_outcome']+=1
    elif new==target_id and not base_correct:stats['rule_local_N01']+=1
    elif base_correct and s['owners'].get(new) is not None and s['owners'][new]!=g:stats['rule_local_N10']+=1
    else:stats['rule_changed_UNKNOWN_or_fragment_semantics']+=1
  # Oracle is a same-state assignment bound, not an executed tracking method.
  oracle=s['raw_scores'].clone()
  for row,target in enumerate(s['targets'].tolist()):
   if target>=0:oracle[row,target]+=1e6
  oracle_choice=native_assignment(oracle,s['lengths'],m.overlap_thresh)
  stats['ORACLE_correct_unique_rows']+=sum(t>=0 and c==t for t,c in zip(s['targets'].tolist(),oracle_choice))
  stats['risk_edges']+=int(s['eligible'].sum());stats['risk_rows']+=int(s['eligible'].any(-1).sum())
  if records and not s['audit_block']:samples.append(s)
  labeler.update(inst,f,v);actor.committed(inst,v,f)
  if time.monotonic()-last[0]>20:
   if report:save(OUT/'opportunity'/f'{kind}_video{video:02d}'/'PROGRESS.json',dict(status='NATIVE_GMT_OPPORTUNITY',frame=f,frames=frames,counts=dict(stats)))
   print('GMT_OPPORTUNITY_PROGRESS',video,kind,f,frames,dict(stats),flush=True);last[0]=time.monotonic()
 m.jev_candidate_commit_observer=after
 with torch.no_grad():raw,_=run(m,values,total,frames-1)
 result=dict(status='COMPLETE',video=video,kind=kind,frames=frames,counts=dict(stats),local_error_groups=len(groups),
  residual_units='average GMT trajectory score; raw residual multiplied by original history support',
  unique_target='first-past>=3 known observations and>=80% coverage; immutable first owner must remain>=80%pure; exactly one clean compatible ID across all native histories, present in active lawful candidates',
  no_multi_positive_alias_targets=True,GT_actor_input=False,actual_mutated_state=True,rule_diagnostic='same original-GMT state; accepted active changes only certified; rejected alternatives have UNKNOWN Bank outcome. Not full-loop rule benefit.',
  ORACLE='OFFLINE_SAME_STATE_UNIQUE_TARGET_ASSIGNMENT_BOUND_NOT_DEPLOYABLE_NOT_HOTA_BOUND',seconds=time.monotonic()-begin,source=binding())
 if report:save(REPORTS/f'OPPORTUNITY_{kind}_VIDEO{video:02d}.json',result)
 print('GMT_OPPORTUNITY_COMPLETE',video,kind,dict(stats),flush=True)
 del m,raw;torch.cuda.empty_cache()
 return samples,result

def main(videos):
 assert read(REPORTS/'BASELINE_PARITY.json')['status']=='PASS_EXACT'
 results=[collect(v,records=False)[1] for v in videos]
 counts=collections.Counter()
 for d in results:counts.update(d['counts'])
 save(REPORTS/'GMT_ERROR_OPPORTUNITY.json',dict(status='COMPLETE_ACTUAL_GMT_PREFIX_DIAGNOSTIC',cases=results,counts=dict(counts),source='complete original Stage2 GMT, not old JEV D0/D1',
  learned_training_signal_gate=counts['rule_local_N01']>counts['rule_local_N10'],native_rule_validation_still_required=True,GT_inference=False,independent_validation=False))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--videos',type=int,nargs='+',default=[12,13,14]);a=p.parse_args();main(a.videos)
