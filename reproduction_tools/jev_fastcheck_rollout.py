"""Paired exogenous inputs; independent native histories and offline certificates."""
import argparse,collections,gzip,lzma,torch
from jev_fastcheck_common import *
class LabelState:
 def __init__(self,video,reader):
  from build_dense_jev_stage2_dataset import OfflineLabels
  self.labels=OfflineLabels(video,reader);self.video=video
  self.votes=collections.defaultdict(collections.Counter);self.total=collections.Counter()
  self.views=collections.defaultdict(collections.Counter);self.view_total=collections.Counter()
  self.owner={};self.booted=False;self.duplicates=0
  for key,image in self.labels.images.items():
   counts=collections.Counter(a['instance_id'] for a in self.labels.gt[image['id']]);bad={g for g,n in counts.items() if n>1}
   if bad:
    gt=self.labels.current(*key);self.duplicates+=sum(g in bad for g in gt if g is not None)
    self.labels.aligned[key]=[None if g in bad else g for g in gt]
 def update(self,frame,view,ids):
  from jev_oneweek_supervision import certify
  gt=self.labels.current(frame,view);assert len(ids)==len(gt)
  for identity,g in zip(ids,gt):
   self.total[identity]+=1;self.view_total[identity,view]+=1
   if g is not None:self.votes[identity][g]+=1;self.views[identity,view][g]+=1
   if identity not in self.owner:
    owner=certify(self.votes[identity],self.total[identity])
    if owner is not None:self.owner[identity]=owner
 def boot(self,frame,view):
  if not self.booted:
   gt=self.labels.current(frame,view);self.update(frame,view,list(range(1,len(gt)+1)));self.booted=True
 def record(self,d,detail):
  from jev_oneweek_supervision import certify
  x=d['batch'];refs=list(d['refs']);frame=d['context']['frame'];view=d['context']['view']
  gt=self.labels.current(frame,view);q,k=len(gt),len(refs);legal=x['legal'][0].cpu()
  known=torch.zeros(q,k+1,dtype=torch.bool);positive=known.clone();cross=torch.full((q,k),-1,dtype=torch.int8)
  for row,g in enumerate(gt):
   if g is None:continue
   for col,identity in enumerate(refs):
    if not legal[row,col]:continue
    if identity in self.owner:known[row,col]=True;positive[row,col]=self.owner[identity]==g
    other=certify(self.views[identity,1-view],self.view_total[identity,1-view])
    if other is not None:cross[row,col]=int(other==g)
   has=bool(positive[row,:k].any());absent=bool((known[row,:k]|~legal[row]).all())
   known[row,k]=has or absent;positive[row,k]=not has and absent
  stability=torch.full((q,k),-1,dtype=torch.int8);stability[known[:,:k]]=positive[:,:k][known[:,:k]].to(torch.int8)
  target=torch.full((q,),k,dtype=torch.long)
  for row in range(q):
   if positive[row].any():target[row]=int(torch.where(positive[row])[0][0])
  chosen=list(detail['plans'][0][int(detail['plan_scores'][0].argmax())]) if detail.get('plans') and detail['plans'][0] else []
  from gtr.modeling.jev_stage2.assignment import lawful_choice
  if not chosen and q:chosen=lawful_choice(d['logits'],x['legal'][0])
  base=detail.get('base_logits',d['logits'][None])[0].detach().cpu()
  plans=[list(p) for p in detail.get('plans',[[]])[0]] if detail.get('plans') else [chosen]
  groups=[];categories=[]
  for row,g in enumerate(gt):
   col=chosen[row];c=k if col<0 else col
   mixed=any(len(self.votes[t])>1 and g in self.votes[t] for t in refs) if g is not None else False
   active=bool(positive[row,:k].any())
   plan_covered=any(p[row]>=0 and bool(positive[row,p[row]]) for p in plans)
   category='UNKNOWN_CURRENT' if g is None else 'CORRECT_ACTIVE_IN_PLANS' if active and plan_covered else 'CORRECT_ACTIVE_OUTSIDE_PLANS' if active else 'NO_CERTIFIED_ACTIVE' if known[row,k] else 'UNKNOWN_OWNER'
   categories.append(category)
   ids=tuple(sorted(refs[c] for c in torch.where(positive[row,:k])[0].tolist()))
   groups.append((self.video,g,ids,view,frame//64))
  return dict(key=[self.video,frame,view],task=0,rows=list(range(q)),refs=refs,
   inputs={n:v.detach().cpu().clone() for n,v in x.items()},anchor_visual=x['anchor_visual'][0].detach().cpu().clone(),anchor_mask=x['anchor_mask'][0].detach().cpu().clone(),
   positive=positive,known_options=known,targets=target,supervised=positive.any(-1),
   oneweek_labels=dict(cross=cross,stability=stability),audit_block=(frame//64)%5==4,
   GT_labels_OFFLINE_ONLY=gt,groups_OFFLINE_ONLY=groups,eligibility_categories=categories,
   native_choice=chosen,base_logits=base,base_choice=lawful_choice(base.to(d['logits'].device),x['legal'][0]),plans=plans,corpus_actor='frozen_R3_JEV',
   availability=torch.where(positive[:,:k].any(-1),1,torch.where(positive[:,-1],0,-1)),
   trust=torch.full((k,),-1,dtype=torch.int8),safety=torch.full((q,k),-1,dtype=torch.int8),uncertainty=torch.full((q,k),-1,dtype=torch.int8),
   commit_positive=positive,commit_kind=torch.zeros(q,dtype=torch.int8))
def main(video,actor_name,frames):
 assert video in [12,13,14,16]
 if video!=16:assert frames==256
 from jev_oneweek_common import seed_all
 from jev_phase13_runtime import build_tracker,cache_inputs,run
 from gtr.modeling.jev_recovery.native import attach_jevtrack
 from jev_phase14_latency import prune
 from gtr.modeling.jev_native_state import fingerprint
 from gtr.modeling.meta_arch.gtr_rcnn import old_reids,poss_ids
 seed_all(SEED);actor,cp=load_policy('jev',trained=actor_name!='teacher')
 values,total,reader=cache_inputs(video);frames=total if frames==0 else frames
 folder=OUT/'rollouts'/actor_name/f'video{video:02d}';folder.mkdir(parents=True,exist_ok=False)
 model=build_tracker(video,policy=actor,react_learned=False);pruned=prune(model);executor=attach_jevtrack(model)
 state=LabelState(video,reader);records=[];chunks=[];journal=gzip.open(folder/'JOURNAL.jsonl.gz','wt')
 counts=collections.Counter();first_input=[None];first_state=[None];pending={};last=[0.];begin=time.monotonic()
 def chunk():
  if not records:return
  p=folder/f'RECORDS_{len(chunks):03d}.pth.xz'
  with lzma.open(p,'wb',preset=1) as f:torch.save(dict(records=list(records)),f)
  chunks.append(ref(p));records.clear()
 def before(**d):
  c=d['context'];key=(c['frame'],c['view'])
  if c.get('first'):state.boot(0,1-c['view'])
  if d['task']==1:
   pending.setdefault(key,{})['stale_refs']=list(d['refs']);return
  if not len(d['logits']):return
  detail=actor.runtime_details;record=state.record(d,detail)
  if first_input[0] is None:first_input[0]=fingerprint(d['batch'])
  records.append(record);pending.setdefault(key,{})['record']=record
  counts['MATCH_events']+=1
 def before_native(**d):
  if first_state[0] is None:
   from audit_jev_stage2_gta_free import phase13_prefix
   first_state[0]=fingerprint(phase13_prefix(model,d))
 def after(**d):
  key=(d['frame'],d['view']);ids=d['instances'][-1].track_ids.cpu().tolist();assert len(ids)==len(set(ids))
  if d['first']:state.boot(0,1-d['view'])
  info=pending.pop(key,{});r=info.get('record');targets=state.labels.current(*key)
  tags=[]
  for row,(identity,g) in enumerate(zip(ids,targets)):
   tag='UNKNOWN' if g is None or identity not in state.owner else 'CORRECT' if state.owner[identity]==g else 'WRONG'
   tags.append(tag);counts[tag]+=1
   if g is not None and state.votes[identity] and g not in state.votes[identity]:counts['new_person_merge']+=1
  if r is not None:
   r['committed_ids']=ids;r['stale_refs']=info.get('stale_refs',[])
   r['GT_and_predicted_owners_OFFLINE_ONLY']={str(i):state.owner.get(i) for i in ids}
  journal.write(json.dumps(dict(key=[video,*key],ids=ids,GT_OFFLINE_ONLY=targets,owner_tags=tags,
    past_owners={str(i):state.owner.get(i) for i in ids},events=d['events'],
    gallery_lengths={str(i):len(d['galleries'][i]) for i in ids},Bank_IDs=old_reids.old_reids[0].track_ids.tolist() if old_reids.old_reids else [],possible_IDs=sorted(poss_ids.poss_ids)))+'\n')
  state.update(*key,ids);counts['commits']+=1
  if key[1]==1 and key[0]%64==63:chunk()
  if time.monotonic()-last[0]>=10:
   save(folder/'PROGRESS.json',dict(status='NATIVE_ROLLOUT',key=[video,*key],frames=frames,counts=dict(counts),seconds=time.monotonic()-begin));journal.flush();print('FASTCHECK_ROLLOUT',actor_name,video,*key,dict(counts),flush=True);last[0]=time.monotonic()
 executor.observer=before;executor.commit_observer=after;model.jev_native_prefix_observer=before_native
 save(folder/'START.json',dict(source=binding('Native self/teacher trajectory capture'),actor=actor_name,checkpoint=cp,frames=frames,cache_index=ref(pathlib.Path('/home/liuyeqiang/WWW_jev_phase13_runtime/20261009_v2/stage1_cache_v1')/f'video{video:02d}/index.jsonl')))
 try:
  with torch.no_grad():raw,_=run(model,values,total,stop=frames-1)
 finally:journal.close()
 chunk()
 from run_jev_phase10_closed_loop import raw_predictions
 pred=raw_predictions(raw[:frames*2],state.labels.images);save(folder/'RAW_PREDICTIONS.json',pred)
 result=dict(status='COMPLETE',actor=actor_name,video=video,frames=frames,checkpoint=cp,record_chunks=chunks,journal=ref(folder/'JOURNAL.jsonl.gz'),
  initial_input_SHA=first_input[0],initial_native_state_SHA=first_state[0],
  raw_predictions=ref(folder/'RAW_PREDICTIONS.json'),counts=dict(counts),mixed_IDs=sum(len(v)>1 for v in state.votes.values()),
  total_IDs=len(state.votes),duplicate_GT_offline_UNKNOWN=state.duplicates,
  actual_mutated_state=True,GT_actor_inputs=False,pruned_dormant_parameters=pruned,
  native_stage2_latency_ms=dict(zip(['p50','p95'],map(float,__import__('numpy').quantile([x['total_stage2_ms'] for x in executor.latency],[.5,.95])))),
  elapsed_seconds=time.monotonic()-begin,full_image_FPS=None,scope='cached perception prefix; not full video or image FPS',source=binding('Completed real native closed loop'))
 save(folder/'RESULT.json',result);save(REPORTS/f'ROLLOUT_{actor_name}_VIDEO{video:02d}.json',result)
 print('FASTCHECK_ROLLOUT_COMPLETE',actor_name,video,dict(counts),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--video',type=int,required=True);p.add_argument('--actor',choices=['teacher','self'],required=True);p.add_argument('--frames',type=int,default=256);a=p.parse_args();main(a.video,a.actor,a.frames)
