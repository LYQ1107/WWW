"""Actual full frozen-policy trajectories; owner labels remain observer-only."""
from jev_phase19_common import *
import argparse,gzip,collections
import torch
from jev_phase16_common import load_policy
from jev_phase19_runtime import build_tracker,cache_inputs,run
from jev_phase16_memory_risk import MemoryRisk
from gtr.modeling.jev_phase15.native_commit_adapter import attach
from gtr.modeling.jev_stage2.assignment import lawful_choice
from gtr.modeling.jev_native_state import fingerprint

def alternatives(refs,logits,cos,legal,original):
 candidates=[j for j,r in enumerate(refs) if legal[j] and r!=original]
 byscore=sorted(candidates,key=lambda j:(-logits[j],refs[j]))[:2];bycos=sorted(candidates,key=lambda j:(-cos[j],refs[j]))[:2]
 result=[]
 for j in byscore+bycos:
  if refs[j] not in result:result.append(refs[j])
 return result[:3]+[None] if original is not None else result[:4]
def main(video):
 protect();allowed(video);folder=OUT/'census_v1'/f'video{video:02d}';folder.mkdir(parents=True,exist_ok=True);assert not (folder/'RESULT.json').exists()
 torch.set_num_threads(1);torch.manual_seed(20261009);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 policy,checkpoint,training=load_policy('v3');values,frames,reader=cache_inputs(video);model=build_tracker(video,policy);executor=attach(model);risk=MemoryRisk(video,reader)
 counts=collections.Counter();pending={};start=time.monotonic();heartbeat=[0]
 queries=gzip.open(folder/'QUERIES.jsonl.gz','wt');commits=gzip.open(folder/'COMMITS.jsonl.gz','wt');events=gzip.open(folder/'EVENTS.jsonl.gz','wt')
 def before(**d):
  risk.before(**d);key=[d['context']['frame'],d['context']['view']];task=d['task'];x=d['batch'];refs=d['refs'];score=d['logits'].detach().cpu();legal=x['legal'][0].cpu();choices=lawful_choice(score,legal)
  queries.write(json.dumps(dict(key=key,task=task,rows=d['context'].get('rows',list(range(len(score)))),refs=refs,legal=legal.tolist(),scores=score.tolist(),assignment=choices,WHO_Q4_SHA=fingerprint({k:x[k] for k in ['history_visual','history_mask','identity_meta','pair_evidence','commitment_features']})))+'\n')
  if task!=0:
   for local,row in enumerate(d['context']['rows']):pending[row]['actual_REACT_refs']=refs;pending[row]['actual_REACT_legal']=legal[local].tolist()
   return
  current_gt=risk.labels.current(*key);cos=x['pair_evidence'][0,:,:,:4].max(-1).values.cpu().tolist();assert len(current_gt)==len(score)
  owners=[risk.owner.get(r) for r in refs];pure=[];mixed=[];known_coverage=[]
  for r in refs:
   v=risk.votes[r];n=sum(v.values());pure.append(next(iter(v)) if len(v)==1 and n>=3 and n/max(1,risk.observations[r])>=.8 else None);mixed.append(len(v)>1);known_coverage.append(n/max(1,risk.observations[r]))
  for row,(gt,col) in enumerate(zip(current_gt,choices)):
   selected=refs[col] if col>=0 else None;mask=legal[row].tolist();available=[r for j,r in enumerate(refs) if mask[j] and gt is not None and owners[j]==gt];pure_available=[r for j,r in enumerate(refs) if mask[j] and gt is not None and pure[j]==gt];alt=alternatives(refs,score[row].tolist(),cos[row],mask,selected)
   ranked=sorted([r for j,r in enumerate(refs) if mask[j]],key=lambda r:(-cos[row][refs.index(r)],r));closest=ranked[0] if ranked else None;close_conflict=[]
   if closest is not None:
    best=cos[row][refs.index(closest)];close_conflict=sorted({owners[j] for j,r in enumerate(refs) if mask[j] and owners[j] is not None and best-cos[row][j]<=.05})
   pending[row]=dict(event_key=[video,*key,row],GT_OFFLINE=gt,refs=refs,legal=mask,owner_GT_OFFLINE=owners,current_pure_owner_GT_OFFLINE=pure,mixed_history=mixed,known_coverage=known_coverage,original_MATCH_ref=selected,original_MATCH_column=col,scores=score[row].tolist(),cosine=cos[row],alternatives_GT_free=alt,correct_owner_refs_OFFLINE=available,current_pure_correct_refs_OFFLINE=pure_available,numeric_Recall_at_K={str(k):bool(set(ranked[:k])&set(available)) if available else None for k in [1,4,8,16,32,len(ranked)]},alternative_owner_coverage_OFFLINE=bool(set(a for a in alt if a is not None)&set(available)) if available else None,numeric_close_owner_conflict_OFFLINE=close_conflict,actual_REACT_refs=None,actual_REACT_legal=None)
 def after(**d):
  d['video']=video;past_owners=dict(risk.owner);risk.after(**d);inst=d['instances'][-1];ids=inst.track_ids.cpu().tolist();key=[d['frame'],d['view']];commits.write(json.dumps(dict(key=key,ids=ids,events=d['events'],committed_vectors_SHA=fingerprint(inst.reid_features),raw_memory_SHA=fingerprint(executor.memory.state_dict())))+'\n')
  for row,(ref,action) in enumerate(zip(ids,d['events'])):
   item=pending.pop(row);gt=item['GT_OFFLINE'];owner=next((o for r,o in zip(item['refs'],item['owner_GT_OFFLINE']) if r==ref),past_owners.get(ref));known=gt is not None and owner is not None;available=bool(item['correct_owner_refs_OFFLINE']);wrong=known and owner!=gt;correct=known and owner==gt;false_split=action['action']=='START_NEW' and available
   category='UNKNOWN_CURRENT' if gt is None else 'AVAILABLE_ERROR' if (wrong or false_split) and available else 'UNAVAILABLE_ERROR' if wrong else 'CORRECT_CONTROL' if correct else 'UNKNOWN_OWNER_OR_NEW'
   item.update(actual_committed_ref=ref,actual_action=action['action'],owner_before_commit_OFFLINE=owner,known_owner=known,wrong_owner=wrong,correct_owner=correct,false_split_with_anchored_support=false_split,category=category,strict=True,this_round_split="TRAIN" if video in TRAIN else "RESERVED")
   events.write(json.dumps(item,allow_nan=False)+'\n');counts[category]+=1;counts['events']+=1;counts['legal_K_gt16']+=sum(item['legal'])>16
  if time.monotonic()-heartbeat[0]>=15:
   queries.flush();commits.flush();events.flush();save(folder/'PROGRESS.json',dict(status='RUNNING',key=key,total_frames=frames,elapsed_seconds=time.monotonic()-start,counts=dict(counts),pid=os.getpid()));heartbeat[0]=time.monotonic();print('CENSUS_PROGRESS',video,key,flush=True)
 executor.observer=before;executor.commit_observer=after
 try:
  with torch.no_grad():run(model,values,frames)
 finally:queries.close();commits.close();events.close()
 with gzip.open(folder/'OWNER_TRACE.jsonl.gz','wt') as f:
  for r in risk.strong_trace:f.write(json.dumps(r)+'\n')
 from run_jev_phase10_closed_loop import raw_predictions
 from jev_phase19_eval import prepare
 predictions=folder/'RAW_PREDICTIONS.json';save(predictions,raw_predictions(raw,risk.labels.images));evaluation=prepare(folder,video,predictions)
 result=dict(status='COMPLETE',raw_predictions=ref(predictions),strict_evaluation=evaluation,binding=binding(checkpoint=checkpoint,training=training),video=video,frames=frames,queries=ref(folder/'QUERIES.jsonl.gz'),commits=ref(folder/'COMMITS.jsonl.gz'),events=ref(folder/'EVENTS.jsonl.gz'),owner_trace=ref(folder/'OWNER_TRACE.jsonl.gz'),owner_propagation=risk.horizon((0,0),frames),counts=dict(counts),GT_actor_inputs=False,actual_native_mutated_state=True,seconds=time.monotonic()-start)
 save(folder/'RESULT.json',result);save(folder/'PROGRESS.json',dict(status='COMPLETE',result=ref(folder/'RESULT.json')));print('CENSUS_COMPLETE',video,dict(counts),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--video',type=int,required=True);main(p.parse_args().video)
