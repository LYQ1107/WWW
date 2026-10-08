"""Actual full sliding inference, complete native commit observations and intervention.

The observer never changes scoring/assignment. Its SHA summaries include every
Instances field and all hit/gallery/bank/ID/RNG containers. No GT is used.
"""
import argparse,copy,json,hashlib,random,time,types
import numpy as np
import torch
from jev_phase9_common import *
from gtr.modeling.jev_candidate_policy import CandidateValuePolicy
from gtr.modeling.jev_perception_cache import FrozenPerceptionCache
from jev_phase7_state import CompleteStateDigest

class CurrentEvidenceValues(torch.nn.Module):
 def forward(self,state,evidence,mask):
  # Deterministic finite producer to verify output plumbing; no fitted weights.
  return evidence[...,0],state.new_full((len(state),),1e6)

def run(video=12,frames=None):
 protect();source=binding();assert not source['dirty'];out=OUT/'production_parity_v2';assert not out.exists();out.mkdir()
 save(out/'START_MANIFEST.json',{'binding':source,'video':video,'frames':frames,'uses_GT':False})
 lab=native_lab(video);model=lab.engine.association_fn.model;model.jev_enabled=True
 model.jev_perception_cache_reader=FrozenPerceptionCache(lab.pilot.CACHE)
 model.jev_perception_cache=None
 from gtr.modeling.jev_runtime import JEVRuntimePolicy
 model.jev_policy=JEVRuntimePolicy('off',None);model._jev_context={}
 total=max(k[1]for k in lab.keys)+1;limit=total if frames is None else min(frames,total)
 records={};outputs={};timing={}
 def pack(v):
  if hasattr(v,'get_fields'):return {'image_size':list(v.image_size),'fields':{k:pack(x)for k,x in v.get_fields().items()}}
  if hasattr(v,'tensor'):return v.tensor
  if isinstance(v,list):return [pack(x)for x in v]
  if isinstance(v,dict):return {k:pack(x)for k,x in v.items()}
  return v
 for tag,policy in [('GMT_OFF',None),('GMT_COMPAT',CandidateValuePolicy('gmt_compat')),('SEMANTIC_NEW_SCRIPTED',CandidateValuePolicy('model',CurrentEvidenceValues()))]:
  if tag=='SEMANTIC_NEW_SCRIPTED':length=min(8,limit)
  else:length=limit
  model.jev_candidate_policy=policy;model._jev_candidate_new_rows=();model.__dict__.pop('_jev_candidate_last',None)
  trace=[];digest=CompleteStateDigest();last=[None]
  def observe(**d):
   candidate=d.pop('candidate');rng=d.pop('trajectory_rng');state=pack(d)
   state['trajectory_rng_state']=rng.getstate()if rng is not None else None
   wrapped=types.SimpleNamespace(**state);fingerprint=digest(wrapped,{})
   row={'key':[int(d['frame']),int(d['view'])],'state_sha256':fingerprint,'id_count':d['id_count'],'committed_ids':d['instances'][-1].track_ids.detach().cpu().tolist(),'memory_counts':{str(t):len(g)for t,g in d['galleries'].items()},'bank_size':sum(len(x)for x in d['old_reids']),'first':d['first']}
   if candidate is not None:
    expected=list(candidate['assignment'].existing_ids);actual=row['committed_ids']
    # Existing choices must actually commit. Explicit NEW must be fresh and
    # cannot be silently replaced by a stale-bank ID.
    for r,t in enumerate(expected):
     if t>=0:assert actual[r]==t,('existing values not committed',tag,row['key'],r,t,actual[r])
     elif candidate['assignment'].semantic_new:assert actual[r]not in candidate['candidate_ids']
    row['match_value_choices']=expected;row['state_features_shape']=list(candidate['batch'].state64.shape);row['evidence_shape']=list(candidate['batch'].evidence12.shape)
   if last[0]is not None and tag=='SEMANTIC_NEW_SCRIPTED':
    assert d['id_count']>=last[0],'birth counter went backwards'
   last[0]=d['id_count'];trace.append(row)
   if len(trace)%128==0:save(out/'PROGRESS.json',{'status':'RUNNING','tag':tag,'commits':len(trace),'key':row['key']})
  model.jev_candidate_commit_observer=observe
  inputs=[]
  for view in range(2):
   for frame in range(length):
    image=lab.pilot.image_for(lab.lookup,video,frame,view)
    inputs.append({'video_id':video,'view_num':2,'height':int(image['height']),'width':int(image['width']),'image':None})
  random.seed(20261008);np.random.seed(20261008);torch.manual_seed(20261008)
  torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
  begin=time.monotonic()
  with torch.no_grad():preds,views=model.sliding_inference_GMT(inputs,2,[None,None,list(range(2*length))])
  timing[tag]=time.monotonic()-begin;records[tag]=trace
  output=[pack(x)for x in preds];outdigest=CompleteStateDigest()(types.SimpleNamespace(predictions=output),{})
  outputs[tag]={'full_output_SHA':outdigest,'frames':length,'commits':len(trace),'actual_unique_per_view':all(len(r['committed_ids'])==len(set(r['committed_ids']))for r in trace)}
  save(out/(tag+'_TRACE.json'),trace);save(out/'PROGRESS.json',{'status':'RUNNING','finished_tag':tag,'outputs':outputs})
 baseline=records['GMT_OFF'];compat=records['GMT_COMPAT'];assert len(baseline)==len(compat)
 assert all(a['state_sha256']==b['state_sha256']and a['committed_ids']==b['committed_ids']for a,b in zip(baseline,compat)),'full mutable-state/RNG compatibility failed'
 assert outputs['GMT_OFF']['full_output_SHA']==outputs['GMT_COMPAT']['full_output_SHA'],'actual postprocessed outputs differ'
 changes=sum(a['committed_ids']!=b['committed_ids']for a,b in zip(baseline,records['SEMANTIC_NEW_SCRIPTED']))
 assert changes>0,'scripted candidate values did not alter deployed identities'
 proof={'status':'PASS_BOUNDED_PRODUCTION_CONTRACT'if frames is not None else'PASS_FULL_VIDEO_GMT_COMPAT_AND_BOUNDED_VALUE_MUTATION','binding':source,'video':video,'full_video_complete':limit==total,'total_frames':total,'tested_frames':limit,'same_frozen_cache':True,'all_native_commit_fields_and_RNG_SHA_parity':True,'actual_postprocessed_output_SHA_parity':True,'semantic_NEW_deployed_changed_commits':changes,'semantic_NEW_bounded_frames':min(8,limit),'outputs':outputs,'elapsed_seconds':timing,'trace_artifacts':[{'path':str(out/(t+'_TRACE.json')),'sha256':sha(out/(t+'_TRACE.json'))}for t in records],'no_GT_or_future_inputs':True,'models_trained':0,'neural_producer':'deterministic Torch module, not learned model or performance result','existing_choice_mutated_state_integration':'NOT_YET_TESTED','learned_policy_full_closed_loop':'NOT_RUN'}
 save(out/'PRODUCTION_RESULT.json',proof);save(REPORTS/'NATIVE_COMMIT_PARITY.json',proof);save(out/'PROGRESS.json',{'status':'COMPLETE','video':video});print('PRODUCTION_PARITY_COMPLETE',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--video',type=int,default=12);p.add_argument('--frames',type=int);a=p.parse_args();run(a.video,a.frames)
