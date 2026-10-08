"""Compare real production evidence with the identical causal-replay prefix.

This is a necessary offline/online feature check, not another identity metric.
Current GT/future labels are never provided to the production scorer.
"""
import json,copy,os
from types import SimpleNamespace
import torch
from jev_phase9_common import *
from gtr.modeling.jev_candidate_features import evidence12
from gtr.modeling.jev_candidate_policy import CandidateValuePolicy
from gtr.modeling.jev_perception_cache import FrozenPerceptionCache
from gtr.modeling.jev_state import count_track_history
from jev_phase7_state import CompleteStateDigest

def main():
 protect();source=binding();assert not source['dirty'];out=OUT/'feature_bridge_v2';assert not out.exists();out.mkdir()
 v=16;m=json.loads((PREVIOUS/'opportunity_scan_v1/video16/SCAN_RESULT.json').read_text());record=m['bounded_snapshots'][0];assert sha(record['path'])==record['sha256'];s=torch.load(record['path'],map_location='cpu');key=tuple(s['key']);p=s['proposal'];state=s['state'];lab=native_lab(v);model=lab.engine.association_fn.model
 model.jev_enabled=True;model.jev_candidate_policy=CandidateValuePolicy('gmt_compat')
 from gtr.modeling.jev_runtime import JEVRuntimePolicy
 model.jev_policy=JEVRuntimePolicy('off',None);model.jev_perception_cache_reader=FrozenPerceptionCache(lab.pilot.CACHE);model.jev_perception_cache=None
 # Both trajectories start from the same real trajectory seed, not just IDs.
 os.environ['JEV_TRAJECTORY_RNG_MASTER_SEED']=str(state.trajectory_rng_seed-v)
 from jev_phase9_gallery_bridge import NativeGalleryBridge
 bridge=NativeGalleryBridge(lab);restored=bridge.galleries(state,key)
 payload=lab.cache.load(*key);galleries={t:xs.mean(0)for t,xs in restored.items()};lengths=torch.tensor([count_track_history(state.association_history,t)for t in p.track_ids],dtype=torch.float32)
 replay=evidence12(p.scores,tuple(p.track_ids),p.pairs,lengths,hits=state.track_hits,galleries=galleries,memory_lengths={int(t):len(xs)for t,xs in restored.items()},bank_eligible=state.possible_memory_ids,observations=payload['reid_features'])
 comparisons=[]
 def observe(**d):
  if (d['frame'],d['view'])!=(key[1],key[2]):return
  c=d['candidate'];batch=c['batch'];assert tuple(batch.candidate_ids)==tuple(p.track_ids),'candidate references diverged before target'
  native=batch.evidence12.detach().cpu();delta=(native-replay).abs();actual=copy.deepcopy(d['galleries'])
  result={'key':list(key),'snapshot_sha256':record['sha256'],'same_candidate_reference_order':True,'current_raw_score_matrix_exact':torch.equal(batch.scores.detach().cpu(),p.scores),'current_raw_score_max_error':float((batch.scores.detach().cpu()-p.scores).abs().max()),'canonical_evidence12_max_error':float(delta.max()),'per_field_max_error':{n:float(delta[...,i].max())for i,n in enumerate(__import__('gtr.modeling.jev_candidate_features',fromlist=['FEATURE_NAMES']).FEATURE_NAMES)},'replay_prefix_hits':dict(state.track_hits),'replay_prefix_memory_lengths':{str(t):len(state.memory.get(t,[]))for t in p.track_ids},'restored_production_gallery_lengths':{str(t):len(restored[t])for t in p.track_ids},'restoration_source':bridge.source,'real_initial_birth_vectors_reconstructed':True,'production_pre_MATCH_memory_lengths':batch.evidence12[0,:,5].tolist(),'matching_native_trajectory_seed':state.trajectory_rng_seed,'current_committed_IDs':d['instances'][-1].track_ids.tolist(),'scope':'same video/frame/view/score/proposed candidates, real full sliding production vs canonical re-encoding of stored causal replay prefix; production observation is after commit but candidate batch contains pre-MATCH gallery fields'}
  torch.save({'production':native,'canonical_replay':replay,'delta':delta,'production_state64':batch.state64.detach().cpu()},out/'EVIDENCE_COMPARISON.pth');comparisons.append(result)
 model.jev_candidate_commit_observer=observe;frames=key[1]+1;inputs=[]
 for view in range(2):
  for frame in range(frames):
   image=lab.pilot.image_for(lab.lookup,v,frame,view);inputs.append({'video_id':v,'view_num':2,'height':image['height'],'width':image['width'],'image':None})
 with torch.no_grad():model.sliding_inference_GMT(inputs,2,[None,None,list(range(2*frames))])
 assert len(comparisons)==1
 r=comparisons[0];passed=r['canonical_evidence12_max_error']<=1e-5 and r['current_raw_score_matrix_exact']
 report={'status':'PASS'if passed else'FAIL_OFFLINE_ONLINE_FEATURE_BRIDGE','binding':source,'comparison':r,'formal_training_authorized_by_this_test':passed,'raw_artifacts':[{'path':str(out/'EVIDENCE_COMPARISON.pth'),'sha256':sha(out/'EVIDENCE_COMPARISON.pth')}],'no_GT_or_future_in_production':True,'WHAT_DID_WE_LEARN':'Complete production value integration alone does not verify the causal-replay input/state bridge. Compare canonical evidence on matched prefix/seed before fitting.'}
 save(out/'FEATURE_BRIDGE_RESULT.json',report);save(REPORTS/'FEATURE_BRIDGE_RESTORATION_PARITY.json',report);print(json.dumps(report,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
