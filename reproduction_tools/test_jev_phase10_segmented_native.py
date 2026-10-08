"""Bounded real native integration; untrained synthetic actors, no GT."""
import random
import faulthandler
import numpy as np
import torch
from jev_phase10_common import *
from gtr.modeling.jev_candidate_models import build_candidate_model,CandidateScorer
from gtr.modeling.jev_candidate_policy import CandidateValuePolicy
from gtr.modeling.jev_native_state import NativeProductionPrefixRecorder,NativeStateForkAdapter,fingerprint
from gtr.modeling.jev_candidate_assignment import assign_candidate_values


def main():
 faulthandler.enable();faulthandler.dump_traceback_later(30,repeat=True)
 runtime=configure_candidate_torchscript()
 protect();source=binding();out=OUT/os.environ.get('JEV_PHASE10_SEGMENTED_TEST_VERSION','segmented_native_engineering_v1');assert not out.exists();out.mkdir();model=build_model(12);rows=inputs(12,8);results=[]
 for name in ['Fixed','CandidateMLP','CandidateDeepSets','CandidateJEV']:
  torch.manual_seed(20261008);actor=CandidateValuePolicy('gmt_values')if name=='Fixed'else CandidateValuePolicy('model_fixed_new',torch.jit.script(CandidateScorer(build_candidate_model(name))).to('cuda:0').eval());model.jev_candidate_policy=actor;timing=[]
  original_score=actor.score
  def logged_score(batch):
   print('ENGINEERING_SCORE_SHAPE',name,tuple(batch.state64.shape),tuple(batch.evidence12.shape),batch.evidence12.stride(),flush=True);return original_score(batch)
  actor.score=logged_score
  def profile(**d):
   b=d['batch'];ref=CandidateValuePolicy('gmt_values');v,n=ref.score(b);assign_candidate_values(v,n,b.candidate_ids,b.legal_mask);assert d['feature_ms']>=0 and d['policy_assignment_ms']>=0;timing.append(d['policy_assignment_ms'])
  model.jev_candidate_latency_observer=profile
  def seeds():random.seed(20261008);np.random.seed(20261008);torch.manual_seed(20261008);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
  seeds();whole=NativeProductionPrefixRecorder(model,[],out/name/'whole');model.jev_native_prefix_observer=whole.before;model.jev_candidate_commit_observer=whole.after
  with torch.no_grad():raw,_=model.sliding_inference_GMT(rows,2,[None,None,list(range(16))],native_raw=True)
  rawsha=fingerprint(raw);seeds();part=NativeProductionPrefixRecorder(model,[(12,4,0)],out/name/'part')
  class Boundary(Exception):pass
  def before(**d):
   part.before(**d)
   if(d['frame'],d['view'])==(4,0):raise Boundary()
  model.jev_native_prefix_observer=before;model.jev_candidate_commit_observer=part.after
  try:
   with torch.no_grad():model.sliding_inference_GMT(rows,2,[None,None,list(range(16))],native_raw=True)
  except Boundary:pass
  else:raise AssertionError('segmented native loop did not stop at pre-association boundary')
  tail=NativeProductionPrefixRecorder(model,[],out/name/'tail');model.jev_native_prefix_observer=tail.before;model.jev_candidate_commit_observer=tail.after
  with torch.no_grad():resumed,_=NativeStateForkAdapter(model).run(part.saved[12,4,0]['path'],rows,stop_frame=7)
  assert fingerprint(resumed)==rawsha and whole.trace==part.trace+tail.trace,'segmented actual actor native state/event/RNG differs'
  assert all(len(r['ids'])==len(set(r['ids']))for r in whole.trace);results.append({'actor':name,'untrained_engineering_only':name!='Fixed','full_raw_instances_exact':True,'all_native_commits_events_gallery_bank_RNG_exact':True,'per_camera_IDs_unique':True,'instrumented_payloads':len(timing)});print('SEGMENTED_NATIVE_ACTOR_PASS',name,flush=True)
 faulthandler.cancel_dump_traceback_later();result={'status':'PASS_BOUNDED_ENGINEERING','binding':source,'JIT_runtime':runtime,'video':12,'frames':8,'boundary':[4,0],'actors':results,'no_GT_actor_or_labels':True,'research_fit_performed':False,'heldout_read':False,'not_an_architecture_performance_result':True};save(out/'RESULT.json',result);save(REPORTS/'SEGMENTED_NATIVE_ENGINEERING_TESTS.json',result)
if __name__=='__main__':main()
