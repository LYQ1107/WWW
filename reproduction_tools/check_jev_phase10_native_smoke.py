"""Real production resume regression, including a mid-frame camera boundary."""
from jev_phase10_common import *
import copy,json,random
import numpy as np
import torch
from gtr.modeling.jev_native_state import NativeProductionPrefixRecorder,NativeStateForkAdapter,fingerprint

def main():
 protect();out=OUT/'smoke_v2';assert not out.exists();out.mkdir()
 source=binding();model=build_model(12);rows=inputs(12,8)
 random.seed(20261008);np.random.seed(20261008);torch.manual_seed(20261008)
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 recorder=NativeProductionPrefixRecorder(model,[(12,3,1)],out/'capture')
 model.jev_native_prefix_observer=recorder.before;model.jev_candidate_commit_observer=recorder.after
 with torch.no_grad():original,_=model.sliding_inference_GMT(rows,2,[None,None,list(range(16))],native_raw=True)
 original_sha=fingerprint(original);prefix=recorder.saved[(12,3,1)]['path']
 fork=NativeProductionPrefixRecorder(model,[],out/'fork')
 model.jev_native_prefix_observer=fork.before;model.jev_candidate_commit_observer=fork.after
 with torch.no_grad():resumed,_=NativeStateForkAdapter(model).run(prefix,rows,stop_frame=7)
 assert fingerprint(resumed)==original_sha,'raw production output state differs after resume'
 expected=[x for x in recorder.trace if tuple(x['key'])>=(12,3,1)]
 assert fork.trace==expected,'full mutable state/event ledger/RNG differs on resumed continuation'
 result={'status':'PASS_ACTUAL_NATIVE_RESUME','binding':source,'test_scope':'diagnostic; canonical221-prefix acceptance separately required','key':[12,3,1],'tested_frames':8,'resumed_commits':len(fork.trace),'full_raw_instances_parity':True,'full_commit_state_and_events_and_RNG':True,'no_GT':True,'prefix_SHA256':sha(prefix)}
 save(out/'RESULT.json',result);save(REPORTS/'NATIVE_RESUME_SMOKE.json',result);print('NATIVE_RESUME_SMOKE_PASS',len(fork.trace),flush=True)
if __name__=='__main__':main()
