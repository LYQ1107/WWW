"""GT-free, in-memory replay of complete native tracker calls.

This is an offline experiment adapter. It never edits track_ids by hand and
does not enable legacy JEV or use its incompatible prefix restore interface.
"""
import copy,inspect,random
import numpy as np
import torch
from torch import nn
from gtr.modeling.jev_native_state import fingerprint
from gtr.modeling.meta_arch.gtr_rcnn import poss_ids,old_ids,old_reids

class RestorationProbe(Exception):pass

class FixedScores(nn.Module):
    enabled=True
    def __init__(self,base,new,ids,lengths):
        super().__init__();self.base=base;self.new=new;self.ids=ids;self.lengths=lengths;self.calls=0
    def forward(self,scores,ids,lengths,*a,**kw):
        assert torch.equal(scores,self.base) and torch.equal(ids,self.ids) and torch.equal(lengths,self.lengths)
        self.calls+=1;return self.new

def globals_packet(model):
    return dict(possible=poss_ids.poss_ids,old_ids=old_ids.old_ids,old_reids=old_reids.old_reids,
        context=model._jev_context,python_rng=random.getstate(),numpy_rng=np.random.get_state(),
        torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state())

def call_digest(model,args,kwargs,out=None):
    return fingerprint(dict(args=args,kwargs=kwargs,native_globals=globals_packet(model),
        final_id_count=None if out is None else int(out[1])))

def replay(model,fn,args,kwargs,base,new,ids,lengths,*,probe_exception=False):
    assert model.jev_policy is None and model.jev_candidate_policy is None
    assert model._jev_trajectory_rng is None and not model.jev_enabled
    original=globals_packet(model)
    before=call_digest(model,args,kwargs)
    # One deepcopy preserves aliases between Gallery, windows and Bank entries.
    cloned=copy.deepcopy(dict(args=args,kwargs=kwargs,globals=original))
    old_correction=model.gmt_identity_correction
    old_commit=model.jev_candidate_commit_observer
    old_prefix=model.jev_native_prefix_observer
    old_memory=model.run_memory_tracker
    old_summary=fingerprint(old_correction.summaries)
    replacement=FixedScores(base,new,ids,lengths)
    result=None
    try:
        poss_ids.poss_ids=cloned['globals']['possible'];old_ids.old_ids=cloned['globals']['old_ids'];old_reids.old_reids=cloned['globals']['old_reids']
        model._jev_context=cloned['globals']['context']
        model.gmt_identity_correction=replacement
        model.jev_candidate_commit_observer=None;model.jev_native_prefix_observer=None
        # Disable the collection-only GT/statistics wrapper in the clone.
        model.run_memory_tracker=type(model).run_memory_tracker.__get__(model)
        assert call_digest(model,cloned['args'],cloned['kwargs'])==before
        if probe_exception:raise RestorationProbe('Intentional offline restoration contract probe')
        with torch.no_grad():out=fn(*cloned['args'],**cloned['kwargs'])
        assert replacement.calls==1
        bound=inspect.signature(fn).bind(*cloned['args'],**cloned['kwargs']);values=bound.arguments
        current=values['instances'][values['k']]
        native_ids=current.track_ids.detach().cpu().tolist()
        assert len(native_ids)==len(set(native_ids))
        result=dict(native_ids=native_ids,final_id_count=int(out[1]),post_SHA256=call_digest(model,cloned['args'],cloned['kwargs'],out),
                    pre_SHA256=before,score_interface_exact=True)
    finally:
        poss_ids.poss_ids=original['possible'];old_ids.old_ids=original['old_ids'];old_reids.old_reids=original['old_reids']
        model._jev_context=original['context'];model.gmt_identity_correction=old_correction
        model.jev_candidate_commit_observer=old_commit;model.jev_native_prefix_observer=old_prefix;model.run_memory_tracker=old_memory
        random.setstate(original['python_rng']);np.random.set_state(original['numpy_rng']);torch.set_rng_state(original['torch_rng']);torch.cuda.set_rng_state(original['cuda_rng'])
        assert call_digest(model,args,kwargs)==before,'Offline clone changed real native pre-state/RNG'
        assert fingerprint(old_correction.summaries)==old_summary,'Offline clone changed real committed summaries'
    return result
