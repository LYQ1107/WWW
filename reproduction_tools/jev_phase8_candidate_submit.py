"""Versioned candidate submission adapter using the real production MATCH hook.

This verifies the production association operator, not full deployment parity.
The downstream mutable replay commit still requires separately scoped tests.
No annotations, GT identities, or future observations are accepted here.
"""
from dataclasses import replace
from types import SimpleNamespace
import torch
from torch import nn
from gtr.modeling.meta_arch.gtr_rcnn import GTRRCNN
from gtr.modeling.jev_state import count_track_history
from jev_phase6_native_match import NativeMatchResolver

def validate_pairs(proposal,pairs):
    m,n=proposal.scores.shape
    assert len(set(proposal.track_ids))==len(proposal.track_ids),'duplicate historical candidate IDs'
    assert len(set(pairs.values()))==len(pairs),'one identity claimed twice in one camera'
    for r,c in pairs.items():
        assert 0<=r<m and 0<=c<n,'missing candidate'
        assert (r,c)not in set(proposal.banned_edges),'masked candidate'
        assert torch.isfinite(proposal.scores[r,c]),'nonfinite candidate score'
    return dict(pairs)

def native_existing_ids(model,perception,state,proposal,pairs,actions):
    pairs=validate_pairs(proposal,pairs)
    assert set(actions)==set(range(len(proposal.scores)))
    assert all(a in ('ACCEPT_CURRENT','START_NEW')for a in actions.values())
    assert all(r in pairs for r,a in actions.items()if a=='ACCEPT_CURRENT')
    class ExplicitAssignment:
        mode='jev'
        def decide(self,feature,question,legal,*,off_action=None,context=None):
            assert not context.get('proposal_round')==2
            a=actions[int(context['detection_index'])];assert a in legal
            return SimpleNamespace(committed_action=a)
    tracker=object.__new__(GTRRCNN);nn.Module.__init__(tracker)
    tracker.register_parameter('_contract_device',nn.Parameter(torch.zeros(1),requires_grad=False))
    tracker.jev_policy=ExplicitAssignment();tracker.jev_state_dim=64;tracker.jev_max_reassociate=1
    tracker.with_bank=bool(model.with_bank);tracker.with_iou=bool(model.with_iou)
    tracker.not_mult_thresh=bool(model.not_mult_thresh);tracker._jev_context={}
    lengths=torch.tensor([count_track_history(state.association_history,t)for t in proposal.track_ids])
    original=torch.tensor([proposal.track_ids[pairs[r]]if actions[r]=='ACCEPT_CURRENT'else -1 for r in actions])
    rows=list(pairs)
    ids=NativeMatchResolver.native_call(tracker,original,proposal,rows,[pairs[r]for r in rows],float(model.overlap_thresh),perception,state,lengths)
    assert torch.equal(ids,original),'real production MATCH rejected an explicit legal assignment'
    return ids.tolist()

class DirectCandidateResolver:
    """One current-key offline intervention; all later decisions are live OFF."""
    def __init__(self,lab,target_key,pairs):
        from jev_phase7_native import AttributionResolver
        from jev_phase8_common import FrozenOFFActor
        self.lab=lab;self.key=tuple(target_key);self.pairs=dict(pairs);self.cache={};self.last=None
        self.fallback=AttributionResolver(lab,FrozenOFFActor());self.model=lab.engine.association_fn.model
    def reset(self):self.cache.clear();self.last=None;self.fallback.reset()
    def resolve(self,perception,state,*,actions=None,threshold=None,proposal=None):
        key=tuple(int(perception[k])for k in ('video_id','frame','view'))
        if key!=self.key:return self.fallback.resolve(perception,state,actions=actions,threshold=threshold,proposal=proposal)
        cache_key=(id(proposal),state.trajectory_rng_calls)
        if cache_key in self.cache:return self.cache[cache_key]
        action=dict(actions)
        # Residual rows keep their OFF decisions; selected existing edges must
        # be explicitly ACCEPTed even if the old proposal was below threshold.
        for r in self.pairs:
            if proposal.pairs.get(r)!=self.pairs[r]:action[r]='ACCEPT_CURRENT'
        for r in range(len(proposal.scores)):
            if r not in self.pairs:action[r]='START_NEW'
        ids=native_existing_ids(self.model,perception,state,proposal,self.pairs,action)
        columns={t:c for c,t in enumerate(proposal.track_ids)}
        final=replace(proposal,pairs={r:columns[t]for r,t in enumerate(ids)if t>=0},transformer_calls=0,proposal_reused=True)
        result={'initial_proposal':proposal,'final_proposal':final,'actions':action,'reassociate_rows':(),
            'existing_track_ids':{r:t if t>=0 else None for r,t in enumerate(ids)}}
        self.last={'native_existing_ids':ids,'submitted_pairs':self.pairs,'actual_pairs':final.pairs,
            'raw_score_tensor_reused':final.scores is proposal.scores,'scope':'real GTR MATCH operator + native mutable replay commit; full deployed sliding inference NOT_VERIFIED'}
        self.cache[cache_key]=result;return result
