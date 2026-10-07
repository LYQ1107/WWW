"""Opt-in native MATCH transition contract; legacy anchor replay is untouched."""
from dataclasses import replace
from types import SimpleNamespace

import torch
from torch import nn

from gtr.modeling.meta_arch.gtr_rcnn import GTRRCNN
from gtr.modeling.jev_runtime import JEVRuntimePolicy
from gtr.modeling.jev_state import count_track_history,legacy_acceptance_threshold,association_window_length


class NativeMatchResolver:
    def __init__(self,engine,controller):
        self.engine=engine;self.original=engine.resolve_actions;self.controller=controller
        self.cache={};self.validations=[]

    def reset(self):
        self.cache.clear();self.validations=[]

    def resolve(self,perception,state,*,actions=None,threshold=None,proposal=None):
        resolution=self.original(perception,state,actions=actions,threshold=threshold,proposal=proposal)
        if not resolution['reassociate_rows']:return resolution
        proposal=resolution['initial_proposal'];actions=resolution['actions']
        cache_key=(id(proposal),tuple(sorted(actions.items())),state.trajectory_rng_calls)
        if cache_key in self.cache:return self.cache[cache_key]
        model=self.engine.association_fn.model
        base=float(model.overlap_thresh if threshold is None else threshold)
        lengths=torch.tensor([count_track_history(state.association_history,t) for t in proposal.track_ids])
        original=torch.full((len(proposal.scores),),-1,dtype=torch.long)
        for row,col in proposal.pairs.items():
            if float(proposal.scores[row,col])>legacy_acceptance_threshold(base,int(lengths[col]),bool(model.not_mult_thresh)):
                original[row]=int(proposal.track_ids[col])
        real_policy=JEVRuntimePolicy('jev',self.controller)
        owner=self
        class FirstActionsThenNativeValidation:
            mode='jev'
            def decide(self,feature,question,legal,*,off_action=None,context=None):
                context=context or {}
                if context.get('proposal_round')!=2:
                    row=int(context['detection_index']);action=actions[row]
                    if action not in legal:raise AssertionError('supplied first-round action is illegal')
                    return SimpleNamespace(committed_action=action)
                result=real_policy.decide(feature,question,legal,off_action=off_action,context=context)
                owner.validations.append({'frame':int(perception['frame']),'view':int(perception['view']),
                    'row':int(context['detection_index']),'action':result.committed_action,
                    'off_action':off_action,'feature_vector':list(result.state_features),
                    'proposal_track_id':context['proposal_track_id'],'probabilities':dict(result.probabilities)})
                return result
        tracker=object.__new__(GTRRCNN);nn.Module.__init__(tracker)
        tracker.register_parameter('_contract_device',nn.Parameter(torch.zeros(1),requires_grad=False))
        tracker.jev_policy=FirstActionsThenNativeValidation();tracker.jev_state_dim=64;tracker.jev_max_reassociate=1
        tracker.with_bank=bool(model.with_bank);tracker.with_iou=bool(model.with_iou);tracker.not_mult_thresh=bool(model.not_mult_thresh);tracker._jev_context={}
        rows=list(proposal.pairs);cols=[proposal.pairs[r] for r in rows]
        native=tracker._apply_jev_match_decisions(original,proposal.scores,torch.tensor(proposal.track_ids),rows,cols,base,
            view=int(perception['view']),frame_index=int(perception['frame']),
            window_length=association_window_length(history_instances=len(state.association_history),view_num=2,view_index=int(perception['view']),
                first_frame_secondary_view=int(perception['frame'])==0 and len(state.association_history)==1),
            track_lengths=lengths,detection_boxes=perception['pred_boxes'],detection_scores=perception['detection_scores'],detection_image_size=perception['image_size'])
        native_ids=native.tolist();columns={t:c for c,t in enumerate(proposal.track_ids)}
        final=replace(resolution['final_proposal'],pairs={r:columns[t] for r,t in enumerate(native_ids) if t>=0})
        result=dict(resolution,final_proposal=final,existing_track_ids={r:t if t>=0 else None for r,t in enumerate(native_ids)})
        self.cache[cache_key]=result
        return result
