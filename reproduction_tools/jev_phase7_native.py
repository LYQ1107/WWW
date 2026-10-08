"""Instrument the unchanged production MATCH operator, not a solver approximation."""
from collections import Counter
from dataclasses import replace
import hashlib,importlib,json,time
from types import SimpleNamespace
import torch
from torch import nn
from gtr.modeling.meta_arch.gtr_rcnn import GTRRCNN
from gtr.modeling.jev_state import count_track_history,legacy_acceptance_threshold
from jev_phase6_native_match import NativeMatchResolver
from jev_phase7_policies import A,R,N

class AttributionResolver:
    def __init__(self,lab,actor,validator='OWN',global_reassociation=True,budget=None,writer=None):
        self.lab=lab;self.engine=lab.engine;self.original=self.engine.resolve_actions
        self.actor=actor;self.validator=validator;self.global_reassociation=global_reassociation
        self.budget=budget;self.writer=writer;self.cache={};self.validations=[];self.stats=Counter()
        self.last=None;self.trigger_rows=0
    def reset(self):self.cache.clear();self.validations=[];self.last=None
    def resolve(self,perception,state,*,actions=None,threshold=None,proposal=None):
        original_actions=dict(actions);cache_key=(id(proposal),tuple(sorted(original_actions.items())),state.trajectory_rng_calls)
        if cache_key in self.cache:return self.cache[cache_key]
        actor_rows={int(r['context']['detection_index']):r for r in self.lab.decision_rows if r['question']=='MATCH_DECISION'}
        actions=dict(original_actions)
        if self.budget is not None:
            candidates=[r for r in proposal.pairs if len(proposal.track_ids)>1]
            desired=int(self.budget*(self.lab.steps_completed+1)/max(1,len(self.lab.keys)-1))
            take=min(len(candidates),max(0,desired-self.trigger_rows))
            ranked=sorted(candidates,key=lambda r:(-actor_rows[r]['probabilities'].get(R,0),r))
            selected=set(ranked[:take])
            for row in candidates:
                if row in selected:actions[row]=R
                elif actions[row]==R:
                    feature=torch.tensor(actor_rows[row]['feature_vector'])
                    actions[row]=self.actor.decide(feature,'MATCH_DECISION',[A,N],off_action=actor_rows[row]['off_action']).committed_action
        reassociate=[r for r,a in actions.items() if a==R]
        # Avoid the Phase VI bridge's discarded legacy global solve. Base
        # resolution is observational and only creates the transport object.
        base_actions={r:A if a==R else a for r,a in actions.items()}
        resolution=self.original(perception,state,actions=base_actions,threshold=threshold,proposal=proposal)
        resolution=dict(resolution,actions=actions,reassociate_rows=tuple(reassociate))
        proposal=resolution['initial_proposal'];model=self.engine.association_fn.model
        base=float(model.overlap_thresh if threshold is None else threshold)
        lengths=torch.tensor([count_track_history(state.association_history,t)for t in proposal.track_ids])
        original=torch.full((len(proposal.scores),),-1,dtype=torch.long)
        for row,col in proposal.pairs.items():
            if float(proposal.scores[row,col])>legacy_acceptance_threshold(base,int(lengths[col]),bool(model.not_mult_thresh)):
                original[row]=int(proposal.track_ids[col])
        banned={(r,c)for r,a in actions.items()if a==N for c in range(len(proposal.track_ids))}
        banned.update((r,proposal.pairs[r])for r in reassociate)
        second_pairs=dict(proposal.pairs);native_ids=[resolution['existing_track_ids'][r] if resolution['existing_track_ids'][r] is not None else -1 for r in range(len(proposal.scores))]
        solve_seconds=validation_seconds=native_seconds=0.;validations=[];input_checks=[]
        if reassociate:
            owner=self
            class Bridge:
                mode='jev'
                def decide(self,feature,question,legal,*,off_action=None,context=None):
                    nonlocal validation_seconds
                    context=context or {};row=int(context['detection_index'])
                    if context.get('proposal_round')!=2:
                        assert actions[row] in legal
                        expected=torch.tensor(actor_rows[row]['feature_vector'])
                        err=float((feature.detach().cpu()-expected).abs().max())
                        assert err==0., ('native first-round state mismatch',err)
                        input_checks.append({'row':row,'feature_max_error':err,'legal':list(legal)})
                        return SimpleNamespace(committed_action=actions[row])
                    stamp=time.perf_counter()
                    if owner.validator=='LEGACY':result=SimpleNamespace(committed_action=off_action,probabilities={a:float(a==off_action)for a in legal})
                    else:result=owner.actor.decide(feature,question,legal,off_action=off_action,context=context)
                    validation_seconds+=time.perf_counter()-stamp
                    validations.append({'row':row,'action':result.committed_action,'off_action':off_action,
                        'feature_vector':feature.detach().cpu().tolist(),'legal':list(legal),
                        'candidate_id':int(context['proposal_track_id']),'rejected_id':context['rejected_track_id'],
                        'probabilities':result.probabilities})
                    return result
            tracker=object.__new__(GTRRCNN);nn.Module.__init__(tracker)
            tracker.register_parameter('_contract_device',nn.Parameter(torch.zeros(1),requires_grad=False))
            tracker.jev_policy=Bridge();tracker.jev_state_dim=64;tracker.jev_max_reassociate=1
            tracker.with_bank=bool(model.with_bank);tracker.with_iou=bool(model.with_iou);tracker.not_mult_thresh=bool(model.not_mult_thresh);tracker._jev_context={}
            native_module=importlib.import_module('gtr.modeling.meta_arch.gtr_rcnn');real_solve=native_module.constrained_hungarian
            def measured_solve(scores,native_banned):
                nonlocal second_pairs,solve_seconds
                assert scores is proposal.scores and set(native_banned)==banned
                stamp=time.perf_counter()
                pairs=real_solve(scores,native_banned) if owner.global_reassociation else [(r,c)for r,c in proposal.pairs.items()if actions[r]!=N]
                solve_seconds+=time.perf_counter()-stamp;second_pairs=dict(pairs);return pairs
            native_module.constrained_hungarian=measured_solve
            rows=list(proposal.pairs);cols=[proposal.pairs[r]for r in rows];stamp=time.perf_counter()
            try:native=NativeMatchResolver.native_call(tracker,original,proposal,rows,cols,base,perception,state,lengths)
            finally:native_module.constrained_hungarian=real_solve
            native_seconds=time.perf_counter()-stamp;native_ids=native.tolist()
        columns={t:c for c,t in enumerate(proposal.track_ids)}
        final=replace(proposal,pairs={r:columns[t]for r,t in enumerate(native_ids)if t>=0},
                      banned_edges=tuple(sorted(banned)) if reassociate else proposal.banned_edges,
                      transformer_calls=0 if reassociate else proposal.transformer_calls,proposal_reused=bool(reassociate))
        resolution.update(final_proposal=final,existing_track_ids={r:t if t>=0 else None for r,t in enumerate(native_ids)})
        changes=[r for r in set(proposal.pairs)|set(second_pairs)if proposal.pairs.get(r)!=second_pairs.get(r)]
        accept_moved=[r for r in changes if actions[r]==A]
        self.trigger_rows+=len(reassociate)
        self.stats.update(payloads=1,match_rows=len(actions),reassociate_rows=len(reassociate),reassociate_payloads=int(bool(reassociate)),
            banned_edges=len(banned) if reassociate else 0,changed_pairs=len(changes),accept_rows_moved=len(accept_moved),
            accepted_moved_committed=sum(native_ids[r]>=0 for r in accept_moved),second_validations=len(validations),
            second_validation_accepted=sum(v['action']==A for v in validations))
        self.stats['solver_seconds']+=solve_seconds;self.stats['validation_seconds']+=validation_seconds;self.stats['native_seconds']+=native_seconds
        score_sha=hashlib.sha256(proposal.scores.detach().cpu().numpy().tobytes()).hexdigest()
        record={'key':[int(perception['video_id']),int(perception['frame']),int(perception['view'])],
                'track_ids':list(proposal.track_ids),'scores':proposal.scores.tolist(),'score_sha256':score_sha,
                'initial_pairs':dict(proposal.pairs),'raw_first_actions':original_actions,'first_actions':actions,
                'banned_edges':sorted(banned) if reassociate else [],'second_pairs':second_pairs,
                'native_existing_ids':native_ids,'changed_rows':changes,'accept_moved_rows':accept_moved,
                'second_validations':validations,'first_input_checks':input_checks,
                'solver_seconds':solve_seconds,'validation_seconds':validation_seconds,'native_seconds':native_seconds,
                'global_reassociation':self.global_reassociation,'validator':self.validator,
                'first_decisions':list(actor_rows.values())}
        self.last=record;self.validations=validations;self.cache[cache_key]=resolution
        if self.writer:self.writer.write(json.dumps(record,sort_keys=True,allow_nan=False)+'\n')
        return resolution
