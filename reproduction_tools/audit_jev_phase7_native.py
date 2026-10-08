"""Regression of new instrumentation against the preserved native B2 bridge."""
import copy,json
import torch
from jev_phase7_common import *

def main(device='cuda:0'):
    rollouts=isolate_phase6_imports()
    from jev_phase6_native_match import NativeMatchResolver
    from jev_phase7_native import AttributionResolver
    from jev_phase7_policies import MatchActor
    from gtr.modeling.jev_runtime import JEVRuntimePolicy
    from gtr.modeling.jev_state import count_track_history,legacy_acceptance_threshold
    from test_constrained_assignment import make_tracker
    lab=rollouts.NativeReplayLab(1,device,compact_context=True);actor=MatchActor('B2');records=[]
    source=Path('/home/liuyeqiang/WWW_jev_phase6_runtime/20261008/reassociate/snapshots')
    for path in sorted(source.glob('*.pth')):
        capture=torch.load(path,map_location='cpu');state=capture['state'];key=capture['key']
        payload=lab.cache.load(*key[:3]);proposal=lab.original_propose(payload,state)
        model=lab.engine.association_fn.model;base=float(model.overlap_thresh)
        lengths=torch.tensor([count_track_history(state.association_history,t)for t in proposal.track_ids])
        original=torch.full((len(proposal.scores),),-1,dtype=torch.long)
        for row,col in proposal.pairs.items():
            if float(proposal.scores[row,col])>legacy_acceptance_threshold(base,int(lengths[col]),bool(model.not_mult_thresh)):original[row]=int(proposal.track_ids[col])
        traces=[]
        class Recording(JEVRuntimePolicy):
            def decide(self,*a,**kw):
                d=super().decide(*a,**kw);traces.append(d);return d
        tracker=make_tracker(Recording('jev',lab.controller))
        tracker.with_bank=bool(model.with_bank);tracker.with_iou=bool(model.with_iou);tracker.not_mult_thresh=bool(model.not_mult_thresh)
        rows=list(proposal.pairs);cols=[proposal.pairs[r]for r in rows]
        direct=NativeMatchResolver.native_call(tracker,original,proposal,rows,cols,base,payload,state,lengths)
        first=[d for d in traces if d.context.get('proposal_round')!=2]
        actions={int(d.context['detection_index']):d.committed_action for d in first}
        lab.decision_rows=[{'question':'MATCH_DECISION','action':d.committed_action,'off_action':d.off_action,
                           'context':dict(d.context),'feature_vector':list(d.state_features),'probabilities':dict(d.probabilities)}for d in first]
        actor_equal=all(actor.decide(torch.tensor(d.state_features),'MATCH_DECISION',list(d.legal_actions),off_action=d.off_action).committed_action==d.committed_action for d in first)
        old=NativeMatchResolver(lab.engine,lab.controller);new=AttributionResolver(lab,actor)
        a=old.resolve(payload,state,actions=actions,proposal=proposal);b=new.resolve(payload,state,actions=actions,proposal=proposal)
        assert a['existing_track_ids']==b['existing_track_ids'] and actor_equal
        ids=[b['existing_track_ids'][r] if b['existing_track_ids'][r]is not None else -1 for r in range(len(proposal.scores))]
        assert ids==direct.tolist()
        memory={r:'WRITE_MEMORY'for r,t in a['existing_track_ids'].items()if t is not None}
        fingerprints=[];commits=[]
        for resolver in (old,new):
            lab.engine.resolve_actions=resolver.resolve
            branch=state.clone();res=lab.original_step(payload,branch,actions=actions,proposal=proposal,memory_actions=memory)
            fingerprints.append(lab.state_fingerprint(branch,{}));commits.append(res['committed_track_ids'])
        lab.engine.resolve_actions=old.original
        assert fingerprints[0]==fingerprints[1] and commits[0]==commits[1]
        second=[d for d in traces if d.context.get('proposal_round')==2]
        assert len(second)==len(new.validations)
        assert all(list(d.state_features)==v['feature_vector'] and d.committed_action==v['action']for d,v in zip(second,new.validations))
        records.append({'snapshot':str(path),'sha256':sha(path),'key':list(key),'actor_equal':actor_equal,
                        'native_commits_equal':True,'second_features_actions_equal':True,
                        'mutable_state_fingerprint_equal':True,'state_sha256':fingerprints[0],
                        'new_solver_calls':int(bool(new.last['second_validations'])),'second_validations':len(second)})
    assert len(records)==11
    save(REPORTS/'NATIVE_TRANSITION_PARITY.json',{'status':'PASS_ON_11_PRESERVED_SNAPSHOTS','binding':binding(),
         'events':records,'relative_REACT_hook':'NOT_IMPLEMENTED','full_native_three_question_parity':'NOT_ESTABLISHED',
         'complete_heldout_runtime_gate':'PENDING','old_sources_rewritten':False})
    print(json.dumps({'status':'PASS_ON_11_PRESERVED_SNAPSHOTS','events':len(records)}));protect()

if __name__=='__main__':main()
