"""Compare actual frozen-B2 reassociation transitions with the native hook."""
import argparse
import json
from pathlib import Path

import torch

from jev_phase6_common import OUT,REPORTS,save,sha,protect_anchor
from jev_phase6_rollouts import NativeReplayLab
from gtr.modeling.jev_runtime import JEVRuntimePolicy
from gtr.modeling.jev_state import count_track_history,legacy_acceptance_threshold,association_window_length
from test_constrained_assignment import make_tracker


def main(device):
    lab=NativeReplayLab(1,device)
    source={}
    for line in (OUT/'gating_tracking/G5/online_decisions.jsonl').open():
        r=json.loads(line);c=r['context']
        if r['question']=='MATCH_DECISION':source[(c['frame'],c['view'],c['detection_index'])]=r
    records=[]
    for path in sorted((OUT/'reassociate/snapshots').glob('*.pth')):
        c=torch.load(path,map_location='cpu');state=c['state'];key=c['key'];payload=lab.cache.load(*key[:3])
        proposal=lab.original_propose(payload,state);scores=proposal.scores;ids=proposal.track_ids
        model=lab.engine.association_fn.model;threshold=float(model.overlap_thresh)
        lengths=torch.tensor([count_track_history(state.association_history,t) for t in ids])
        original=torch.full((len(scores),),-1,dtype=torch.long)
        for row,col in proposal.pairs.items():
            if float(scores[row,col])>legacy_acceptance_threshold(threshold,int(lengths[col]),bool(model.not_mult_thresh)):
                original[row]=int(ids[col])
        traces=[]
        class RecordingPolicy(JEVRuntimePolicy):
            def decide(self,*args,**kwargs):
                d=super().decide(*args,**kwargs);traces.append(d);return d
        tracker=make_tracker(RecordingPolicy('jev',lab.controller))
        tracker.with_bank=bool(model.with_bank);tracker.with_iou=bool(model.with_iou);tracker.not_mult_thresh=bool(model.not_mult_thresh)
        rows=list(proposal.pairs);cols=[proposal.pairs[r] for r in rows]
        native=tracker._apply_jev_match_decisions(original,scores,torch.tensor(ids),rows,cols,threshold,
            view=int(key[2]),frame_index=int(key[1]),window_length=association_window_length(
                history_instances=len(state.association_history),view_num=2,view_index=int(key[2])),
            track_lengths=lengths,detection_boxes=payload['pred_boxes'],detection_scores=payload['detection_scores'],detection_image_size=payload['image_size'])
        first=[d for d in traces if d.context.get('decision_scope')=='match'];second=[d for d in traces if d.context.get('proposal_round')==2]
        features=[];actions={}
        for d in first:
            row=int(d.context['detection_index']);ref=source[(key[1],key[2],row)]
            error=float((torch.tensor(d.state_features)-torch.tensor(ref['feature_vector'])).abs().max())
            features.append(error);actions[row]=d.committed_action
            if d.committed_action!=ref['action']:raise AssertionError('native first-round B2 action mismatch; isolate feature contract before commit audit')
        resolution=lab.engine.resolve_actions(payload,state,actions=actions,proposal=proposal)
        result=lab.original_step(payload,state.clone(),actions=actions,proposal=proposal)
        research=[t if t<=state.next_id else -1 for row,t in sorted(result['committed_track_ids'].items())]
        native_ids=native.tolist();memory_research={r:t for r,t in resolution['existing_track_ids'].items() if t is not None}
        memory_native={r:t for r,t in enumerate(native_ids) if t>=0}
        from jev_phase6_native_match import NativeMatchResolver
        resolver=NativeMatchResolver(lab.engine,lab.controller)
        corrected=resolver.resolve(payload,state,actions=actions,proposal=proposal)
        lab.engine.resolve_actions=resolver.resolve
        try:corrected_step=lab.original_step(payload,state.clone(),actions=actions,proposal=proposal)
        finally:lab.engine.resolve_actions=resolver.original
        corrected_ids=[t if t<=state.next_id else -1 for row,t in sorted(corrected_step['committed_track_ids'].items())]
        corrected_memory={r:t for r,t in corrected['existing_track_ids'].items() if t is not None}
        records.append({'snapshot':path.name,'key':list(key),'snapshot_sha256':sha(path),
            'first_round_actions_match_frozen_B2':True,'first_round_max_feature_error':max(features,default=0),
            'native_second_round_decisions':[{'row':int(d.context['detection_index']),'action':d.committed_action,'off_action':d.off_action,'track':d.context['proposal_track_id']} for d in second],
            'research_existing_commits':research,'native_existing_commits':native_ids,
            'commit_mismatch_rows':[r for r,(a,b) in enumerate(zip(research,native_ids)) if a!=b],
            'research_MEMORY_eligible':memory_research,'native_MEMORY_eligible':memory_native,
            'MEMORY_eligibility_identical':memory_research==memory_native,
            'corrected_native_replay_commits':corrected_ids,
            'corrected_native_replay_commits_identical':corrected_ids==native_ids,
            'corrected_native_replay_MEMORY_identical':corrected_memory==memory_native})
    matched=all(not r['commit_mismatch_rows'] and r['MEMORY_eligibility_identical'] for r in records)
    report={'status':'PASS_ON_11_SNAPSHOTS' if matched else 'FAIL','events':records,
        'commit_mismatch_events':sum(bool(r['commit_mismatch_rows']) for r in records),
        'MEMORY_eligibility_mismatch_events':sum(not r['MEMORY_eligibility_identical'] for r in records),
        'source_native':'GTRRCNN._apply_jev_match_decisions performs binary learned second-round validation',
        'source_research':'resolve_actions applies legacy threshold only to MEMORY eligibility; step commits constrained assignment without native second-round controller call',
        'canonical_relative_reactivation_native_hook':'NOT_IMPLEMENTED','full_native_lifecycle_contract':'NOT_ESTABLISHED',
        'B2_original_checkpoint_and_research_results_unchanged':True,'official_test_read':False,
        'opt_in_corrected_native_MATCH_contract':'PASS_ON_11_EVENTS' if all(r['corrected_native_replay_commits_identical'] and r['corrected_native_replay_MEMORY_identical'] for r in records) else 'FAIL',
        'what_did_we_learn':'Frozen research SHA reproduction is not sufficient to establish native lifecycle transition equivalence; inspect commit and MEMORY eligibility separately.'}
    save(REPORTS/'NATIVE_TRANSITION_CONTRACT_AUDIT.json',report);protect_anchor();print(json.dumps({k:v for k,v in report.items() if k!='events'}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:0');main(p.parse_args().device)
