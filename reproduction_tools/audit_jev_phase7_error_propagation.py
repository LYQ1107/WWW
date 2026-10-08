"""Available native MATCH->MEMORY observations, without claiming mediation/Unified."""
from collections import Counter
import json
import torch
from jev_phase7_common import *
from jev_phase7_offline import IdentityEvaluator
from jev_phase6_offline_utility import prefix_identity
from summarize_jev_phase7_paired import window_audit


def main():
    events=[];counts=Counter()
    for video in (7,6):
        source=OUT/'miniset_v1'/f'video{video:02d}/MINISET_RESULT.json';result=json.loads(source.read_text())
        evaluator=IdentityEvaluator(video);factual=evaluator.align(json.loads(Path(result['factual']['predictions']).read_text()))
        for index,record in enumerate(result['records']):
            snapshot=source.parent/'snapshots'/f'MATCH_{index:03d}.pth'
            capture=torch.load(snapshot,map_location='cpu');assert sha(snapshot)==record['snapshot_sha256']
            next_id=capture['state'].next_id
            if record['offline_only']['proposal_correct']is None:continue
            key=(video,record['frame'],record['camera']);mapping,_=prefix_identity(factual,key[1:]);target=record['offline_only']['GT'];row=record['detection_index']
            for action in ('ACCEPT_CURRENT','REASSOCIATE','START_NEW'):
                branch=record['branches'][action];first=branch['commits'][0]['committed_ids'];track=first.get(str(row),first.get(row))
                wrong_existing=track in mapping and mapping[track]!=target
                if not wrong_existing:continue
                decisions=json.loads(Path(branch['result']['decisions']).read_text());rows=evaluator.align(json.loads(Path(branch['result']['predictions']).read_text()))
                current_write=any(d['question']=='MEMORY_DECISION'and d['frame']==key[1]and d['view']==key[2]and d['row']==row and d['action']=='WRITE_MEMORY'for d in decisions)
                later_queries=[d for d in decisions if d['question']=='REACTIVATION_DECISION'and track in d.get('candidate_track_ids',[])]
                horizons={str(h):window_audit(rows,factual,key,h,next_id)[0]for h in (8,16,32)}
                event={'event_id':record['event_id'],'sequence':video,'action':action,'offline_target_GT':target,
                    'incorrect_existing_identity':track,'prefix_identity_GT':mapping[track],
                    'actual_same_payload_MEMORY_WRITE_on_wrong_association':current_write,
                    'later_reactivation_query_rows_including_edited_identity':len(later_queries),'true_temporal_horizons':horizons,
                    'source_prediction_sha256':branch['prediction_sha256'],'complete_native_state_diverged':branch['fingerprints']!=record['branches']['CONTROL']['fingerprints']}
                events.append(event);counts.update(wrong_existing_commits=1,immediate_MEMORY_WRITE=int(current_write),
                                                   later_REACT_query_exposures=int(bool(later_queries)))
    audit={'status':'COMPLETE_AVAILABLE_NATIVE_MATCH_MEMORY_OBSERVATION','events':events,'counts':dict(counts),
        'scope':'posthoc descriptive audit of already executed bounded legal interventions; no new label/fit/sequence selection',
        'causal_limit':'MATCH was intervened; subsequent MEMORY writes are observed consequences. Bank-mediated REACT effects are not isolated by a lifecycle factorial; absent relative production hook and supervision gates block that claim',
        'cross_camera_definition':'temporal errors reported separately per camera over common exogenous windows',
        'Unified_or_learned_lifecycle_advantage_demonstrated':False,'binding':binding(),'official_test_read':False}
    save(REPORTS/'IDENTITY_ERROR_PROPAGATION_AUDIT.json',audit);print(json.dumps({'status':audit['status'],'counts':dict(counts)}))

if __name__=='__main__':main()
