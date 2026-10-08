"""Common-window MEMORY mechanism and actual target-READ attribution."""
from collections import Counter
import json
from jev_phase7_common import *
from jev_phase7_offline import IdentityEvaluator


def main():
    output=[];overall=Counter()
    for video in (24,23):
        root=OUT/'memory_followup'/f'video{video:02d}';audit=json.loads((root/'MEMORY_FOLLOWUP_AUDIT.json').read_text());assert audit['status']=='COMPLETE'
        old=Path('/home/liuyeqiang/WWW_jev_phase6_runtime/20261008/lifecycle_native')/f'video{video:02d}'
        evaluator=IdentityEvaluator(video);baseline=json.loads((old/'baseline/result.json').read_text())
        factual=evaluator.align(json.loads(Path(baseline['predictions']).read_text()))
        for record in audit['records']:
            w=record['branches']['WRITE_MEMORY'];s=record['branches']['SKIP_MEMORY'];target=record['offline_target_GT']
            wr={tuple(r['key']):r for r in w['reads']};sr={tuple(r['key']):r for r in s['reads']};shared=sorted(set(wr)&set(sr))
            assert set(wr)==set(sr)
            proto_changes=score_changes=target_queries=known_other=unknown_queries=0;maximum_delta=maximum_score=0.
            for key in shared:
                a=wr[key];b=sr[key];assert a['candidate_ids']==b['candidate_ids']and a['query_rows']==b['query_rows']
                proto_changes+=a['prototype_sha256']!=b['prototype_sha256'];score_changes+=a['target_column_scores']!=b['target_column_scores']
                for row,wa,sa in zip(a['query_rows'],a['target_column_scores'],b['target_column_scores']):
                    gt=factual.get((key[1],key[2],row),{}).get('gt')
                    target_queries+=gt==target;known_other+=gt is not None and gt!=target;unknown_queries+=gt is None
                    maximum_delta=max(maximum_delta,abs(wa-sa));maximum_score=max(maximum_score,wa,sa)
            gallery_changed=w['states'][0]['gallery_sha256']!=s['states'][0]['gallery_sha256']
            native_state_changed=w['states']!=s['states'];commits_same=w['commits']==s['commits']
            prediction_same=w['prediction_sha256']==s['prediction_sha256']
            d={'video':video,'key':record['key'],'known_target_GT':target,'immediate_gallery_changed':gallery_changed,
                'immediate_bank_prototype_changed':w['states'][0]['prototype_sha256']!=s['states'][0]['prototype_sha256'],
                'complete_mutable_state_diverged':native_state_changed,'first_READ_frame_WRITE':w['first_read_frame'],'first_READ_frame_SKIP':s['first_read_frame'],
                'shared_READ_payloads':len(shared),'READ_prototype_changed_payloads':proto_changes,'READ_target_column_score_changed_payloads':score_changes,
                'actual_target_GT_query_observations':target_queries,'target_identity_READ_right_censored_in_H64':target_queries==0,'other_known_query_observations':known_other,'unknown_GT_query_observations':unknown_queries,
                'maximum_absolute_candidate_score_delta':maximum_delta,'maximum_edited_candidate_score':maximum_score,
                'current_native_bank_threshold':.4,'committed_ids_identical':commits_same,'raw_predictions_identical':prediction_same,
                'utility_ties_H8_H16_H32_H64':record['utility_ties'],'video_end_truncated':w['truncated']or s['truncated'],
                'CONTROL_WRITE_complete_state_parity':True,'source_audit_sha256':sha(root/'MEMORY_FOLLOWUP_AUDIT.json')}
            output.append(d);overall.update(events=1,gallery_changed=int(gallery_changed),state_diverged=int(native_state_changed),
                READ_prototype_changed_events=int(proto_changes>0),READ_score_changed_events=int(score_changes>0),
                actual_target_READ_events=int(target_queries>0),target_identity_READ_right_censored_events=int(target_queries==0),committed_id_divergence=int(not commits_same),prediction_divergence=int(not prediction_same),
                H64_utility_ties=int(record['utility_ties']['64']),shared_READ_payloads=len(shared))
    assert overall['H64_utility_ties']==8 and overall['committed_id_divergence']==0
    result={'status':'BLOCKED_MEMORY_IDENTIFIABILITY','execution_status':'COMPLETE','counts':dict(overall),'records':output,
        'shared_horizons':[8,16,32,64],'effective_informative_sample_size':0,'selection':'first four known-target READ-enriched preserved sources per video; diagnostic, not population rate',
        'causal_interpretation':'single WRITE changes gallery, later READ prototype and candidate scores, but never changes committed identities or fixed-window utility in these eight events; immediate persistent bank prototype may update only at promotion',
        'single_write_dilution':'native bank uses latest10-vector mean; a mature-gallery single observation has roughly 1/10 weight; source rule, not empirical universal effect size',
        'learned_MEMORY_training_started':False,'WHAT_DID_WE_LEARN':'H8/H16/H32 being too short is not sufficient explanation: same H64 windows still tie despite real bank reads and score differences; six lack target-identity queries, two expose target queries without identity consequences',
        'binding':binding(),'official_test_read':False}
    save(REPORTS/'MEMORY_H64_IDENTIFIABILITY_AUDIT.json',result);protect();print(json.dumps({'status':result['status'],'counts':dict(overall)}))

if __name__=='__main__':main()
