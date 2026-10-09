"""Do not turn changed galleries or observational stale pools into fake labels."""
from jev_phase12_common import *

def main():
    protect();native=[];events=[];stale=[]
    for video in TRAIN:
        path=OUT/'lifecycle_native_v1'/f'video{video:02d}/RESULT.json';r=json.loads(path.read_text());assert r['status']=='COMPLETE';native.append({'video':video,'path':str(path),'SHA256':sha(path),'binding':r['binding']});events.extend(r['events'])
        for event in r['events']:
            stale.extend({'intervention_key':event['key'],**q} for q in event['branches']['WRITE']['stale_questions'])
    assert len(events)==8
    compact=[{k:v for k,v in e.items() if k!='branches'} for e in events]
    report={'status':'COMPLETE','protocol_SHA256':sha(REPORTS/'LIFECYCLE_AUDIT_PROTOCOL.json'),'native_runs':native,'MEMORY':{'status':'NO_GO_INSUFFICIENT_IDENTIFIABLE_SUPERVISION','genuine_WRITE_KEEP_events':8,'independent_TRAIN_videos':TRAIN,'Gallery_changed':sum(e['Gallery_changed'] for e in events),'actual_target_bank_READ_events':sum(e['actual_bank_READ_count']>0 for e in events),'native_next_score_changes':sum(e['next_native_score_packets_changed']>0 for e in events),'native_committed_identity_changes':sum(bool(e['committed_ID_changed_keys']) for e in events),'non_tie_H32':sum(e['utility_WRITE_minus_KEEP']['32']!=0 for e in events),'non_tie_H64':sum(e['non_tie_H64'] for e in events),'events':compact,'causal_scope':'single genuine native observation WRITE versus KEEP, same actual pre-action prefix/RNG and GMT_OFF future. 8 early predeclared observations; not proof that memory never matters. No unchanged/tie branch is fabricated as a correctness negative.','TRAIN_qualified_groups':sum(e['non_tie_H64'] for e in events),'VALIDATION_qualified_groups':0,'runtime':'UNTRAINED/FROZEN_FALLBACK'},'REACTIVATION':{'status':'NO_GO_NO_NATIVE_CAUSAL_ACTION_LABEL_DATASET','observed_actual_stale_questions':stale,'observed_questions':len(stale),'observed_pools_with_at_least2_candidates':sum(len(q['candidate_ids'])>=2 for q in stale),'observed_pools_with_at_least2_reliably_anchored_candidates':sum(sum(v is not None for v in q['anchor_GT'].values())>=2 for q in stale),'actual_stale_refs_only':True,'executed_alternative_recovery_utility_forks':0,'verified_correct_and_wrong_recovery_supervision':None,'reason':'Actual stale pool observations and frozen rule outcomes do not provide verified opposite-action causal supervision or independent TRAIN/VAL data. No fake stale identity or synthetic label is introduced.','runtime':'UNTRAINED/FROZEN_FALLBACK'},'MATCH':{'status':'QUALIFIED_NATIVE_PARTIAL_SUPERVISION','TRAIN':132,'VALIDATION':94,'verified_corrections_TRAIN':74,'verified_corrections_VALIDATION':59},'SHARED_MULTI_QUESTION_TRAINING':{'status':'NOT_RUN','metrics':None,'reason':'MEMORY and REACTIVATION data qualification unmet; architecture has real typed forward interfaces, not three trained policies'},'FULL_LIFECYCLE_ONLINE':{'status':'NOT_RUN','metrics':None,'reason':'Unqualified lifecycle tasks; MATCH_ONLY preserves original native memory/bank rules'},'historical_memory_audits_unchanged':True,'heldout':'SEALED','Full24':False,'official_TEST':False}
    followup=REPORTS/'LIFECYCLE_VISUAL_FUTURE_PROTOCOL.json'
    if followup.exists():
        previous=REPORTS/'LIFECYCLE_DATA_ELIGIBILITY_V1.json'
        if not previous.exists():previous.write_bytes((REPORTS/'LIFECYCLE_DATA_ELIGIBILITY.json').read_bytes())
        visual_events=[];visual_runs=[]
        for video in TRAIN:
            path=OUT/'lifecycle_visual_future_v1'/f'video{video:02d}/RESULT.json';r=json.loads(path.read_text());assert r['status']=='COMPLETE'
            visual_runs.append({'video':video,'path':str(path),'SHA256':sha(path),'binding':r['binding'],'case':r['future_case']});visual_events.extend(r['events'])
        assert len(visual_events)==8 and [e['key'] for e in visual_events]==[e['key'] for e in events]
        qualified=sum(e['non_tie_H64'] and e['reliable_prefix_anchor'] is not None for e in visual_events)
        visual={'status':'COMPLETE','protocol_SHA256':sha(followup),'native_runs':visual_runs,'genuine_same8_events':8,'Gallery_changed':sum(e['Gallery_changed'] for e in visual_events),'actual_visual_MATCH_READ_events':sum(e['actual_visual_MATCH_READ_count']>0 for e in visual_events),'changed_visual_MATCH_READ_events':sum(e['visual_MATCH_READ_changed'] for e in visual_events),'changed_policy_value_events':sum(e['policy_value_packets_changed']>0 for e in visual_events),'changed_native_score_events':sum(e['next_native_score_packets_changed']>0 for e in visual_events),'native_committed_identity_changes':sum(bool(e['committed_ID_changed_keys']) for e in visual_events),'non_tie_H32':sum(e['utility_WRITE_minus_KEEP']['32']!=0 for e in visual_events),'non_tie_H64':sum(e['non_tie_H64'] for e in visual_events),'reliably_anchored_non_tie_H64':qualified,'VALIDATION_qualified_groups':0,'events':[{k:v for k,v in e.items() if k!='branches'} for e in visual_events],'scope':'Current MATCH fixed GMT_OFF to preserve genuine selected WRITE, same pre-action native/RNG prefix; next payload onward frozen Full seed20261008 MATCH_ONLY. Conditional consequence audit, not8 new independent groups or evidence of learned MEMORY.'}
        save(REPORTS/'LIFECYCLE_VISUAL_FUTURE_AUDIT.json',visual)
        report['MEMORY']['visual_future_followup']=visual
        report['MEMORY']['TRAIN_qualified_groups_for_visual_future']=qualified
        report['MEMORY']['qualification_reason']='Original GMT_OFF audit ties do not identify effects under a Gallery-reading visual future. Same8 bounded followup events, at most8 TRAIN groups and0VAL, cannot meet preregistered12TRAIN/4VAL minimum; no MEMORY labels fabricated or shared training authorized.'
        report['original_lifecycle_eligibility_SHA256']=sha(previous)
    save(REPORTS/'LIFECYCLE_DATA_ELIGIBILITY.json',report);print('LIFECYCLE_QUALIFICATION_PUBLISHED',flush=True)
if __name__=='__main__':main()
