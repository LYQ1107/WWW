"""Frozen-rule paired outcomes on the authorized expanded TRAIN scope."""
from jev_phase19_common import *
from jev_phase18_outcomes import compare, coverage
import collections
import gzip


def verify_whole_census_protocol(events, protocol):
    """Reproduce the frozen owner-conflict graph including unsampled bridges."""
    parent={}
    def find(x):
        parent.setdefault(x,x)
        if parent[x]!=x:parent[x]=find(parent[x])
        return parent[x]
    def union(a,b):
        a=find(a);b=find(b)
        if a!=b:parent[max(a,b)]=min(a,b)
    for video in SCOPE:
        census=read(census_case(video)/'RESULT.json')
        source=census['events'];assert sha(source['path'])==source['SHA256']
        with gzip.open(source['path'],'rt') as f:
            for line in f:
                e=json.loads(line);gt=e['GT_OFFLINE']
                if gt is None:continue
                a=(video,gt);find(a)
                conflicts=list(e['numeric_close_owner_conflict_OFFLINE'])
                if e['wrong_owner'] and e['owner_before_commit_OFFLINE'] is not None:
                    conflicts.append(e['owner_before_commit_OFFLINE'])
                for other in conflicts:union(a,(video,other))
    reproduced={}
    for e in events:
        key=str(find((e['event_key'][0],e['GT_OFFLINE'])))
        reproduced.setdefault(key,[]).append(e['selection_index'])
    assert reproduced==protocol['qualification_components'],(reproduced,protocol['qualification_components'])
    return True


def qualify(protocol, events, safe):
    units=[]
    for component,members in protocol['qualification_components'].items():
        splits={events[i]['split'] for i in members}
        matched=sorted(set(members)&safe)
        units.append(dict(component=component,members=members,
            mixed_splits=len(splits)>1,safe_event_members=matched,
            qualified_train=bool(matched) and splits=={'TRAIN'},
            qualified_reserved=bool(matched) and splits=={'RESERVED'}))
    return units


def main():
    protect()
    events=read(OUT/'selection/INTERVENTION_EVENTS.json')
    sampled=read(OUT/'selection/SAMPLED_EVENTS.json')
    selected_protocol=read(REPORTS/'INDEPENDENCE_QUALIFICATION_PROTOCOL.json')
    whole_protocol=read(REPORTS/'WHOLE_CENSUS_DEPENDENCY_PROTOCOL.json')
    assert whole_protocol['original_intervention_events_SHA256']==sha(OUT/'selection/INTERVENTION_EVENTS.json')
    assert verify_whole_census_protocol(events,whole_protocol)
    expected=sum(1+len(e['alternatives_GT_free']) for e in events)
    assert len(events)==16 and expected==66
    queue=read(OUT/'intervention_queue_v1/RESULT.json')
    assert queue['status']=='COMPLETE' and not queue['failed']
    sources=[];lookup={};payloads=0
    for video in sorted({e['event_key'][0] for e in events}):
        p=OUT/'interventions'/f'video{video:02d}/RESULT.json';v=read(p)
        assert v['status']=='COMPLETE' and v['forks_and_restores_exact']
        assert v['binding']['source_commit']==queue['binding']['source_commit']
        sources.append(ref(p));payloads+=v['full_original_replay_verified_payloads']
        for s in v['branches']:
            assert sha(s['path'])==s['SHA256'];b=read(s['path'])
            assert b['status']=='COMPLETE' and b['actual_mutated_state'] and b['first_scores_exact']
            assert b['GT_actor_input'] is False
            for f in ['owner_trace','commits']:
                assert sha(b[f]['path'])==b[f]['SHA256']
            for h in [16,32,64]:
                x=b['H'][str(h)]
                assert x['scene_frames_observed']==h and x['matching_history_retained_from_frame_zero']
                assert x['evaluator']['strict_clear_causal_gate_eligible']
                assert not x['evaluator']['duplicate_gt_diagnostic']
                assert x['future_CLEAR_IDSW'] is not None
            if b['variant']!='BASE':
                assert b['constraint']['legal_masks_unchanged'] and b['constraint']['scores_unchanged']
                assert b['constraint']['native_capacity_feasibility_verified']
                assert b['constraint']['second_scoring_validation_present'] is False
            k=tuple(b['event_key'])+(b['variant'],);assert k not in lookup;lookup[k]=b
    assert len(lookup)==expected
    pairs=[];groups=[]
    for e in events:
        key=tuple(e['event_key']);base=lookup[key+('BASE',)]
        for n,r in enumerate(e['alternatives_GT_free']):
            probe=lookup[key+(f'ALT{n}',)];assert probe['reference']==r
            pairs.append(compare(e,base,probe))
        ps=[p for p in pairs if p['selection_index']==e['selection_index']]
        groups.append(dict(selection_index=e['selection_index'],event_key=e['event_key'],split=e['split'],
            category=e['sampled_stratum'],correct_owner_in_alternatives=e['alternative_owner_coverage_OFFLINE'],
            safe_variants=[p['variant'] for p in ps if p['safe_correction']],
            harmful_variants=[p['variant'] for p in ps if p['harm_reasons']],
            real_correction_variants=[p['variant'] for p in ps if p['real_correction']]))
    safe={g['selection_index'] for g in groups if g['safe_variants']}
    units=qualify(whole_protocol,events,safe)
    old_units=qualify(selected_protocol,events,safe)
    nt=sum(u['qualified_train'] for u in units);nr=sum(u['qualified_reserved'] for u in units)
    summary=dict(groups=len(events),sampled_groups=len(sampled),baselines=len(events),alternatives=len(pairs),
        total_complete_native_branches=len(lookup),full_original_replay_verified_payloads=payloads,
        safe_temporal_representatives=len(safe),safe_action_variants=sum(p['safe_correction'] for p in pairs),
        real_correction_action_variants=sum(p['real_correction'] for p in pairs),
        harmful_action_variants=sum(bool(p['harm_reasons']) for p in pairs),
        correction_and_harm_action_variants=sum(p['real_correction'] and bool(p['harm_reasons']) for p in pairs),
        owner_only_safe_action_variants=sum(p['owner_only_safe_diagnostic'] for p in pairs),
        strict_CLEAR_unevaluable_action_variants=sum(not p['strict_CLEAR_all_H_evaluable'] for p in pairs),
        independent_qualification_units=len(units),independent_safe_TRAIN_groups=nt,independent_safe_RESERVED_groups=nr,
        required_TRAIN=8,required_RESERVED=3,
        learning_gate='PASS_DATA_COUNT_ONLY' if nt>=8 and nr>=3 else 'NO_GO_INDEPENDENCE_OR_SAFETY',
        new_training='NOT_RUN',new_policy_performance=None,full_cold_online_new_policy='NOT_RUN',new_HOTA_AssA_IDF1_MOTA=None)
    by_video={}
    for v in sorted({e['event_key'][0] for e in events}):
        ps=[p for p in pairs if p['event_key'][0]==v]
        by_video[str(v)]=dict(alternatives=len(ps),safe=sum(p['safe_correction'] for p in ps),
            corrective=sum(p['real_correction'] for p in ps),harmful=sum(bool(p['harm_reasons']) for p in ps),
            H64_N01_rows=sum(p['H']['64']['transitions'].get('N01',0) for p in ps),
            H64_N10_rows=sum(p['H']['64']['transitions'].get('N10',0) for p in ps))
    save(REPORTS/'NATIVE_PAIRED_OWNER_OUTCOMES.json',dict(status='COMPLETE_BOUNDED_DIAGNOSTIC',binding=binding(),
        sources=sources,queue=ref(OUT/'intervention_queue_v1/RESULT.json'),
        rules=ref(ROOT/'reports/JEV_PHASE18/OUTCOME_QUALIFICATION_RULES.json'),
        outcome_comparator=ref(ROOT/'reproduction_tools/jev_phase18_outcomes.py'),summary=summary,
        groups=groups,qualification_units=units,selected_event_only_units_diagnostic=old_units,
        qualification_protocol=ref(REPORTS/'WHOLE_CENSUS_DEPENDENCY_PROTOCOL.json'),per_video=by_video,pairs=pairs,
        scientific_limitations=[
            'Executed action values are offline diagnostics, not a trained or deployable controller.',
            'Whole-census dependency screen frozen before interventions; cross-time bridges merged. No independence/significance proof.',
            'Actions, horizons and observed rows overlap; row sums across alternatives are descriptive, not independent gains.',
            'Reserved10/11/15 are this-round validation; historical exposure prevents a fresh-test claim.',
            'UNKNOWN is not correction; earliest past-only owner is an observer estimate, not perfect identity evidence.',
            'Full-video raw GT duplicates remain in02/03/15; selected prefixes/futures are strict and unique.',
            'Active MATCH availability does not prove absence from Bank or full history.',
            'Frozen native adapter has no second MATCH scoring validator; no new training or new-policy full cold run.']))
    save(REPORTS/'CANDIDATE_ACTION_COVERAGE.json',dict(status='COMPLETE',
        sampled_events=ref(OUT/'selection/SAMPLED_EVENTS.json'),intervention_events=ref(OUT/'selection/INTERVENTION_EVENTS.json'),
        sampled=coverage(sampled),intervened=coverage(events),
        availability_scope='legal active MATCH; actual REACT queries separately recorded; counterfactual Bank pool on accepted MATCH not observed',
        unknown_correct_owner_availability=None,no_top16_truncation=True,unexecuted_action_values=None,GT_only_offline=True))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
