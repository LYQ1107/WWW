"""Aggregate frozen admission and native factorial without tuning or promotion."""
from jev_phase16_common import *
import collections
import gzip
from types import SimpleNamespace
from jev_phase16_memory_risk import MemoryRisk


def scalar(result, horizon):
    h=result['H'][horizon];c=h['identity_consequences'];counts=c['counts']
    return dict(future_CLEAR_IDSW=h['future_CLEAR_IDSW'],
        new_cross_GT_gallery_mix=counts.get('new_cross_GT_gallery_mix',0),
        wrong_owner_observations=c['wrong_owner_observed_frames'],
        wrong_owner_episode_max_frames=c['wrong_owner_episode_max_frames'],
        false_split_birth=counts.get('false_split_birth',0),
        false_birth_seen_GT_proxy=counts.get('false_birth_seen_GT_proxy',0),
        owner_correct_observations=counts.get('owner_correct',0),
        writes_into_polluted_raw_gallery=counts.get('already_polluted_write',0))


def main():
    protect();paths=[];groups={}
    for arm in 'ABCD':
        for video in TRAIN:
            path=OUT/'P3_factorial'/arm/f'video{video:02d}/RESULT.json';data=read(path)
            assert data['status']=='COMPLETE';paths.append(ref(path))
            for r in data['cases']:groups.setdefault((video,tuple(r['key']),r['row']),{})[arm]=r
    assert len(groups)==24 and all(set(g)==set('ABCD') for g in groups.values())
    comparisons=[];violations=collections.defaultdict(list);benefits=collections.defaultdict(set)
    totals=collections.defaultdict(collections.Counter);safe_labels=set();unsafe_labels=set();normal=set()
    reserved_safe=set();reserved_unsafe=set();all24_parity=True;independent=set()
    safety_fields=['new_cross_GT_gallery_mix','wrong_owner_observations','wrong_owner_episode_max_frames','false_split_birth']
    for (video,key,row),g in groups.items():
        assert len({r['original_state_SHA256'] for r in g.values()})==1
        assert len({r['derived_state_SHA256'] for r in g.values()})==1
        assert len({r['actual_RNG_SHA256'] for r in g.values()})==1
        assert len({r['intervention']['first_original_batch_SHA256'] for r in g.values()})==1,'first inputs differ across memory arms'
        assert len({r['intervention']['first_original_logits_SHA256'] for r in g.values()})==1
        assert g['A']['native_original_parity']=='ALL_XV_V3_H32_COMMITTED_IDS_EXACT'
        all_traces={}
        for arm,r in g.items():
            with gzip.open(r['trace']['path'],'rt') as stream:all_traces[arm]=list(map(json.loads,stream))
            for h in r['H']:
                original=r['H'][h]['identity_consequences']
                corrected=MemoryRisk.horizon(SimpleNamespace(strong_trace=all_traces[arm]),tuple(key[1:]),key[1]+int(h))
                assert corrected['counts']==original['counts']
                assert corrected['wrong_owner_observed_frames']==original['wrong_owner_observed_frames']
                assert corrected['wrong_owner_episode_max_frames']==original['wrong_owner_episode_max_frames']
                r['H'][h]['identity_consequences']=corrected
        first=next(r for r in all_traces['A'] if r['key']==list(key[1:]) and r['row']==row)
        gt=first['GT_OFFLINE_ONLY'];cluster=(video,gt,key[2],key[1]//64);strict=video!=14
        if strict and gt is not None:independent.add(cluster)
        horizons={};unsafe_arm={arm:False for arm in 'BCD'};benefit_arm={arm:False for arm in 'BCD'}
        for h in g['A']['H']:
            values={arm:scalar(r,h) for arm,r in g.items()}
            deltas={name:{f:values[a][f]-values[b][f] for f in values[a]} for name,a,b in [
                ('B_minus_A_memory','B','A'),('C_minus_A_first_reader','C','A'),('D_minus_A_joint','D','A'),
                ('D_minus_C_memory_given_reader','D','C'),('D_minus_B_reader_given_memory','D','B')]}
            interaction={f:values['D'][f]-values['C'][f]-values['B'][f]+values['A'][f] for f in values['A']}
            durations={};recovery={}
            for arm,r in g.items():
                c=r['H'][h]['identity_consequences'];per={}
                for e in c['error_propagation_intervals']:
                    tag=str(e['GT_OFFLINE_ONLY'])+'_camera'+str(e['view']);per[tag]=max(per.get(tag,0),e['observed_wrong_frames'])
                durations[arm]=per
                correct=next((e for e in all_traces[arm] if e['GT_OFFLINE_ONLY']==gt and e['key'][1]==key[2] and e['owner_correct'] and key[1]<=e['key'][0]<key[1]+int(h)),None)
                recovery[arm]=dict(first_wrong_in_A=first['wrong_owner'],scene_frames_to_first_observed_correct_owner=correct['key'][0]-key[1] if first['wrong_owner'] and correct else None,
                    unavailable_or_right_censored=not first['wrong_owner'] or correct is None)
            horizons[h]=dict(actual=values,paired_deltas=deltas,interaction_D_minus_C_minus_B_plus_A=interaction,
                per_GT_camera_wrong_owner_max_episode=durations,target_recovery_latency=recovery,
                corrected_observed_propagation={a:r['H'][h]['identity_consequences'] for a,r in g.items()})
            for arm in 'ABCD':
                if strict:totals[arm+'_H'+h].update(values[arm])
            for arm in 'BCD':
                delta={f:values[arm][f]-values['A'][f] for f in values[arm]}
                worsening={f:v for f,v in delta.items() if f in safety_fields and v>0}
                episode_worse={tag:n-durations['A'].get(tag,0) for tag,n in durations[arm].items() if n>durations['A'].get(tag,0)}
                if episode_worse:worsening['per_GT_camera_duration_increases']=episode_worse
                improved=any(delta[f]<0 for f in ['new_cross_GT_gallery_mix','wrong_owner_observations'])
                unsafe_arm[arm] |= bool(worsening);benefit_arm[arm] |= improved
                if strict and worsening:violations[arm].append(dict(video=video,key=list(key),row=row,H=int(h),worsening=worsening,
                    future_IDSW_delta=delta['future_CLEAR_IDSW'],case=g[arm]['trace']))
        for arm in 'BCD':
            if strict and gt is not None and benefit_arm[arm] and not unsafe_arm[arm]:benefits[arm].add(cluster)
        if strict and gt is not None and first['owner_correct']:normal.add(cluster)
        # Recovery supervision requires an actual changed first decision, correct
        # immutable owner at that same detection and safe mutated future.
        for arm in 'CD':
            with gzip.open(g[arm]['trace']['path'],'rt') as stream:
                actual=next(r for r in map(json.loads,stream) if r['key']==list(key[1:]) and r['row']==row)
            changed=actual['identity']!=first['identity']
            safe_recovery=changed and actual['owner_correct'] and first['wrong_owner'] and benefit_arm[arm] and not unsafe_arm[arm]
            unsafe_recovery=changed and (actual['wrong_owner'] or unsafe_arm[arm])
            if strict and gt is not None:
                if safe_recovery:
                    safe_labels.add(cluster)
                    if key[1]//64%5==4:reserved_safe.add(cluster)
                if unsafe_recovery:
                    unsafe_labels.add(cluster)
                    if key[1]//64%5==4:reserved_unsafe.add(cluster)
        comparisons.append(dict(video=video,key=list(key),row=row,category=g['A']['category'],
            independent_cluster=list(cluster),strict_gate_eligible=strict,paired_H=horizons,
            any_worsening_vs_A=unsafe_arm,any_recovery_or_mixing_benefit_vs_A=benefit_arm,
            first_IDs={a:r['first_actual_ID'] for a,r in g.items()},arm_cases={a:r['trace'] for a,r in g.items()},
            identical_initial_native_state_and_RNG=True,identical_first_original_tensors_and_scores=True))
    src=binding(inputs=paths+[ref(REPORTS/'P2_P3_PROTOCOL.json')],evaluator='exact native paired outputs + unmodified TrackEval CLEAR + qualified ownership observer',
        scope='24 original overlapping TRAIN prefixes, 96 actual mutated futures; strict18, TRAIN14 diagnostic6; H64 predeclared8 prefixes')
    result=dict(status='COMPLETE',binding=src,cases=comparisons,all24_prefixes_retained=True,actual_native_branches=96,
        strict_prefixes=18,duplicate_diagnostic_prefixes=6,strict_independent_prefix_clusters=len(independent),
        strict_window_sums={k:dict(v) for k,v in totals.items()},
        window_sums_are_overlapping_descriptive_not_pooled_video_metrics=True,
        safety_violations_vs_A={a:violations[a] for a in 'BCD'},
        safe_benefit_clusters={a:[list(x) for x in sorted(benefits[a])] for a in 'BCD'},
        all_A_H32_IDs_exact_vs_preserved_XV=True,first_state_and_RNG_and_original_scores_equal=True,
        GT_or_future_online_inputs=False,failed_E_is_first_decision_negative_control=True,
        candidate_action_and_Hungarian_constructor_unchanged=True,future_histories_not_synchronized=True,
        offline_censoring_correction='Native traces, owner decisions and counts unchanged. Gap/horizon censor flags recomputed from closed traces; original wrong counts and episode maxima exact. Raw actor reports retained.')
    save(REPORTS/'MATCH_VS_MEMORY_CAUSAL_FACTORIAL.json',result)
    gate=dict(status='PASS' if len(benefits['B'])>=8 and not violations['B'] else 'NO_GO_SAFE_MEMORY',
        binding=src,selected_rule='bounded_quarantine',independent_safe_benefit_clusters=len(benefits['B']),
        required_clusters=8,strict_worsened_prefix_H_checks=len(violations['B']),
        safety_violations=violations['B'],P1_reader_qualified=False,V3_learning_qualified=False,
        major_scientific_versions_used=2,DEV_not_used_for_gate=True,thresholds_unchanged=True)
    save(REPORTS/'NATIVE_FUTURE_SAFETY_GO_NO_GO.json',gate)
    save(REPORTS/'RECOVERY_AND_CONTAMINATION_EFFECTS.json',dict(status='COMPLETE',binding=src,
        factorial=ref(REPORTS/'MATCH_VS_MEMORY_CAUSAL_FACTORIAL.json'),safety=gate,
        conditional_interactions_included=True,no_claim_of_dynamic_JEV_independent_value=True))
    parity=read(REPORTS/'NATIVE_MEMORY_PARITY.json');parity.update(broader_all24_parity='PASS_ALL24_A_H32_XV_V3_IDS_EXACT',
        all96_same_native_prefix_RNG_original_first_inputs=True,factorial=ref(REPORTS/'MATCH_VS_MEMORY_CAUSAL_FACTORIAL.json'))
    save(REPORTS/'NATIVE_MEMORY_PARITY.json',parity)
    label=dict(status='PASS' if len(safe_labels)>=8 and len(unsafe_labels)>=8 and len(reserved_safe)>=8 and normal else 'NO_GO_RECOVERY_LABEL_ELIGIBILITY',
        binding=src,safe_recovery_independent_clusters=len(safe_labels),unsafe_recovery_independent_clusters=len(unsafe_labels),
        reserved_safe_recovery_clusters=len(reserved_safe),reserved_unsafe_recovery_clusters=len(reserved_unsafe),
        necessary_correction_reserved_clusters=len(reserved_safe),normal_continuation_clusters=len(normal),
        safe_clusters=[list(x) for x in sorted(safe_labels)],unsafe_clusters=[list(x) for x in sorted(unsafe_labels)],
        reserved_rule='(scene_frame//64)%5==4; TRAIN-only frozen before outcomes',minimum_safe=8,minimum_unsafe=8,minimum_reserved_necessary_correction=8,
        semantics='Changed first native action with actual correct immutable owner, prior wrong owner, all-H safe future and benefit required for safe recovery; no unchanged action or shadow cosine support manufactured into recovery label',
        UNKNOWN_not_forced_wrong=True,GT_actor_inputs=False,Tiny_allowed=False,Pilot_allowed=False,formal_allowed=False,
        prerequisite_P1_failed=True,prerequisite_safe_memory_pass=gate['status']=='PASS')
    save(REPORTS/'LABEL_ELIGIBILITY.json',label)
    shadow_paths=[];shadow_groups={k:collections.Counter() for k in ['strict_TRAIN','TRAIN14_diagnostic','DEV_diagnostic']}
    for video in TRAIN+DEV:
        p=OUT/'P2_shadow'/f'video{video:02d}/RESULT.json';d=read(p);assert d['status']=='COMPLETE';shadow_paths.append(ref(p))
        group='TRAIN14_diagnostic' if video==14 else 'strict_TRAIN' if video in TRAIN else 'DEV_diagnostic'
        for rule,c in d['counts'].items():
            for field,v in c.items():
                if field.endswith('_peak'):shadow_groups[group][rule+'_'+field]=max(shadow_groups[group][rule+'_'+field],v)
                else:shadow_groups[group][rule+'_'+field]+=v
    save(REPORTS/'SAFE_MEMORY_ADMISSION.json',dict(status='COMPLETE',binding=binding(inputs=shadow_paths,evaluator='five unchanged-trajectory GT-free admission rules',scope='shadow write diagnostics; causal qualification from native factorial'),
        summaries={k:dict(c) for k,c in shadow_groups.items()},selected_rule='bounded_quarantine',GT_free=True,
        runtime_confirmed_is_not_GT_certified=True,raw_observations_preserved=True,raw_ID_and_Bank_updates_unfiltered=True,
        native_safety=ref(REPORTS/'NATIVE_FUTURE_SAFETY_GO_NO_GO.json'),shadow_is_not_native_tracking_gain=True))
    print('PHASE16_NATIVE_MEMORY_FINAL_GATE',gate['status'],len(violations['B']),len(benefits['B']),label['status'],flush=True)


if __name__=='__main__':main()
