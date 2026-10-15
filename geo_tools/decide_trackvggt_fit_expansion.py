"""One finite TRAIN-only expansion decision after a complete sealed inventory."""
import collections
import hashlib
from common import *
from audit_decision_trace import records
from summarize_full_train_identity_audit import FIT


def category(event):
    if event['pure_positive_top8'] and (event['ranking_state']=='wrong_link' or event['known_wrong']):return 'hard'
    if event['pure_positive_top8'] and (event['ranking_state']=='ambiguous_reference' or event['relative_margin']<.1):return 'ambiguous'
    if event['pure_positive_top8'] and event['relative_margin']<.4:return 'moderate'
    if event['pure_positive_top8']:return 'easy'
    return 'missing_or_unknown'


def stable_key(event):
    return hashlib.sha256(json.dumps(event['key'],separators=(',',':')).encode()).hexdigest()


def scene_rotating_select(pool,limit,used,selected_frames,gap=3):
    queues={s:collections.deque(sorted(events,key=stable_key)) for s,events in pool.items()};result=[]
    while len(result)<limit and any(queues.values()):
        for scene in sorted(queues):
            while queues[scene]:
                event=queues[scene].popleft();key=tuple(event['key'])
                if key in used:continue
                history_key=(scene,event['camera'],event['query_gt_identity'])
                if event['query_gt_identity'] is not None and any(abs(f-event['frame'])<gap for f in selected_frames[history_key]):continue
                used.add(key);selected_frames[history_key].add(event['frame']);result.append(event);break
            if len(result)>=limit:break
    return result


def cohort_statistics(events):
    known={(e['key'][0],e['query_gt_identity']) for e in events if e['query_gt_identity'] is not None};frames=collections.defaultdict(set)
    for e in events:
        if e['query_gt_identity'] is not None:frames[e['key'][0],e['camera'],e['query_gt_identity']].add(e['frame'])
    return dict(queries=len(events),unique_events=len({tuple(e['key']) for e in events}),scenes=sorted({e['key'][0] for e in events}),
        unique_known_scene_identities=len(known),per_scene=dict(collections.Counter(e['key'][0] for e in events)),
        categories=dict(collections.Counter(category(e) for e in events)),patch_bins=dict(collections.Counter(e['patch_bin'] for e in events)),
        reachable_rank_or_primary_errors=sum(category(e)=='hard' for e in events),known_Top1_errors=sum(e['ranking_state']=='wrong_link' for e in events),
        pure_positive_in_top8_queries=sum(e['pure_positive_top8'] for e in events),
        genuine_NULL_within_legal_history_queries=sum(e.get('genuine_NULL_within_legal_candidates',False) for e in events),
        unknown_current_GT_queries=sum(e['query_gt_identity'] is None for e in events),
        adjacent_group_pairs_same_scene_camera_identity=sum(sum(b-a==1 for a,b in zip(sorted(v),sorted(v)[1:])) for v in frames.values()),
        groups_with_gap_le_3_same_scene_camera_identity=sum(sum(b-a<=3 for a,b in zip(sorted(v),sorted(v)[1:])) for v in frames.values()))


def main():
    invpath=ROOT/'results/trackvggt_full_train_event_inventory_v1.json';inv=json.loads(invpath.read_text())
    if inv['status']!='COMPLETE_12_ASSOCIATION_TRAIN_FULL_CAUSAL_AUDITS':raise ValueError('all12 full scans must finish before deciding')
    original=json.loads((ROOT/'results/trackvggt_expanded_cohort_v2.json').read_text());v2keys={tuple(r['key']) for r in original['fit']}
    all_events={};pools=collections.defaultdict(lambda:collections.defaultdict(list));scene_stats={}
    for scene in inv['scenes']:
        if scene['scene'] not in FIT:continue # held-out12 cannot choose the new fit
        path=Path(scene['events'])
        if sha256(path)!=scene['events_sha256']:raise ValueError('pool changed')
        events=list(records(path));scene_stats[scene['scene']]=cohort_statistics(events)
        for event in events:
            key=tuple(event['key'])
            if key in all_events:raise ValueError('pool duplicate')
            all_events[key]=event
            if key not in v2keys:pools[category(event)][scene['scene']].append(event)
    if not v2keys<=all_events.keys():raise ValueError('locked V2 query missing from full audit')
    v2events=[all_events[tuple(r['key'])] for r in original['fit']];old_stats=cohort_statistics(v2events)
    new_hard=sum(len(v) for v in pools['hard'].values());hard_scenes=[s for s,v in pools['hard'].items() if len(v)>=10]
    stats=inv['fit_pool_statistics']
    # Finite operational rule fixed after tier1 diagnostics, before any teacher
    # fitting/comparison. Not a significance test or evidence of teacher efficacy.
    gate=stats['reachable_rank_or_primary_failure']>=260 and stats['reachable_ranking_errors']>=38 and new_hard>=130 and len(hard_scenes)>=3
    decision='EXPAND_COHORT_V3' if gate else 'LIMITED_CORRECTABLE_ERROR_HEADROOM'
    coverage=json.loads((ROOT/'results/trackvggt_candidate_headroom_v1.json').read_text())
    counts=coverage['fit_counts']
    if not gate and counts['N_missing_candidate']>counts['N_pure_positive_full']:
        decision='CANDIDATE_COVERAGE_BOTTLENECK'
    comparison=dict(status='DATA_AUDIT_COMPLETE_TEACHER_COMPARISON_NOT_MEASURED',decision=decision,V2=old_stats,V3='NOT_CREATED',
        fit_pool_counts=stats,new_reachable_known_errors=new_hard,new_error_scenes_with_at_least10=hard_scenes,
        operational_gate=dict(min_reachable_errors=260,min_reachable_ranking_errors=38,min_errors_outside_V2=130,min_scenes_with10_new_errors=3),
        validation_original300_unchanged=True,validation_scene12_never_in_fit_pool=True,teacher_network_training_budget_and_validation_metrics='NOT_MEASURED',
        V2_raw_artifacts_and_features_unchanged=True,source_inventory_sha256=sha256(invpath),selection_uses_DEV5_or_TEST=False)
    purity_path=ROOT/'results/trackvggt_reference_purity_supplement_v1.json'
    if not purity_path.exists():raise ValueError('mixed-vs-unknown and cross-camera supplement required')
    purity=json.loads(purity_path.read_text())
    if purity['status']!='COMPLETE_ALL12_SAVED_TRACE_SUPPLEMENT':raise ValueError('reference purity supplement incomplete')
    comparison['cross_camera_correctable_query_counts']={str(r['scene']):r['counts'] for r in purity['rows'] if r['scene'] in FIT}
    comparison['reference_purity_supplement_sha256']=sha256(purity_path)
    if gate:
        used=set(v2keys);frames=collections.defaultdict(set)
        for e in v2events:frames[e['key'][0],e['camera'],e['query_gt_identity']].add(e['frame'])
        selected=[];quotas=dict(hard=1000,ambiguous=350,moderate=300,easy=200,missing_or_unknown=150)
        for group,quota in quotas.items():selected+=scene_rotating_select(pools[group],quota,used,frames)
        # Fill unused quota only from legal known-positive categories, never
        # duplicate queries or inflate unknown/missing samples to reach a cap.
        remaining=collections.defaultdict(list)
        for group in ['hard','ambiguous','moderate','easy']:
            for scene,events in pools[group].items():remaining[scene].extend(events)
        selected+=scene_rotating_select(remaining,2000-len(selected),used,frames)
        if len(selected)<1000:raise ValueError('V3 gate passed but unique spaced training support insufficient')
        combined=v2events+selected;new_stats=cohort_statistics(combined)
        manifest_path=ROOT/'manifests/trackvggt_expanded_training_v3.json'
        if manifest_path.exists():raise FileExistsError('do not overwrite locked V3')
        selected_index=[]
        for event in selected:
            positive=event['pure_positive_top8'];local_NULL=event['genuine_NULL_within_legal_candidates']
            selected_index.append(dict(key=event['key'],scene=event['key'][0],frame=event['frame'],camera=event['camera'],
                category=category(event),patch_bin=event['patch_bin'],query_gt_identity_OFFLINE_LABEL_ONLY=event['query_gt_identity'],
                positive_in_top8=positive,genuine_NULL_within_full_legal_history=local_NULL,
                supervised_assignment_eligible=positive or local_NULL,missing_or_impure_mask=not positive and not local_NULL,
                source_events=next(s['events'] for s in inv['scenes'] if s['scene']==event['key'][0])))
        manifest=dict(status='LOCKED_TRAIN_EVENT_INDEX_REQUIRES_SELECTED_FEATURE_CAPTURE',version=3,
            base_cohort='Cohort-V2',base_cohort_sha256=sha256(ROOT/'results/trackvggt_expanded_cohort_v2.json'),preserved_V2_fit_rows=original['fit'],additional_fit=selected_index,
            validation_rows=original['validation'],total_fit_queries=len(combined),total_validation_queries=300,statistics=new_stats,
            selection='Fixed TRAIN-only category quotas, scene round robin, stable physical-key hash order; new queries spaced at least3 groups within scene/camera/known identity including V2. Unused quotas fill from known-positive categories; no duplicate events.',
            added_quotas=quotas,selected_new_statistics=cohort_statistics(selected),independent_events_are_queries_not_independent_error_episodes=True,
            GT_used_only_for_postrun_TRAIN_fit_selection=True,DEV5_or_TEST_used=False,heldout_validation_GT_error_resampling=False,
            feature_status='NOT_CAPTURED',max_combined_teacher_cache_bytes=2147483648,
            feature_plan='Reuse exact-hash V2 pools. Capture native summaries and matching VGGT/DINO target tokens only for newly locked selected events; no foundation encoder during statistical scans. Continuous GMT replay, if necessary, constructs missing selected features and must reproduce the sealed scan predictions/history.',
            training_comparison='Same AG/QD architecture,1000 optimizer updates,batch4,seeds and fixed validation300 for V2/V3; no fitted comparison exists yet.',
            full_scene12_diagnostic_has_been_viewed=True,validation_is_not_blind_external_test=True,inventory_sha256=sha256(invpath))
        save_json(manifest_path,manifest);comparison['V3']=new_stats;comparison['V3_manifest_sha256']=sha256(manifest_path);comparison['added_queries']=len(selected)
    save_json(ROOT/'results/trackvggt_cohort_v2_v3_comparison.json',comparison)
    doc=['# Fit data expansion decision','',f'Decision: {decision}. This is a data sufficiency decision; teacher benefit is NOT_MEASURED.','',
        f"The complete11-scene fit pool contains {stats['reachable_rank_or_primary_failure']} Top8-reachable known ranking or primary wrong/missed queries, including {stats['reachable_ranking_errors']} reachable known ranking errors. {new_hard} reachable events are outside V2. New errors span {len(hard_scenes)} scenes with at least10 events each. Counts can repeat along an identity; scene/identity and adjacent-group statistics accompany the inventory.", '',
        f"Cohort-V2 remains1000 fit and300 validation with all original artifacts unchanged. V3 status: {comparison['V3'] if isinstance(comparison['V3'],str) else str(comparison['V3']['queries'])+' unique fit queries, selected feature capture pending'}. No model was selected using these data diagnostics.", '',
        'AG, ordinary/track-conditioned QD and causal temporal consistency retain their original independent research gates. Larger supervision cannot establish that VGGT exceeds DINO/native controls. Known candidate absence and mixed history remain masked diagnostic classes, not true new-ID targets. The full12 validation audit is disclosed as viewed; original300 validation queries remain fixed, with no error-driven resampling. DEV5 is retrospective only and TEST is unused.', '',
        'Next, compare V2/V3 under identical network, optimizer-update budget and fixed validation300. Preserve ROI/AG-A/B/C, matched DINO/native, query-shuffle, QD and temporal controls. Bounded224/336 crop comparisons are required to distinguish resolution support from identity representation on small targets. No teacher or student efficacy follows from this audit.', '',
        'VisionTrack supports an internal mechanism study with explicitly limited identity and scene diversity. Original GMT was trained on original TRAIN, including DEV; temporal query counts are correlated. This protocol alone does not support claims of unseen-scene or broad benchmark generalization. No external dataset download is authorized by this decision.']
    (ROOT/'docs/TRACKVGGT_FIT_DATA_EXPANSION_DECISION.md').write_text('\n'.join(doc)+'\n')
    print(json.dumps(dict(decision=decision,V2=old_stats,V3=comparison['V3'])))


if __name__=='__main__':main()
