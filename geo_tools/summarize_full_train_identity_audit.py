"""Generate complete/incomplete TRAIN reports from sealed causal audit records."""
import argparse
import collections
import csv
import io
from common import *
from audit_decision_trace import records

NEW=Path('/data1/liuyeqiang/WWW_trackvggt_runtime')
FIT=[4,5,6,7,8,9,13,17,20,21,24]


def write_csv(path,rows):
    fields=list(dict.fromkeys(key for row in rows for key in row))
    stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
    for row in rows:writer.writerow({k:json.dumps(v,separators=(',',':')) if isinstance(v,(list,dict)) else v for k,v in row.items()})
    Path(path).write_text(stream.getvalue())


def ratio(n,d):
    return dict(numerator=n,denominator=d,value=n/d if d else None)


def summarize(base,allow_incomplete):
    plan=json.loads((ROOT/'manifests/trackvggt_full_audit_plan_v1.json').read_text());census=json.loads((ROOT/'results/trackvggt_gt_quality_summary_v1.json').read_text())
    sizes_by_scene={r['video_id']:r for r in census['rows']};rows=[];scene_events={};machine=[];artifacts={};inventory=[];identities=collections.defaultdict(set)
    fit_sizes=collections.defaultdict(collections.Counter);fit_total=collections.Counter();event_types={};full_headroom={};all_keys=set();fit_stats=collections.Counter();identity_errors=collections.Counter();hard_frames=collections.defaultdict(set)
    v2=json.loads((ROOT/'results/trackvggt_expanded_cohort_v2.json').read_text());v2_keys={tuple(r['key']) for r in v2['fit']};v2_seen=set()
    for scene in sorted(FIT+[12]):
        folder=base/f'scene{scene}_full_v1';mp=folder/'manifest.json';c=sizes_by_scene[scene]
        if not mp.exists() or json.loads(mp.read_text()).get('status')!='COMPLETE_FULL_CAUSAL_AUDIT':
            rows.append(dict(Scene=scene,Status='INCOMPLETE',Role='validation_diagnostic' if scene==12 else 'fit',Frames=c['synchronized_groups'],Targets=c['gt_valid_target_rows'],N_total='NOT_MEASURED',N_known='NOT_MEASURED',N_wrong='NOT_MEASURED'))
            continue
        m=json.loads(mp.read_text())
        if not m['full_video'] or not m['continuous_from_first_frame'] or not m['prefix40_raw_prediction_exact'] or not m['raw_GT_labeling_after_full_inference_and_OFF_parity']:
            raise ValueError('unverified causal audit: '+str(mp))
        if m['groups']!=c['synchronized_groups'] or m['images']!=sum(c['camera_frames'].values()):raise ValueError('not complete source scene')
        if m['GT_features_to_tracker'] or m['future_query_features'] or m['vggt_calls'] or m['dinov2_calls']:raise ValueError('scope/causality breach')
        for name,digest in m['artifacts'].items():
            if sha256(folder/name)!=digest:raise ValueError('changed audit artifact: '+name)
            artifacts[str(folder/name)]=digest
        artifacts[str(mp)]=sha256(mp);summary=json.loads((folder/'summary.json').read_text());count=summary['counts']
        row=dict(Scene=scene,Status='COMPLETE',Role='validation_diagnostic' if scene==12 else 'fit',Frames=m['groups'],Targets=c['gt_valid_target_rows'],**count,
                 Top8Coverage=summary['rates']['CandidateCoverage@8'],UnknownRate=ratio(count['N_unknown'],count['N_total']),RawGTConflictingGroups=c['conflicting_bbox_groups'],UnreliableGTImageFrames=len(summary['GT_unreliable_frames']),
                 InferenceSeconds=m['inference_seconds'],GPU=m['provenance']['cuda_visible_devices'],PredictionSHA256=m['predictions_sha256'],GitSHA=m['provenance']['commit'],FullVideo=True,
                 RawRankAccuracy=ratio(count.get('N_raw_rank_known_correct',0),count.get('N_raw_rank_known_evaluable',0)),GTARankAccuracy=ratio(count.get('N_GTA_rank_known_correct',0),count.get('N_GTA_rank_known_evaluable',0)),
                 DetectorUnmatchedGTRows=summary['detector_coverage']['unmatched_GT_target_rows'],FinalOutputIdentityMetric='NOT_MEASURED_BY_QUERY_CENSUS')
        rows.append(row);machine.append(dict(scene=scene,summary=summary,manifest=str(mp)));event_types[str(scene)]=dict(types=summary['event_types_overlapping'],primary_states=summary['primary_states'],failure_evidence=summary['assignment_failure_evidence_overlapping'])
        full_headroom[str(scene)]=summary['rates']
        queries=[];scene_stat=collections.Counter();state_counts=collections.Counter();patch_counts=collections.Counter()
        for event in records(folder/'events.jsonl.gz'):
            key=tuple(event['key'])
            if key in all_keys:raise ValueError('physical query duplicated in inventory')
            all_keys.add(key);queries.append(event);scene_stat['queries']+=1;state_counts[event['actual_primary_state']]+=1;patch_counts[event['patch_bin']]+=1
            known=event['query_gt_identity'] is not None
            if known:identities[scene].add(event['query_gt_identity'])
            reachable=event['pure_positive_top8'] and (event['ranking_state']=='wrong_link' or event['known_wrong'])
            event['reachable_rank_or_primary_failure']=reachable
            scene_stat['reachable_rank_or_primary_failure']+=reachable;scene_stat['known_Top1_errors']+=event['ranking_state']=='wrong_link'
            scene_stat['reachable_ranking_errors']+=event['pure_positive_top8'] and event['ranking_state']=='wrong_link'
            scene_stat['known_primary_wrong_or_missed']+=event['known_wrong'];scene_stat['genuine_local_NULL']+=event['genuine_NULL_within_legal_candidates']
            if scene in FIT:
                if key in v2_keys:v2_seen.add(key)
                fit_stats['reachable_rank_or_primary_failure']+=reachable
                fit_stats['reachable_ranking_errors']+=event['pure_positive_top8'] and event['ranking_state']=='wrong_link'
                fit_stats['reachable_errors_outside_V2']+=reachable and key not in v2_keys
                if reachable:
                    identity_errors[scene,event['query_gt_identity']]+=1;hard_frames[scene,event['camera'],event['query_gt_identity']].add(event['frame'])
            scene_stat['unknown_query']+=not known
        if len(queries)!=count['N_total']:raise ValueError('all-query inventory/count mismatch')
        scene_events[scene]=queries
        inventory.append(dict(scene=scene,role=row['Role'],events=str(folder/'events.jsonl.gz'),events_sha256=sha256(folder/'events.jsonl.gz'),
            full_legal_query_index=True,counts=dict(scene_stat),known_GT_identities=len(identities[scene]),patch_bins=dict(patch_counts),primary_states=dict(state_counts),
            retained_examples=str(folder/'retained_examples.json'),retained_count=summary['retained_events'],features_saved=False,GT_labels_are_offline_only=True))
        if scene in FIT:
            fit_total.update(count)
            for size,value in summary['patch_bins'].items():fit_sizes[size].update(value)
    complete_scenes={r['Scene'] for r in rows if r['Status']=='COMPLETE'};complete=complete_scenes==set(FIT+[12])
    if not complete and not allow_incomplete:raise ValueError('required full scenes INCOMPLETE: '+str(sorted(set(FIT+[12])-complete_scenes)))
    status='COMPLETE_12_ASSOCIATION_TRAIN_FULL_CAUSAL_AUDITS' if complete else 'INCOMPLETE'
    common=dict(status=status,completed_scenes=sorted(complete_scenes),pending_scenes=sorted(set(FIT+[12])-complete_scenes),source_artifacts=artifacts,
        generator=dict(commit=SOURCE_COMMIT,source_tree_dirty=bool(SOURCE_STATUS),script_sha256=sha256(Path(__file__)),command=sys.argv),
        pooled_rate_scope='Completed fit sequences only; excludes scene12 and pending scenes. All-fit claim requires COMPLETE status.',
        scope='All actual primary association queries, excluding initial largest-camera identity initialization; no cross-camera-only selection. Scene12 is separate validation diagnostic.',
        overlaps='A/B/C/D/E/F, pure-positive availability, ranking and actual solver failure counts overlap. Unknown/impure is not a known error. No sum of categories.',
        final_output='Predicted IDs are arbitrary permutations of GT IDs. Reported final-ID anchors are past-output consistency only, not literal GT-ID equality or CVIDF1.')
    headroom=dict(**common,scene_rates=full_headroom,fit_counts=dict(fit_total),fit_rates={f'CandidateCoverage@{k}':ratio(fit_total[f'N_top{k}'],fit_total['N_pure_positive_full']) for k in [4,8,16]},
        fixed_native_history_GT_assisted_correctable_rate=ratio(fit_total['N_correctable_known_wrong'],fit_total['N_wrong_eligible']),
        no_metric_upper_bound=True,definitions=machine[0]['summary']['definitions'] if machine else 'NOT_MEASURED')
    save_json(ROOT/'results/trackvggt_candidate_headroom_v1.json',headroom)
    save_json(ROOT/'results/trackvggt_event_type_distribution_v1.json',dict(**common,by_scene=event_types,fit_patch_bins={k:dict(v) for k,v in fit_sizes.items()},VGGT_ROI_and_DINO_same_batch_recall='NOT_MEASURED'))
    adjacency=dict(adjacent_group_pairs=sum(sum(b-a==1 for a,b in zip(sorted(v),sorted(v)[1:])) for v in hard_frames.values()),
        distinct_scene_camera_identity_tracks=len(hard_frames),scope='Adjacent distinct groups within same scene/camera/GT identity among reachable known error queries; not independent physical episodes.')
    inv=dict(**common,pool_name='FullTrain-Hard-Pool',scenes=inventory,fit_pool_statistics=dict(fit_stats),fit_unique_known_scene_identities=sum(len(identities[s]) for s in FIT),
        fit_reachable_error_scene_identities=len(identity_errors),fit_reachable_errors_by_identity=[dict(scene=k[0],identity=k[1],queries=v) for k,v in identity_errors.most_common()],
        error_temporal_adjacency=adjacency,V2_existing_fit_queries_recovered=len(v2_seen),V2_expected_fit_queries=1000,V2_lock_sha256=sha256(ROOT/'results/trackvggt_expanded_cohort_v2.json'),
        V2_reachable_mixed_errors=130,V2_known_ranking_errors=19,selected_error_counts_not_whole_train=True)
    save_json(ROOT/'results/trackvggt_full_train_event_inventory_v1.json',inv)
    write_csv(ROOT/'results/trackvggt_full_train_error_census_v1.csv',rows)
    difficulty=[dict(Scene=r['Scene'],Frames=r['Frames'],Targets=r['Targets'],KnownQueries=r.get('N_known','NOT_MEASURED'),KnownWrong=r.get('N_wrong','NOT_MEASURED'),PurePositiveAvailable=r.get('N_pure_positive_full','NOT_MEASURED'),Top8Coverage=r.get('Top8Coverage','NOT_MEASURED'),ImpureReferences=r.get('N_impure','NOT_MEASURED'),UnknownRate=r.get('UnknownRate','NOT_MEASURED'),Status=r['Status'],Role=r['Role']) for r in rows]
    write_csv(ROOT/'results/trackvggt_scene_difficulty_v1.csv',difficulty)
    lines=['# Full association-TRAIN causal identity audit','',f'Status: {status}. Complete scenes: {sorted(complete_scenes)}. Pending scenes: {common["pending_scenes"]}.','',
        'Each sequence starts at group1 and runs continuously to the actual final group using the frozen GMT checkpoint and seed20261014. DecisionTrace observes the original Hungarian/threshold/memory methods. GT is read only after full inference and independent uninstrumented OFF40 equality. Existing400-prefix collectors must agree. No VGGT/DINO, future or GT tracker features. Detailed examples are bounded1000/scene; complete lightweight query indices retain coverage. Runtime is offline audit cost, not deployment FPS.','',
        'Raw GT conflicts in association scenes20/24 are preserved. Entire camera-frames containing duplicate or invalid GT are unreliable during post-run proposal matching; subsequent reference purity includes their unknown labels. These are conservatively masked diagnostics, not repaired official evaluations.','',
        '|Scene|Groups|Known queries|Known wrong/missed|Pure positive full|Top8 numerator /denominator|Impure histories|Unknown|Status|','|---|---:|---:|---:|---:|---|---:|---:|---|']
    for r in rows:
        cov=r.get('Top8Coverage',{});cv=f"{cov['numerator']} /{cov['denominator']}" if isinstance(cov,dict) and 'numerator' in cov else 'NOT_MEASURED'
        lines.append(f"|{r['Scene']}|{r['Frames']}|{r.get('N_known','NOT_MEASURED')}|{r.get('N_wrong','NOT_MEASURED')}|{r.get('N_pure_positive_full','NOT_MEASURED')}|{cv}|{r.get('N_impure','NOT_MEASURED')}|{r.get('N_unknown','NOT_MEASURED')}|{r['Status']}|")
    lines += ['', 'Counts overlap and describe the frozen native history. Known wrong includes pure known wrong links and rejected queries with a pure positive, not unknown selections. A ranking error can already be corrected by Hungarian. Missing pure positive can coexist with an impure track containing a correct observation. Candidate coverage divides by known queries with a pure full-history positive. Correctable error rate divides by reliably known primary wrong/missed queries. Neither is a HOTA/CVIDF1 upper bound.','',
              'Actual raw/GTA ranks, Hungarian pairs, strict thresholds, memory reactivation, new-ID allocation and previous-group output anchors are separate in each sealed summary. Descriptor averages are not assigned GT from source tags. Final output IDs are permutation invariant; query counts cannot establish an end-to-end metric gain.']
    (ROOT/'docs/TRACKVGGT_FULL_TRAIN_AUDIT.md').write_text('\n'.join(lines)+'\n')
    analysis=['# Correctable identity error analysis','',f'Status: {status}. Fit totals exclude validation scene12.','',
        '|Scene|A ranking|B actual known failure|Threshold rejected (all)|No pure candidate|Impure|Known local NULL|GT not detector-matched|','|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        analysis.append('|'+ '|'.join(str(v) for v in [r['Scene'],r.get('N_rank_correctable','NOT_MEASURED'),r.get('N_assignment_correctable','NOT_MEASURED'),r.get('N_threshold_rejected','NOT_MEASURED'),r.get('N_missing_candidate','NOT_MEASURED'),r.get('N_impure','NOT_MEASURED'),r.get('N_genuine_NULL_within_legal_candidates','NOT_MEASURED'),r.get('DetectorUnmatchedGTRows','NOT_MEASURED')])+'|')
    analysis += ['', 'These columns are not mutually exclusive. Threshold rejection is an actual solver observation; NULL dominance and multiplicity-boundary evidence are separate overlapping flags, not invented exclusive causes. No pure candidate does not identify a causal detector failure. GT not detector-matched is a separate one-to-one detector coverage statistic and cannot be added to query errors.','',
        '|518 patch centers|Queries|Known wrong/missed|Correctable /known wrong|GTA Recall1 /pure-positive-full|VGGT ROI Recall1|DINO Recall1|','|---|---:|---:|---|---|---|---|']
    for size in ['0','1-3','4-15','16+']:
        s=fit_sizes[size];analysis.append(f"|{size}|{s['queries']}|{s['known_wrong']}|{s['correctable_known_wrong']} /{s['known_wrong']}|{s['GTA_top1_correct']} /{s['pure_positive_full']}|NOT_MEASURED|NOT_MEASURED|")
    analysis += ['', f"Completed fit sequences contain {fit_stats['reachable_rank_or_primary_failure']} Top8-reachable known ranking or actual primary wrong/missed queries; {fit_stats['reachable_errors_outside_V2']} lie outside fixed V2. Reachable pure-known Top1 errors: {fit_stats['reachable_ranking_errors']}. These overlap actual solver errors and can repeat along one identity. {adjacency['adjacent_group_pairs']} adjacent-group pairs occur within the same scene/camera/identity error history; do not equate queries with independent error episodes.", '',
        'DEV5 retrospective is saved separately. It cannot determine fit selection or AG/QD architecture. Scene1 saved actual solver evidence cannot establish candidate coverage in the other four scenes, where those fields remain NOT_MEASURED.']
    (ROOT/'docs/TRACKVGGT_CORRECTABLE_ERROR_ANALYSIS.md').write_text('\n'.join(analysis)+'\n')
    save_json(base/'aggregate_snapshot.json',dict(**common,fit_pool_statistics=dict(fit_stats),completed=True if complete else False))
    print(json.dumps(dict(status=status,fit_counts=dict(fit_total),fit_pool_statistics=dict(fit_stats),completed_scenes=sorted(complete_scenes))))
    return complete,scene_events,inv


def main():
    p=argparse.ArgumentParser();p.add_argument('--audit-root',type=Path,default=NEW/'full_train_audit_v1');p.add_argument('--allow-incomplete',action='store_true');a=p.parse_args()
    summarize(a.audit_root,a.allow_incomplete)


if __name__=='__main__':main()
