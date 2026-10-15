"""Reuse saved complete strict DEV5 outputs. No GMT or encoder replay."""
import collections
from common import *
from audit_tracking_identity_pairs import predictions, matched_detection_rows
from audit_tracking_anchor_errors import anchor_states, recent_output_links
from audit_decision_trace import records
from audit_full_train_identity_events import original_truth, label_detectors
from collect_track_hard_events import pure_reference_identity
from census_trackvggt_train import patch_bin, patch_count


def main():
    baseline_path=ROOT/'results/causal_B0_strict_suite_v1_metrics.json'
    candidate_path=ROOT/'results/causal_B5_dense_seed20261014_strict_suite_v1_metrics.json'
    baseline=json.loads(baseline_path.read_text());candidate=json.loads(candidate_path.read_text())
    data=train_data();sources={str(p):sha256(p) for p in [baseline_path,candidate_path]};rows=[]
    census=json.loads((ROOT/'results/trackvggt_gt_quality_summary_v1.json').read_text());census_by={r['video_id']:r for r in census['rows']}
    for scene in [1,10,11,18,19]:
        paths=[];manifests=[];scene_metrics=[]
        for suite in [baseline,candidate]:
            entry=suite['run_manifests'][str(scene)];mp=Path(entry['path'])
            if sha256(mp)!=entry['sha256']:raise ValueError('saved run manifest changed')
            path=mp.parent/'predictions.json';metric=path.with_suffix('.metrics.json')
            if sha256(path)!=suite['prediction_sha256'][str(path)]:raise ValueError('saved prediction changed')
            m=json.loads(metric.read_text())
            if m['prediction_sha256']!=sha256(path) or not m['full_video'] or not m['causal_online']:raise ValueError('saved metrics not paired complete causal output')
            sources.update({str(p):sha256(p) for p in [mp,path,metric]});paths.append(path);manifests.append(json.loads(mp.read_text()));scene_metrics.append(m['metrics'])
        for field in ['checkpoint_sha256','config_sha256','annotation_sha256','seed']:
            if manifests[0][field]!=manifests[1][field]:raise ValueError('DEV retrospective unpaired: '+field)
        (base,_),(new,_)=[predictions(p) for p in paths]
        images=video_records(data,scene);index={r['image_id']:r for r in images}
        truth,blocked,hashes,_=original_truth(data,scene,images)
        if blocked:raise ValueError('strictDEV5 unexpectedly has unreliable frame')
        sources.update(hashes)
        by_view_frame={(r['view_id'],r['frame_id']):truth[r['image_id']] for r in images}
        matched=sorted(matched_detection_rows(base,new,index,by_view_frame),key=lambda r:(r['frame'],r['camera'],r['image_id'],r['bbox']))
        anchor={};size=collections.Counter()
        for cross in [False,True]:
            old_states=anchor_states(matched,'base',cross);new_states=anchor_states(matched,'new',cross)
            anchor['cross_camera_past_GT_matched_output' if cross else 'all_camera_past_GT_matched_output']=dict(
                baseline=dict(collections.Counter(r['state'] for r in old_states)),candidate=dict(collections.Counter(r['state'] for r in new_states)),
                transitions=dict(collections.Counter(a['state']+' -> '+b['state'] for a,b in zip(old_states,new_states))))
        for row in matched:
            im=index[row['image_id']];size[patch_bin(patch_count(row['bbox'],im['width'],im['height']))]+=1
        common=base.keys()&new.keys();ids_count=len({r['gt'] for r in matched});c=census_by[scene]
        rows.append(dict(scene=scene,groups=c['synchronized_groups'],GT_target_rows=c['gt_valid_target_rows'],raw_GT_identities=c['unique_raw_gt_identities'],
            baseline_metrics=scene_metrics[0],candidate_metrics=scene_metrics[1],metric_deltas={k:scene_metrics[1][k]-scene_metrics[0][k] for k in scene_metrics[0]},
            detector_cohort=dict(baseline=len(base),candidate=len(new),common=len(common),added=len(new.keys()-base.keys()),removed=len(base.keys()-new.keys()),
                common_GT_matched=len(matched),GT_identities_among_common_matches=ids_count,common_without_reliable_GT=len(common)-len(matched)),
            GT_patch_bins={k:c[k] for k in ['patch_0','patch_1_3','patch_4_15','patch_16_plus']},common_matched_detection_patch_bins=dict(size),
            GT_matched_output_anchor_diagnostics=anchor,recent_output_link_diagnostic=recent_output_links(matched),
            actual_candidate_coverage='NOT_MEASURED',actual_solver_errors='NOT_MEASURED',detector_failures_vs_candidate_absence='NOT_MEASURED',
            sources=list(map(str,paths))))
    # Only scene1 has saved complete actual-solver traces. No ranking matrix was
    # retained, so TopK coverage cannot be reconstructed from final outputs.
    actual_path=ROOT/'results/actual_decision_full_scene1_cohort_verified_v2.json';actual=json.loads(actual_path.read_text());sources[str(actual_path)]=sha256(actual_path)
    for name,digest in actual['source_artifacts'].items():
        if sha256(name)!=digest:raise ValueError('actual scene1 trace changed')
        sources[name]=digest
    scene1=[]
    for run in actual['runs']:
        folder=Path(run['run']);images=video_records(data,1);index={i['image_id']:i for i in images}
        truth,blocked,hashes,_=original_truth(data,1,images)
        labels,_,_,det=label_detectors(folder,truth,blocked);counts=collections.Counter()
        for decision in records(folder/'actual_decisions.jsonl.gz'):
            if decision['stage']=='run_memory_tracker':continue
            refs=collections.defaultdict(list);cams=collections.defaultdict(set)
            for ref in decision['references']:
                key=tuple(ref['source']);refs[ref['track']].append(labels[key]);cams[ref['track']].add(index[key[0]]['view_id'])
            pure={t:pure_reference_identity(v) for t,v in refs.items()}
            for source in decision['queries']:
                q=labels[tuple(source)];camera=index[source[0]]['view_id'];positive=[t for t,v in pure.items() if q is not None and v==q]
                counts['actual_primary_queries']+=1;counts['current_GT_known']+=q is not None;counts['pure_positive_full_legal_history']+=bool(positive)
                counts['positive_available_only_other_camera']+=bool(positive) and not any(camera in cams[t] for t in positive)
                counts['any_pure_positive_other_camera']+=any(any(c!=camera for c in cams[t]) for t in positive)
        scene1.append(dict(run=str(folder),counts=dict(counts),primary_states=run['counts'],detector_coverage=det,Top4='NOT_MEASURED',Top8='NOT_MEASURED',Top16='NOT_MEASURED'))
    rows[0]['actual_candidate_coverage']=scene1;rows[0]['actual_solver_errors']=dict(common_queries=actual['common_primary_queries'],known_fixes=actual['baseline_wrong_method_correct'],known_harms=actual['baseline_correct_method_wrong'],unknown_either=actual['unknown_either'],transition_counts=actual['transition_counts'])
    stability_path=ROOT/'results/causal_dense_seed_stability_v1.json';stability=json.loads(stability_path.read_text());sources[str(stability_path)]=sha256(stability_path)
    parity_path=ROOT/'results/causal_dense_geometry_full_strict_attribution_v1.json';parity=json.loads(parity_path.read_text());sources[str(parity_path)]=sha256(parity_path)
    concentration=dict(scene1_CVIDF1_delta=rows[0]['metric_deltas']['CVIDF1'],other_scene_CVIDF1_deltas={str(r['scene']):r['metric_deltas']['CVIDF1'] for r in rows[1:]},
        pooled_CVIDF1_delta=candidate['metrics']['CVIDF1']-baseline['metrics']['CVIDF1'],
        IDSW_reduction_by_scene={str(r['scene']):-r['metric_deltas']['IDSW'] for r in rows},
        caution='Pooled CVIDF1 is not a mean or sum of scene percentages; no additive percentage-point attribution from those deltas.')
    report=dict(status='COMPLETE_RETROSPECTIVE_WITH_UNMEASURED_CANDIDATE_FIELDS',no_GMT_or_VGGT_replay=True,used_for_fit_or_model_selection=False,
        rows=rows,concentration=concentration,training_and_inference_seed_stability=stability['groups'],scene1_training_head_stability=stability['scene1_training_head_diagnostic'],
        geometry_attribution=parity,source_artifacts=sources,causal_mechanism_conclusions=[
            'Scene1 has large GT boxes (4151/4395 raw target rows cover16+ centers); scenes18/19 mostly0–3 centers. This is association, not a controlled resolution effect.',
            'Complete actual solver evidence exists for scene1 only; other scenes cannot establish whether candidate coverage, detector misses or visual scoring dominate.',
            'Scene1 gains are not stable over training heads: one seed harms CVIDF1 despite fewer IDSW. Pooled fixed-head inference-RNG trials also include a negative delta.',
            'B5 features and B7 geometry have exactly identical final predictions in all5 scenes for the primary head. Prior gains do not establish a geometry mechanism.',
            'Past GT-matched output-anchor diagnostics may skip unmatched observations. They are linear-query diagnostics and must not be relabeled actual solver errors or pure history coverage.',
            'Original GMT backbone saw original TRAIN including DEV5; these results do not establish unseen-scene generalization.'],provenance=provenance())
    save_json(ROOT/'results/trackvggt_dev5_retrospective_analysis_v1.json',report)
    print(json.dumps(dict(status=report['status'],concentration=concentration,scene1_actual=scene1)))


if __name__=='__main__':main()
