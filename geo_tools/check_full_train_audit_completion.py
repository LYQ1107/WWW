"""Cross-check completed real query census, masks, locks and V3 indices."""
import collections
from common import *
from audit_decision_trace import records
from census_trackvggt_train import verify_v2


def main():
    base=Path('/data1/liuyeqiang/WWW_trackvggt_runtime/full_train_audit_v1');rows=[];signatures=set();artifacts={};allkeys=set();fitkeys=set()
    v3path=ROOT/'manifests/trackvggt_expanded_training_v3.json';v3=json.loads(v3path.read_text()) if v3path.exists() else None
    wanted={tuple(r['key']) for group in ['preserved_V2_fit_rows','additional_fit'] for r in v3[group]} if v3 else set();selected_metadata={}
    data=train_data();aligned_images=0
    for video in data['videos']:
        for image in video_records(data,video['id']):
            path=Path(image['file_name']);expected=f"{video['file_name']}_View{image['view_id']}"
            assert path.parent.parent.name==expected and path.stem==expected+f"_{image['frame_id']:06d}", 'GT/image metadata camera-frame alignment failed'
            aligned_images+=1
    for scene in [4,5,6,7,8,9,12,13,17,20,21,24]:
        folder=base/f'scene{scene}_full_v1';mp=folder/'manifest.json';m=json.loads(mp.read_text());s=json.loads((folder/'summary.json').read_text());c=s['counts']
        assert m['status']=='COMPLETE_FULL_CAUSAL_AUDIT' and m['full_video'] and m['prefix40_raw_prediction_exact']
        assert m['raw_GT_labeling_after_full_inference_and_OFF_parity'] and not m['GT_features_to_tracker'] and not m['future_query_features']
        assert m['vggt_calls']==m['dinov2_calls']==m['persistent_native_feature_vectors']==m['persistent_dense_candidate_score_matrices']==0
        if scene in [4,5,6,7,8,12]:assert m['existing_prefix_parity']['exact'] and m['existing_prefix_parity']['groups']==400
        p=m['provenance'];signatures.add(tuple(p[k] for k in ['commit','checkpoint_sha256','config_sha256','annotation_sha256','seed']))
        assert sum(s['primary_states'].values())==c['N_total']
        assert c['N_known']+c.get('N_current_unknown',0)==c['N_total']
        assert c['N_correctable_known_wrong']<=c['N_wrong_eligible']==c['N_wrong']
        assert c['N_top4']<=c['N_top8']<=c['N_top16']<=c['N_pure_positive_full']<=c['N_known']
        assert c['N_rank_correctable']<=c['N_known_ranking_wrong']
        assert c['N_assignment_correctable']==c['N_correctable_known_wrong']
        assert c['N_same_camera']+c['N_cross_camera']==c['N_pure_positive_full']
        assert c['N_primary_accepted']+c.get('N_threshold_rejected',0)+c.get('N_hungarian_unassigned',0)==c['N_total']
        assert c.get('N_memory_reactivated_after_primary_rejection',0)+c.get('N_newID_after_primary_rejection',0)==c['N_total']-c['N_primary_accepted']
        derived=collections.Counter()
        for e in records(folder/'events.jsonl.gz'):
            key=tuple(e['key']);assert key not in allkeys;allkeys.add(key)
            if key in wanted:selected_metadata[key]=e
            if scene!=12:fitkeys.add(key)
            assert e['known_wrong']==(e['actual_primary_state'] in ['wrong_link','missed_known_positive'])
            assert e['correctable_known_wrong']==(e['known_wrong'] and bool(e['positive_tracks']))
            if e['query_gt_identity'] is None:assert not e['known_wrong'] and not e['positive_tracks'] and 'F' in e['types']
            if e['actual_primary_state'] in ['ambiguous_reference','unknown_candidate_coverage']:assert not e['known_wrong'] and 'B' not in e['types']
            if e['positive_tracks']:assert not e['genuine_NULL_within_legal_candidates']
            if e['genuine_NULL_within_legal_candidates']:assert e['query_gt_identity'] is not None and not e['observed_positive_including_impure'] and 'E' not in e['types']
            derived['N_total']+=1;derived['N_wrong']+=e['known_wrong'];derived['N_correctable_known_wrong']+=e['correctable_known_wrong']
            derived['N_pure_positive_full']+=bool(e['positive_tracks'])
            for k in [4,8,16]:derived[f'N_top{k}']+=bool(set(e['positive_tracks'])&set(e[f'top{k}_tracks']))
        for k,v in derived.items():assert c[k]==v,(scene,k,c[k],v)
        assert len(json.loads((folder/'retained_examples.json').read_text()))<=m['max_retained_events']
        artifacts[str(mp)]=sha256(mp);rows.append(dict(scene=scene,groups=m['groups'],queries=c['N_total'],OFF40_exact=True,old_prefix400_exact=m['existing_prefix_parity'] if scene in [4,5,6,7,8,12] else 'NOT_APPLICABLE'))
    assert len(signatures)==1,'mixed frozen source/checkpoint/config/data/seed'
    v2=verify_v2()
    if v3path.exists():
        v3=json.loads(v3path.read_text());old={tuple(r['key']) for r in v3['preserved_V2_fit_rows']};added={tuple(r['key']) for r in v3['additional_fit']}
        assert len(old)==1000 and not old&added and len(added)==len(v3['additional_fit'])
        assert old|added<=fitkeys and not any(k[0]==12 for k in old|added)
        assert len(old|added)==v3['total_fit_queries'] and 2000<=len(old|added)<=4000
        current=json.loads((ROOT/'results/trackvggt_expanded_cohort_v2.json').read_text());assert v3['validation_rows']==current['validation'] and v3['preserved_V2_fit_rows']==current['fit']
        selected_frames=collections.defaultdict(set)
        for key in old:
            e=selected_metadata[key]
            if e['query_gt_identity'] is not None:selected_frames[key[0],e['camera'],e['query_gt_identity']].add(e['frame'])
        for r in v3['additional_fit']:
            e=selected_metadata[tuple(r['key'])]
            assert r['positive_in_top8']==e['pure_positive_top8'] and r['genuine_NULL_within_full_legal_history']==e['genuine_NULL_within_legal_candidates']
            assert r['query_gt_identity_OFFLINE_LABEL_ONLY']==e['query_gt_identity']
            assert r['supervised_assignment_eligible']==(r['positive_in_top8'] or r['genuine_NULL_within_full_legal_history'])
            assert r['missing_or_impure_mask']==(not r['positive_in_top8'] and not r['genuine_NULL_within_full_legal_history'])
            if e['query_gt_identity'] is not None:
                key=(r['scene'],r['camera'],e['query_gt_identity']);assert all(abs(f-r['frame'])>=3 for f in selected_frames[key]);selected_frames[key].add(r['frame'])
        assert sum(r['genuine_NULL_within_full_legal_history'] for r in v3['additional_fit'])==v3['rare_genuine_NULL_selected']==12
        artifacts[str(v3path)]=sha256(v3path)
    for name in ['trackvggt_full_train_event_inventory_v1.json','trackvggt_candidate_headroom_v1.json','trackvggt_event_type_distribution_v1.json']:
        path=ROOT/'results'/name;d=json.loads(path.read_text());assert d['status']=='COMPLETE_12_ASSOCIATION_TRAIN_FULL_CAUSAL_AUDITS';assert not d['pending_scenes'];artifacts[str(path)]=sha256(path)
    save_json(ROOT/'results/trackvggt_full_train_completion_audit_v1.json',dict(status='PASS_COMPLETE_REAL_AUDIT_CONTRACTS',rows=rows,total_actual_primary_queries=len(allkeys),fit_primary_queries=len(fitkeys),
        V3_evidence_counts=dict(known_Top1_wrong_with_pure_Top8_positive=sum(e['ranking_state']=='wrong_link' and e['pure_positive_top8'] for e in selected_metadata.values()),
            actual_primary_known_wrong_or_missed=sum(e['known_wrong'] for e in selected_metadata.values()),genuine_legal_history_NULL=sum(e['genuine_NULL_within_legal_candidates'] for e in selected_metadata.values())) if v3 else 'NOT_CREATED',
        common_frozen_source_checkpoint_config_annotation_seed=list(next(iter(signatures))),all24_metadata_filename_camera_frame_alignment_images=aligned_images,V2=v2,V3_contract='PASS_LOCKED_INDEX_NOT_FEATURES_OR_TEACHER_EFFICACY' if v3path.exists() else 'NOT_CREATED',
        source_artifacts=artifacts,limits='Checks validate data/purity/accounting/parity/locked membership only. No teacher fitting, learned corrections, deployment speed or paper success is claimed.',provenance=provenance()))
    print(json.dumps(dict(status='PASS_COMPLETE_REAL_AUDIT_CONTRACTS',total_queries=len(allkeys),fit_queries=len(fitkeys))))


if __name__=='__main__':main()
