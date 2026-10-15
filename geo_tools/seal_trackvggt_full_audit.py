"""Seal phase metadata and data-derived conclusions after completion checks."""
from common import *


def main():
    check=json.loads((ROOT/'results/trackvggt_full_train_completion_audit_v1.json').read_text())
    assert check['status']=='PASS_COMPLETE_REAL_AUDIT_CONTRACTS'
    plan_path=ROOT/'manifests/trackvggt_full_audit_plan_v1.json';plan=json.loads(plan_path.read_text());base=Path(plan['full_audit_output'])
    plan.update(status='COMPLETE_ALL24_CENSUS_ALL12_CAUSAL_AUDITS_V3_INDEX_LOCKED',completed_full_video_ids=[4,5,6,7,8,9,12,13,17,20,21,24],
                completion_checks='results/trackvggt_full_train_completion_audit_v1.json',V3_status='LOCKED_INDEX_FEATURE_CAPTURE_PENDING',teacher_training_started=False,
                scan_source_commit='9620ac46',full_validation12_diagnostic_viewed=True,original_validation300_unchanged=True,official_TEST_used=False)
    for row in plan['scene_plan']:
        folder=base/f"scene{row['scene']}_full_v1";mp=folder/'manifest.json';m=json.loads(mp.read_text());s=json.loads((folder/'summary.json').read_text())
        assert m['status']=='COMPLETE_FULL_CAUSAL_AUDIT'
        row.update(causal_status=m['status'],manifest=str(mp),manifest_sha256=sha256(mp),actual_groups=m['groups'],actual_images=m['images'],
            source_git_SHA=m['provenance']['commit'],checkpoint_sha256=m['provenance']['checkpoint_sha256'],config_sha256=m['provenance']['config_sha256'],
            annotation_sha256=m['provenance']['annotation_sha256'],prediction_sha256=m['predictions_sha256'],raw_GT_sha256=s['raw_gt_sha256'],
            command=m['provenance']['command'],GPU=m['provenance']['cuda_visible_devices'],wall_seconds=m['wall_seconds'],inference_seconds=m['inference_seconds'],
            full_video=True,OFF40_exact=m['prefix40_raw_prediction_exact'],old_prefix_parity=m['existing_prefix_parity'],counts=s['counts'],GT_labels_post_inference=True)
    save_json(plan_path,plan)
    h=json.loads((ROOT/'results/trackvggt_candidate_headroom_v1.json').read_text());v3=json.loads((ROOT/'manifests/trackvggt_expanded_training_v3.json').read_text())
    compare=json.loads((ROOT/'results/trackvggt_cohort_v2_v3_comparison.json').read_text());purity=json.loads((ROOT/'results/trackvggt_reference_purity_supplement_v1.json').read_text())
    fit=[r for r in purity['rows'] if r['scene']!=12]
    cross=sum(r['counts'].get('correctable_with_any_other_camera_pure_positive',0) for r in fit);crossonly=sum(r['counts'].get('correctable_only_other_camera_pure_positive',0) for r in fit)
    impure=sum(r['counts'].get('missing_pure_positive_correct_observation_only_in_impure_tracks',0) for r in fit)
    block=['<!-- SEALED_DATA_CONCLUSIONS -->','',
        f"Fit primary correctable errors with any other-camera pure positive: {cross}; with only other-camera pure positives: {crossonly}. Same-camera support often coexists, so these are not all exclusively cross-camera failures.", '',
        f"Of {h['fit_counts']['N_missing_candidate']} fit queries without a pure positive, {impure} still contain the correct GT observation in impure/unreliable reference tracks. This is conservative supervised candidate availability, not proof of {h['fit_counts']['N_missing_candidate']} detector failures or absent correct observations.", '',
        '|Cohort|Unique fit queries|Scenes|Known scene/identities|Reachable ranking or primary errors|Pure-Top8 ranking errors|Genuine legal-history NULL|',
        '|---|---:|---:|---:|---:|---:|---:|',
        f"|V2|1000|5|{compare['V2']['unique_known_scene_identities']}|130|19|0|",
        f"|V3|{v3['total_fit_queries']}|{len(v3['statistics']['scenes'])}|{v3['statistics']['unique_known_scene_identities']}|{v3['statistics']['reachable_rank_or_primary_errors']}|{check['V3_evidence_counts']['known_Top1_wrong_with_pure_Top8_positive']}|{v3['statistics']['genuine_NULL_within_legal_history_queries']}|", '',
        f"V3 includes {v3['statistics']['pure_positive_in_top8_queries']} pure-Top8-positive queries and {v3['statistics']['genuine_NULL_within_legal_history_queries']} genuine legal-history NULL queries; the other {v3['total_fit_queries']-v3['statistics']['pure_positive_in_top8_queries']-v3['statistics']['genuine_NULL_within_legal_history_queries']} queries retain supervision masks. Zero-center queries: {v3['statistics']['patch_bins']['0']}/{v3['total_fit_queries']}; crop/ROI controls remain necessary.", '',
        f"A single pre-feature/pre-training index correction reserved rare genuine NULL before the abundant missing/unknown pool. The first V3 index and comparison remain archived by exact hash in {v3['previous_revision']['path'] if v3['previous_revision'] else 'NOT_APPLICABLE'}. The final revision adds no duplicate query and all added known queries respect the3-group spacing from both V2 and each other. Original V2 is unchanged. Teacher V2/V3 benefit is NOT_MEASURED."]
    for name in ['TRACKVGGT_FIT_DATA_EXPANSION_DECISION.md','TRACKVGGT_CORRECTABLE_ERROR_ANALYSIS.md']:
        path=ROOT/'docs'/name;text=path.read_text().split('<!-- SEALED_DATA_CONCLUSIONS -->')[0].rstrip();path.write_text(text+'\n\n'+'\n'.join(block)+'\n')
    required=['docs/TRACKVGGT_FULL_TRAIN_AUDIT.md','docs/TRACKVGGT_DATASET_DIFFICULTY_AUDIT.md','docs/TRACKVGGT_CORRECTABLE_ERROR_ANALYSIS.md','docs/TRACKVGGT_FIT_DATA_EXPANSION_DECISION.md',
        'results/trackvggt_all24_dataset_census_v1.csv','results/trackvggt_gt_quality_summary_v1.json','results/trackvggt_full_train_error_census_v1.csv','results/trackvggt_candidate_headroom_v1.json',
        'results/trackvggt_event_type_distribution_v1.json','results/trackvggt_scene_difficulty_v1.csv','results/trackvggt_dev5_retrospective_analysis_v1.json','results/trackvggt_full_train_event_inventory_v1.json',
        'manifests/trackvggt_full_audit_plan_v1.json','manifests/trackvggt_expanded_training_v3.json','results/trackvggt_cohort_v2_v3_comparison.json','results/trackvggt_full_train_completion_audit_v1.json']
    save_json(base/'phase_seal_v1.json',dict(status='COMPLETE_AUDIT_AND_LOCKED_V3_INDEX',files={name:sha256(ROOT/name) for name in required},
        teacher_features_new='NOT_CAPTURED',teacher_comparison='NOT_MEASURED',new_large_model_training=False,original_TrackVGGT_goal_continues=True,provenance=provenance()))
    print(json.dumps(dict(status='COMPLETE_AUDIT_AND_LOCKED_V3_INDEX',files=len(required),V3=v3['total_fit_queries'],fit_cross_camera_correctable=cross)))


if __name__=='__main__':main()
