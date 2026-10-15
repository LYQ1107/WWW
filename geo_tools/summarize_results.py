"""Generate status and tables directly from metric artifacts, never enter numbers by hand."""
import csv,json
from common import *

def write_csv(path,rows,fields):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore',lineterminator='\n');writer.writeheader();writer.writerows(rows)

def main():
    metric_rows=[];runtime_rows=[];experiments=[];retrieval_rows=[];scoring_rows=[]
    for path in sorted((RUNTIME/'baseline').glob('*/*.json')):
        if path.name.endswith('_manifest.json') and 'aborted' not in path.name:
            value=json.loads(path.read_text());runtime_rows.append(dict(method=path.parent.name,video_id=value['video_id'],full_video=value['full_video'],causal_online=value['causal_online'],speed_eligible=value.get('speed_eligible',True),
                camera_image_fps=value['camera_image_fps'],synchronized_group_fps=value['synchronized_group_fps'],total_seconds=value['total_seconds'],inference_seconds=value['inference_seconds'],read_preprocess_seconds=value['read_preprocess_seconds'],peak_gpu_bytes=value['peak_gpu_bytes'],
                geo_calls=value.get('geo_calls',0),geo_seconds=value.get('geo_seconds',0),detection_p50=value['detection_latency_p50'],detection_p95=value['detection_latency_p95'],end_to_end_p50=None,end_to_end_p95=None,
                latency_note=('association trace synchronizes tensors; diagnostic timing only' if not value.get('speed_eligible',True) else 'shared GPU contention detected; no fair speed claim' if path.parent.name=='B2_l17_sym_w025_smoke60' else 'full-video offline; detector latency is not online identity latency'),source=str(path)))
            experiments.append(dict(path=str(path),sha256=sha256(path),**{k:v for k,v in value.items() if k not in {'latencies_seconds','async_worker','source_hashes'}}))
        elif path.name.endswith('.metrics.json'):
            value=json.loads(path.read_text());metric_rows.append(dict(method=path.parent.name,video_id=value['video_id'],full_video=value['full_video'],causal_online=value['causal_online'],
                raw_gt_diagnostic=value['allow_duplicate_gt'],**value['metrics'],source=str(path),sha256=sha256(path)))
    # Online runs have a separate manifest and real input-to-identity latency.
    for path in sorted((RUNTIME/'online').glob('*/manifest.json')):
        value=json.loads(path.read_text())
        runtime_rows.append(dict(method=path.parent.name,video_id=value['video_id'],full_video=value['full_video'],causal_online=True,speed_eligible=value.get('speed_eligible',True),
            camera_image_fps=value['camera_image_fps'],synchronized_group_fps=value['synchronized_group_fps'],total_seconds=value['seconds'],
            inference_seconds=None,read_preprocess_seconds=None,peak_gpu_bytes=value['peak_gpu_bytes'],geo_calls=value.get('geo_calls',0),
            geo_seconds=None,detection_p50=None,detection_p95=None,end_to_end_p50=value['input_to_identity_p50'],end_to_end_p95=value['input_to_identity_p95'],
            latency_note='current synchronized group image reading to CPU identity output; static model setup excluded; cold lazy geometry included',source=str(path)))
        experiments.append(dict(path=str(path),sha256=sha256(path),**{k:v for k,v in value.items() if k not in {'latencies_seconds','async_worker','source_hashes'}}))
        for metric_path in path.parent.glob('*.metrics.json'):
            report=json.loads(metric_path.read_text())
            metric_rows.append(dict(method=path.parent.name,video_id=report['video_id'],full_video=report['full_video'],causal_online=report['causal_online'],raw_gt_diagnostic=report['allow_duplicate_gt'],**report['metrics'],source=str(metric_path),sha256=sha256(metric_path)))
    for suite_path in sorted((ROOT/'results').glob('*_suite*_metrics.json')):
        value=json.loads(suite_path.read_text())
        if value.get('status')!='PASS':continue
        metric_rows.append(dict(method=suite_path.stem.removesuffix('_metrics'),video_id='ALL_STRICT',full_video=True,causal_online=value['causal_online'],raw_gt_diagnostic=False,**value['metrics'],source=str(suite_path),sha256=sha256(suite_path)))
    for path in sorted((RUNTIME/'retrieval').glob('*/manifest.json')):
        value=json.loads(path.read_text())
        experiments.append(dict(path=str(path),sha256=sha256(path),**{k:v for k,v in value.items() if k not in {'source_hashes'}}))
        metric_path=path.parent/'retrieval.metrics.json'
        if not metric_path.exists():continue
        report=json.loads(metric_path.read_text())
        if report.get('status')!='PASS':continue
        tracking_vggt=sum(v.get('vggt_calls',0) for v in value.get('tracking_runtime_stats',{}).values())
        retrieval_vggt=sum(v['vggt_calls'] for v in value['extractors'].values())
        tracking_dino=sum(v.get('dinov2_calls',0) for v in value.get('tracking_runtime_stats',{}).values())
        retrieval_dino=sum(v['dinov2_calls'] for v in value['extractors'].values())
        retrieval_rows.append(dict(method=path.parent.name,tracking_method=value.get('tracking_method','gmt_off'),backbone=value['backbone'],video_ids=','.join(map(str,value['video_ids'])),
            camera_streams=value['camera_streams'],full_video=value['full_video'],interval=value['interval'],used_cache=value['used_cache'],speed_eligible=value.get('speed_eligible',True),
            requests=report['requests'],eligible_queries=report['gallery_eligible_queries'],query_detection_coverage=report['gt_query_detection_coverage'],
            gt_matched_queries=report['gt_matched_queries'],detector_query_precision=report['detector_query_precision'],
            gallery_eligibility_among_matched=report['gallery_eligibility_among_matched'],
            all_requests_Recall1=report.get('success_at_k_all_requests',{}).get('1'),
            all_requests_Recall5=report.get('success_at_k_all_requests',{}).get('5'),
            all_requests_Recall10=report.get('success_at_k_all_requests',{}).get('10'),
            Recall1=report['recall_at_k']['1'],Recall5=report['recall_at_k']['5'],Recall10=report['recall_at_k']['10'],MRR=report['mrr'],
            query_p50=value['query_latency_seconds']['p50'],query_p95=value['query_latency_seconds']['p95'],camera_image_fps=value['camera_image_fps'],
            round_p50=value['round_latency_seconds']['p50'],round_p95=value['round_latency_seconds']['p95'],
            vggt_calls=tracking_vggt+retrieval_vggt,tracking_vggt_calls=tracking_vggt,retrieval_vggt_calls=retrieval_vggt,
            dinov2_calls=tracking_dino+retrieval_dino,tracking_dinov2_calls=tracking_dino,retrieval_dinov2_calls=retrieval_dino,
            peak_index_payload_bytes=value['index']['peak_payload_bytes'],peak_host_rss_bytes=value['peak_host_rss_bytes'],peak_gpu_bytes=value['peak_gpu_bytes'],
            source=str(metric_path),sha256=sha256(metric_path)))
        for mode,stats in report.get('scoring_diagnostics',{}).items():
            timing=value.get('scoring_mode_latency_seconds',{}).get(mode,{})
            scoring_rows.append(dict(method=path.parent.name,mode=mode,backbone=value['backbone'],camera_streams=value['camera_streams'],full_video=value['full_video'],
                requests=stats['requests'],eligible_queries=stats['eligible'],candidate_positive_coverage=stats['candidate_positive_coverage'],
                Recall1=stats['recall_at_k']['1'],Recall5=stats['recall_at_k']['5'],Recall10=stats['recall_at_k']['10'],MRR=stats['mrr'],
                query_p50=timing.get('p50'),query_p95=timing.get('p95'),source=str(metric_path),sha256=sha256(metric_path)))
    scoring_fields=['method','mode','backbone','camera_streams','full_video','requests','eligible_queries','candidate_positive_coverage','Recall1','Recall5','Recall10','MRR','query_p50','query_p95','source','sha256']
    write_csv(ROOT/'results/retrieval_scoring_diagnostics.csv',scoring_rows,scoring_fields)
    retrieval_fields=['method','tracking_method','backbone','video_ids','camera_streams','full_video','interval','used_cache','speed_eligible','requests','eligible_queries','query_detection_coverage',
        'gt_matched_queries','detector_query_precision','gallery_eligibility_among_matched','all_requests_Recall1','all_requests_Recall5','all_requests_Recall10',
        'Recall1','Recall5','Recall10','MRR','query_p50','query_p95','camera_image_fps','round_p50','round_p95','vggt_calls','tracking_vggt_calls','retrieval_vggt_calls','dinov2_calls','tracking_dinov2_calls','retrieval_dinov2_calls',
        'peak_index_payload_bytes','peak_host_rss_bytes','peak_gpu_bytes','source','sha256']
    write_csv(ROOT/'results/retrieval_results.csv',retrieval_rows,retrieval_fields)
    relation_path=ROOT/'results/identity_relations_matrix_v2.json'
    if relation_path.exists():
        relation_report=json.loads(relation_path.read_text())
        for path,digest in relation_report['source_artifacts'].items():
            if sha256(path)!=digest:raise ValueError('identity relation source changed; refresh audit: '+path)
        by_suite={row['suite']:row for row in relation_report['rows']}
        for row in metric_rows:
            relation=by_suite.get(row['method']) if row['video_id']=='ALL_STRICT' else None
            if relation:
                if bool(row['causal_online'])!=(relation['protocol']=='causal'):raise ValueError('relation/metric protocol mismatch')
                if row['CVIDF1']!=relation['CVIDF1'] or row['IDSW']!=relation['IDSW']:raise ValueError('relation/metric result mismatch')
                row.update(output_pair_scope='common GT-matched all-time pairs within scenes; not Hungarian decisions',
                    cross_camera_output_pair_fixes=relation['cross_camera_baseline_wrong_method_correct'],
                    cross_camera_output_pair_harms=relation['cross_camera_baseline_correct_method_wrong'],
                    output_pair_matched_detections=relation['matched_detections'])
    metrics=['method','video_id','full_video','causal_online','raw_gt_diagnostic','HOTA','DetA','AssA','IDF1','CVIDF1','CVMA','CVMA_sequential','MOTA','IDSW',
             'output_pair_scope','cross_camera_output_pair_fixes','cross_camera_output_pair_harms','output_pair_matched_detections','source','sha256']
    runtime=['method','video_id','full_video','causal_online','speed_eligible','camera_image_fps','synchronized_group_fps','total_seconds','inference_seconds','read_preprocess_seconds','peak_gpu_bytes','geo_calls','geo_seconds','detection_p50','detection_p95','end_to_end_p50','end_to_end_p95','latency_note','source']
    write_csv(ROOT/'results/ablation_results.csv',metric_rows,metrics);write_csv(ROOT/'results/runtime_results.csv',runtime_rows,runtime)
    save_json(ROOT/'results/experiment_manifest.json',dict(status='IN_PROGRESS',experiments=experiments,raw_gt_policy='unchanged, diagnostic and strict protocols separate',split=json.loads((ROOT/'manifests/geo_split_v1.json').read_text())))
    baseline=[r for r in metric_rows if r['method'].startswith('B0') and isinstance(r['video_id'],int) and r['full_video'] and not r['raw_gt_diagnostic']]
    signal_path=ROOT/'results/causal_B7_dense_positive_scene1_v1.json'
    signal=json.loads(signal_path.read_text()) if signal_path.exists() else None
    def evidence(name,fields):
        path=ROOT/'results'/name
        if not path.exists():return None
        value=json.loads(path.read_text())
        return dict(path=str(path),sha256=sha256(path),**{key:value[key] for key in fields if key in value})
    attribution=evidence('causal_dense_scene1_backbone_controls_partial_v1.json',['status','scope'])
    camera_control=evidence('camera_index_corrected_scene1_accuracy_audit_v1.json',['status','video_id','exact_serialized_predictions','metric_delta','interpretation'])
    cached_prefix=evidence('target_cache_actual_gpu_prefix40_60_v1.json',['status','groups','exact_identity_output_prefix','exact_scheduler_prefix','actual_vggt_calls','limitations'])
    cached_full=evidence('target_cache_full_scene1_actual_budget_accuracy_v1.json',['status','baseline','rows','speed_scope'])
    cached_controls=evidence('target_cache_three_backbone_scene1_controls_v1.json',['status','all_exact_baseline_predictions','same_selected_observation_groups','scope','timing_scope'])
    settings_audit=evidence('suite_settings_audit_v1.json',['status','inconsistent_suites','unverified_default_suites','scope','limitations'])
    complete_off=evidence('OFF_c214_full_strict_prediction_parity_v1.json',['status','full_strict_suite','scope'])
    sparse_trace=evidence('causal_dense_vs_cache_scene1_actual_trace_v1.json',['status','groups','first_dense_partition_change','trace_summaries','limitations'])
    anchor_diagnostic=evidence('causal_dense_scene1_scene10_anchor_diagnostic_v1.json',['status','video_ids','full_strict_suite','scope','limitations','unproven_cause_categories'])
    resident_structure=evidence('vggt_resident_parameter_structure_v1.json',['status','full_official_parameters','resident_parameters','parameter_tensor_bytes_default_fp32','scope','limitations'])
    complete_zero=evidence('ZERO_c214_full_strict_prediction_parity_v1.json',['status','full_strict_suite','scope'])
    post_cache_off=evidence('OFF_post_cache_trace_strict_v1_parity.json',['status','full_strict_suite','scope','rows'])
    post_cache_zero=evidence('ZERO_post_cache_trace_strict_v1_parity.json',['status','full_strict_suite','scope','rows'])
    constant_prefix=evidence('cache_gate_and_constant_prefix_audit_v1.json',['status','groups','constant_evidence_exact_dense_identity_prefix','equal_actual_vggt_calls','equal_feature_validity_and_observations','interpretation','limitations'])
    constant_native=evidence('constant_native_head_scene1_accuracy_v1.json',['status','baseline','rows','speed_scope'])
    link_name='causal_dense_full_dev_anchor_output_links_v1.json' if (ROOT/'results/causal_dense_full_dev_anchor_output_links_v1.json').exists() else 'causal_dense_first3_scenes_anchor_output_links_v1.json'
    linear_links=evidence(link_name,['status','video_ids','full_strict_suite','scope','limitations'])
    headfree_prefix=evidence('native_prior_prefix_gate_v1.json',['status','scope','rows'])
    headfree_scene=evidence('native_prior_scene1_accuracy_v1.json',['status','baseline','rows','speed_scope'])
    motion_budget=evidence('causal_motion_vs_fixed_full_budget_audit_v1.json',['status','matched_actual_vggt_calls_for_each_scene','scope','limitations'])
    dense=evidence('causal_B7_fixed_k1_strict_suite_v1_metrics.json',['status','video_ids','full_video','causal_online','metrics'])
    dense_pairs=evidence('causal_B7_dense_full_strict_identity_pair_audit_v1.json',['status','full_strict_suite','rows','limitations'])
    dense_budget=evidence('causal_dense_fixed_full_dev_budget_v1.json',['status','methods','limitations'])
    native_full=evidence('native_prior_seed20261014_strict_suite_v1_metrics.json',['status','video_ids','full_video','causal_online','metrics'])
    native_pairs=evidence('native_prior_full_dev_identity_pair_audit_v1.json',['status','full_strict_suite','rows','limitations'])
    native_budget=evidence('native_prior_vs_vggt_full_dev_budget_v1.json',['status','methods','limitations'])
    native_verified=bool(native_full and native_pairs and native_budget and native_full['status']=='PASS' and native_pairs['status']=='PASS' and native_pairs['full_strict_suite'] and all(row['added_detections']==0 and row['removed_detections']==0 for row in native_pairs['rows']))
    identity_trigger=evidence('causal_identity_total80_strict_suite_v1_metrics.json',['status','video_ids','metrics'])
    trigger_budget=evidence('causal_trigger_full_dev_realized_budget_v1.json',['status','methods','matched_actual_vggt_calls_for_each_scene','limitations'])
    retrieval_parity=evidence('retrieval_10streams_full_tracking_output_parity_v1.json',['status','full_video','full_strict_suite','scope','limitations'])
    encoder_reference=evidence('fast_vggt_short_pair_v2_comparison.json',['status','scope','failures','sources'])
    encoder_bridge=evidence('fast_vggt_short_pair_v3_comparison.json',['status','scope','rows','limitations'])
    encoder_fp16=evidence('fast_vggt_short_pair_fp16_v4_comparison.json',['status','scope','rows','explicit_bfloat16_constants','source_diff_sha256','limitations'])
    history_training=evidence('history_rank_train_audit_v1.json',['status','unique_vggt_primary_gate','native_rank_errors','native_error_unique_query_observations','actual_whole_command_seconds','collectors','scope','limitations'])
    retrieval_prior_gate=evidence('tracking_retrieval_prefix_gate_v1.json',['status','scope','limitations'])
    prior_retrieval_parity=evidence('native_prior_retrieval_full18_tracking_parity_v1.json',['status','full_video','full_strict_suite','scope','limitations'])
    prior_multistream_parity=evidence('native_prior_retrieval_full4_10_tracking_parity_v1.json',['status','full_video','full_strict_suite','scope','limitations','rows'])
    camera_full_control=evidence('camera_index_corrected_full_strict_audit_v2.json',['status','full_strict_suite','rows','scope','limitations'])
    camera_live_hook=evidence('live_camera_index_hook_probe_v1.json',['status','scope','observations'])
    whole_camera_order=evidence('live_whole_gmt_camera_order_v1.json',['status','full_video','max_frames','scope','limitations','rows'])
    native_seed_stability=evidence('native_prior_inference_seed_stability_v1.json',['status','completed_inference_seeds','pending_inference_seeds','paired_cvidf1_delta_range','scope','limitations'])
    relation_matrix=evidence('identity_relations_matrix_v2.json',['status','pending','scope','limitations'])
    dense_geometry_attribution=evidence('causal_dense_geometry_full_strict_attribution_v1.json',['status','full_strict_suite','scope','conclusion','limitations','rows'])
    dense_full2_retrieval=evidence('vggt_dense_retrieval_full2_tracking_parity_v1.json',['status','full_video','full_strict_suite','scope','limitations'])
    retrieval_query_pairs=evidence('retrieval_paired_query_outcomes_v1.json',['status','rows','pending','scope','limitations'])
    readonly_preflight=evidence('preflight_readonly_locked_inputs_v3.json',['status','locked_files_unchanged','raw_gt_files_verified','model_strict_load_performed','new_video_inference_performed','scope'])
    geometry_failures=evidence('geometry_failure_taxonomy_v1.json',['status','summaries','source_rule_snapshots','gt_used_for_gate_classification','limitations'])
    dense_retrieval_gate=evidence('vggt_tracking_retrieval_prefix_gate_v2.json',['status','full_video','full_strict_suite','scope','limitations'])
    order_parity=evidence('complete_causal_vggt_view_order_audit_v2.json',['status','scope','limitations','rows'])
    dense_verified=bool(dense and dense_pairs and dense_budget and dense['status']=='PASS' and dense_pairs['status']=='PASS' and dense_pairs['full_strict_suite'] and all(row['added_detections']==0 and row['removed_detections']==0 for row in dense_pairs['rows']))
    if linear_links:
        data=json.loads(Path(linear_links['path']).read_text())
        linear_links['per_scene_recent_output_link_counts']=[dict(video_id=row['video_id'],**row['recent_other_camera_output_links']['counts']) for row in data['rows']]
    resource_ledger=evidence('resource_ledger_v1.json',['status','scopes'])
    tiny_path=ROOT/'results/causal_B8_fixed20_total80_strict_suite_v1_metrics.json'
    tiny_audit_path=ROOT/'results/causal_B8_fixed20_total80_identity_pair_audit_v1.json'
    tiny=None
    if tiny_path.exists() and tiny_audit_path.exists():
        candidate=json.loads(tiny_path.read_text());audit=json.loads(tiny_audit_path.read_text())
        base=json.loads((ROOT/'results/causal_B0_strict_suite_v1_metrics.json').read_text())['metrics']
        strict=sorted(json.loads((ROOT/'manifests/geo_split_v1.json').read_text())['strict_development_video_ids'])
        fixes=sum(row['cross_camera_all_time_identity_pairs'][key] for row in audit['rows'] for key in ['fixes_false_negative_pairs','fixes_false_positive_pairs'])
        harms=sum(row['cross_camera_all_time_identity_pairs'][key] for row in audit['rows'] for key in ['harms_true_positive_pairs','harms_true_negative_pairs'])
        if candidate['status']=='PASS' and candidate['full_video'] and candidate['causal_online'] and candidate['video_ids']==strict and audit['status']=='PASS' and audit.get('full_strict_suite') and candidate['metrics']['CVIDF1']>base['CVIDF1'] and fixes>harms and all(row['added_detections']==0 and row['removed_detections']==0 for row in audit['rows']):
            tiny=candidate
    science_status='PRELIMINARY_PAIRED_FULL_DEV_GAIN_NOT_STABLE' if tiny else 'NOT_ESTABLISHED'
    summary=dict(status='IN_PROGRESS',scientific_success=science_status,engineering_complete=False,paper_ready=False,
        complete_strict_baseline_videos=sorted({r['video_id'] for r in baseline}),metrics=metric_rows,feature_screening_complete=(ROOT/'results/feature_screening_manifest.json').exists(),
        off_zero_parity=json.loads((ROOT/'results/off_zero_parity.json').read_text()) if (ROOT/'results/off_zero_parity.json').exists() else None,
        limitations=['original GMT offline full-video postprocessing','conflicting raw GT identities in several original TRAIN sequences','Dense B7 and sparse B8 have preliminary full-strict numerical identity gains, not stable method improvement; scene1 attribution does not establish independent VGGT/geometry benefit; primary capacity controls complete, dense stability pending','WWW application evidence pending'],
        preliminary_complete_video_signal=signal,
        preliminary_full_dev_signal=tiny,
        scene1_attribution=attribution,camera_encoding_control=camera_control,cached_feature_prefix=cached_prefix,cached_feature_complete_scene=cached_full,resource_ledger=resource_ledger,
        cached_three_backbone_controls=cached_controls,recorded_suite_settings=settings_audit,original_matrix_complete_off=complete_off,
        sparse_failure_trace=sparse_trace,identity_anchor_diagnostic=anchor_diagnostic,vggt_resident_parameter_structure=resident_structure,
        original_matrix_complete_zero=complete_zero,encoder_matched_constant_prefix=constant_prefix,constant_native_head_control=constant_native,
        post_cache_source_complete_off=post_cache_off,post_cache_source_complete_zero=post_cache_zero,
        recent_other_camera_output_link_diagnostic=linear_links,
        headfree_native_prior_prefix=headfree_prefix,motion_fixed_realized_budget_audit=motion_budget,
        headfree_native_prior_complete_scene=headfree_scene,
        dense_full_dev_signal=dense if dense_verified else None,dense_full_dev_identity_audit=dense_pairs,dense_full_dev_budget=dense_budget,
        headfree_native_prior_full_dev=native_full if native_verified else None,headfree_native_prior_full_dev_identity_audit=native_pairs,
        headfree_native_prior_full_dev_budget=native_budget,
        identity_trigger_full_dev=identity_trigger,trigger_realized_budgets=trigger_budget,
        complete_ten_stream_tracking_parity=retrieval_parity,fast_encoder_original_interface_reference=encoder_reference,
        fast_encoder_camera_bridge_reference=encoder_bridge,bounded_train_history_recovery=history_training,
        fast_encoder_fp16_reference=encoder_fp16,
        tracking_to_retrieval_prior_prefix_gate=retrieval_prior_gate,
        native_prior_complete_scene_retrieval_tracking_parity=prior_retrieval_parity,
        native_prior_complete_multistream_retrieval_tracking_parity=prior_multistream_parity,
        corrected_camera_complete_strict_control=camera_full_control,
        actual_camera_index_mlp_hook=camera_live_hook,whole_gmt_camera_order_diagnostic=whole_camera_order,
        native_prior_paired_inference_seed_stability=native_seed_stability,complete_ablation_output_relations=relation_matrix,
        complete_primary_dense_geometry_attribution=dense_geometry_attribution,
        dense_b7_complete_two_camera_retrieval_tracking_parity=dense_full2_retrieval,
        paired_retrieval_query_outcomes=retrieval_query_pairs,readonly_reproduction_preflight=readonly_preflight,
        recorded_geometry_failure_taxonomy=geometry_failures,
        dense_tracking_to_retrieval_prefix_gate=dense_retrieval_gate,
        complete_encoder_input_order_control=order_parity,
        retrieval_experiments=retrieval_rows,retrieval_scoring_diagnostics=scoring_rows,retrieval_full_video_evaluation=any(row['full_video'] for row in retrieval_rows),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    save_json(ROOT/'results/final_summary.json',summary)
    lines=['# VGGT+GMT report (automatically generated; research in progress)','',
        f'SCIENTIFIC_SUCCESS: {science_status}. ENGINEERING_COMPLETE: false. PAPER_READY: false.','',
        'Tables below are read from actual evaluation files. Raw-GT diagnostic rows cannot substitute for strict evaluation. All input frames are retained. Original GMT full-video postprocessing is offline; no real-time identity latency claimed. For synchronized two-camera tracking, per-camera delivered stream FPS equals group FPS; aggregate camera-image throughput is twice that value.','',
        'All new fitting/selection uses original TRAIN scenes. The frozen GMT checkpoint was trained on original TRAIN including these development scenes, so this split isolates new-head supervision and is not backbone-unseen generalization. Historical official TEST exposure is disclosed in geo_split_v1.json; this project does not use TEST to select/calibrate its new methods.','',
        '| Method | Video | Protocol | Extent | Strict GT | HOTA | AssA | IDF1 | CVIDF1 | IDSW |','|---|---:|---|---|---|---:|---:|---:|---:|---:|']
    if signal:
        row=signal['rows'][0];delta=row['metric_delta']
        attribution_note=(' Scene1 attribution is complete: feature-only exactly equals joint; masked geometry-only exactly equals baseline; DINO matches VGGT CVIDF1 with one fewer IDSW; a GMT-only head captures much of the gain. Independent VGGT/geometry credit is unsupported.' if attribution and attribution['status']=='COMPLETE_SCENE1_ATTRIBUTION_NOT_FULL_DEV' else ' Independent attribution controls are pending.')
        lines[6:6]=['Complete causal scene1 preliminary signal: dense B7 has paired changes HOTA '+f"{delta['HOTA']:+.4f}, AssA {delta['AssA']:+.4f}, CVIDF1 {delta['CVIDF1']:+.4f} percentage points, IDSW {delta['IDSW']:+d}."+' Exact detector-key parity is audited separately.'+attribution_note+' Primary full-dev GMT/DINO capacity controls are complete; repeated dense stability remains pending.','']
    if camera_control and camera_control['exact_serialized_predictions']:
        lines[6:6]=['The inherited single-image camera encoding issue is confirmed in source. Actual corrected-camera GMT-only scene1 predictions exactly equal the locked causal baseline; all metric differences are zero. The repair does not explain this scene1 head gain. Other strict scenes remain pending.','']
    if cached_prefix and cached_prefix['status']=='PASS':
        lines[6:6]=['Actual cached-feature VGGT40/60-group GPU runs pass exact40-group identity-output and scheduler-prefix equality with active target reuse (4/6 actual VGGT calls). This is a short-prefix correctness result, not a full-video accuracy, speed, or whole-process memory claim. Per-target source-age traces were not recorded.','']
    if cached_full and cached_full['status']=='PASS':
        row=cached_full['rows'][0]
        if row['exact_baseline_predictions']:
            controls_note=(' All three cached backbone controls now complete and exactly equal baseline under the same40 selected observation groups (VGGT80 / DINO40 / GMT0 actual encoder calls; different compute).' if cached_controls and cached_controls['all_exact_baseline_predictions'] else ' Other cached backbone controls remain pending.')
            trace_note=(' Bounded dense/cache traces preserve production output prefixes; the first split at dataset frame27 has no valid cached feature for the affected query, while dense fusion changes one same-live-history counterfactual assignment. Gate telemetry and fixed-score prefix controls are recorded below.' if sparse_trace and sparse_trace['status']=='PASS' else ' Bounded failure traces remain pending.')
            lines[6:6]=[f"Complete cached-feature scene1 run: {row['actual_encoder_calls']} actual VGGT calls, exact baseline identities and unchanged metrics; no retained dense identity gain. Measured {row['camera_image_fps']:.4f} images/s and P95 {row['input_to_identity_p95']*1000:.2f}ms are one descriptive full-pipeline run, not stable speed superiority."+controls_note+trace_note,'']
    if complete_off and complete_off['status']=='PASS':
        zero_note=' Both OFF and zero-weight complete strict5 controls pass.' if complete_zero and complete_zero['status']=='PASS' else ' Complete zero-weight control remains pending.'
        later_note=' Post-cache/diagnostic-source complete OFF/zero regressions remain pending.'
        if post_cache_off and post_cache_off['status']=='PASS':
            sources=sorted({row['source_commits'][1][:8] for row in post_cache_off['rows']})
            later_note=' Later post-cache source '+','.join(sources)+' also passes complete strict5 OFF raw equality.'
            later_note+=(' Its zero-weight full regression also passes.' if post_cache_zero and post_cache_zero['status']=='PASS' else ' Its zero-weight full regression remains running/pending.')
        lines[6:6]=['Exact complete strict5 OFF prediction parity passes for the original immutable c214 source, including every serialized box/score/category/image/track ID and row order.'+zero_note+later_note+' Proof is limited to the explicitly recorded source versions.','']
    if constant_native and constant_native['status']=='PASS':
        row=constant_native['rows'][0]
        lines[6:6]=[f"Complete scene1 native-head constant-score control improves paired CVIDF1 {row['metric_delta']['CVIDF1']:+.6f} percentage points with IDSW {row['metric_delta']['IDSW']:+d}, exceeding current learned VGGT scene1 identity metrics. It retains frozen native-head encoding for eligibility but replaces learned pair evidence by fixed+1; no VGGT calls. This is a single-scene score ablation with diagnostic tracing, not a production-speed or zero-extra-parameter claim. Head-free complete strict5 is reported above; paired inference-seed controls remain pending.",'']
    if constant_prefix and constant_prefix['constant_evidence_exact_dense_identity_prefix']:
        lines[6:6]=['In the actual80-call VGGT40-group prefix, fixed+1 evidence exactly reproduces learned dense identity outputs under identical feature eligibility. Learned score discrimination is unnecessary for that early split; whole-video/DEV attribution is not inferred. Cache gate telemetry exactly reproduces all decisions: all74 rejected reuse observations fail only the fixed native cosine .9 gate; spatial candidate filtering does not change admission. No DEV threshold relaxation is applied.','']
    if headfree_prefix and headfree_prefix['status']=='PASS':
        full_note=(' Complete head-free scene1 now exactly reproduces the constant-head output SHA, with CVIDF1 '+f"{headfree_scene['rows'][0]['metrics']['CVIDF1']:.6f}"+f", IDSW {headfree_scene['rows'][0]['metrics']['IDSW']}, descriptive throughput {headfree_scene['rows'][0]['camera_image_fps']:.4f} images/s and P95 {headfree_scene['rows'][0]['input_to_identity_p95']*1000:.2f}ms. This is one inference seed/scene, not a stable speed or full-DEV result." if headfree_scene else ' Complete head-free evaluation remains pending.')
        lines[6:6]=['The head-free fixed-prior implementation now passes actual40-group equality to the measured native-head constant-score control and zero60-group equality to locked causal GMT. No extra encoder/head or learned parameters are loaded, and the extra target-vector width is zero.'+full_note+' Complete seed-paired inference evaluations remain running/queued; full strict5 primary-seed status is reported above. This is a strong ordinary calibration control, not a VGGT-specific contribution.','']
    if tiny:
        base=json.loads((ROOT/'results/causal_B0_strict_suite_v1_metrics.json').read_text())['metrics']
        delta=tiny['metrics']['CVIDF1']-base['CVIDF1']
        lines[6:6]=[f'Complete strict B8 fixed20/80-call-per-video CVIDF1 change is only {delta:+.6f} percentage points, with unchanged total IDSW. The common-detection identity audit finds one small scene19 relation correction and no harms. This is an extremely small preliminary full-DEV signal; no stable VGGT/geometry/scheduler contribution is established.','']
    if dense_verified:
        base=json.loads((ROOT/'results/causal_B0_strict_suite_v1_metrics.json').read_text())['metrics']
        delta={key:dense['metrics'][key]-base[key] for key in dense['metrics']}
        measured=dense_budget['methods'][0]
        lines[6:6]=[f"Complete causal dense B7 strict5: CVIDF1 {dense['metrics']['CVIDF1']:.6f} ({delta['CVIDF1']:+.6f} percentage points), HOTA {delta['HOTA']:+.6f}, AssA {delta['AssA']:+.6f}, IDSW {base['IDSW']} to {dense['metrics']['IDSW']}. All five scenes retain exact detector-key sets. Cross-camera all-time identity-pair audits include both corrections and harms; these correlated pair counts are not Hungarian decision counts. Actual VGGT calls {measured['total_actual_vggt_calls']}, numerical geometry gate passes {measured['total_valid_geometry']}, module failures {measured['total_module_failures']}; descriptive cumulative throughput {measured['cumulative_camera_image_fps']:.4f} images/s. One seed, incomplete full-dev attribution, and a stronger full-dev native fixed-prior rival prevent a stable or VGGT-specific success claim.",'']
    if native_verified:
        base=json.loads((ROOT/'results/causal_B0_strict_suite_v1_metrics.json').read_text())['metrics']
        measured=native_budget['methods'][0]
        lines[6:6]=[f"Complete head-free native fixed-prior strict5: CVIDF1 {native_full['metrics']['CVIDF1']:.6f} ({native_full['metrics']['CVIDF1']-base['CVIDF1']:+.6f} percentage points), IDSW {base['IDSW']} to {native_full['metrics']['IDSW']}; exact detector-key sets in all five scenes, with corrections and harms retained separately. No extra learned parameters/encoder/head or target-vector columns. Descriptive cumulative throughput {measured['cumulative_camera_image_fps']:.4f} images/s, zero VGGT calls. This ordinary camera/time logit prior exceeds primary-seed dense VGGT full-dev CVIDF1 and IDSW; current evidence does not establish VGGT-specific identity gain. Paired inference-seed controls remain pending; no stable timing, final candidate selection or three-module success claim.",'']
    if identity_trigger and trigger_budget:
        base=json.loads((ROOT/'results/causal_B0_strict_suite_v1_metrics.json').read_text())['metrics']
        measured=next(row for row in trigger_budget['methods'] if 'causal_identity_' in row['source'])
        lines[6:6]=[f"Identity-only trigger complete strict5: CVIDF1 {identity_trigger['metrics']['CVIDF1']-base['CVIDF1']:+.6f} percentage points and IDSW {identity_trigger['metrics']['IDSW']-base['IDSW']:+d}; actual VGGT calls {measured['total_actual_vggt_calls']}. Fixed/joint each use400 calls, motion uses0, so this is not an equal-realized-budget identity-trigger superiority result. Its full common-detection audit localizes the identity corrections to scene1; repeated stability and VGGT-specific attribution are unproved.",'']
    if retrieval_parity and retrieval_parity['status']=='PASS' and retrieval_parity['full_strict_suite']:
        lines[6:6]=['Both complete ten-camera native and VGGT-feature retrieval producers preserve every standalone causal GMT OFF output across all five scenes, including boxes/scores/classes/image/track IDs and row order. This verifies complete interleaved tracking output isolation for these runs; their retrieval ranking quality and whole-producer runtime remain separate.','']
    if encoder_reference and encoder_reference['status']=='NO_VALID_SPEED_COMPARISON':
        lines[6:6]=['The original/Fast three-TRAIN-pair microbenchmark v2 cannot support a speed comparison: original passes36 calls; official Fast fails at camera-head LayerNorm on its BF16 output under V100. The failure remains archived separately from subsequent camera/precision bridge tests.','']
    if encoder_bridge and encoder_bridge['status']=='PASS_MICROBENCHMARK_ONLY':
        ratios=[row['original_over_fast_ratio'] for row in encoder_bridge['rows']]
        lines[6:6]=[f"The FP32 camera-input bridge v3 passes36 calls per variant. Original/Fast timing ratios range {min(ratios):.4f}–{max(ratios):.4f}: the available Fast path is slower on this V100 setup. Fast still forces BF16 internally, so v3 is not a precision-matched architectural speed conclusion; the separate FP16 reference is reported independently.",'']
    if encoder_fp16 and encoder_fp16['status']=='PASS_MICROBENCHMARK_ONLY':
        ratios=[row['original_over_fast_ratio'] for row in encoder_fp16['rows']]
        lines[6:6]=[f"Controlled FP16 original/Fast reference v4 completes36 calls per variant, with only10 Fast dtype constants changed in a separate two-file source worktree and identical FP32 camera-head input policy. Original/Fast ratios {min(ratios):.6f}–{max(ratios):.6f}: the available Fast implementation is about {1/max(ratios):.2f}–{1/min(ratios):.2f} times slower for these short two-view V100 fixtures; saved target features differ. No current production integration follows. This is not a long-sequence, other-hardware or full-system speed claim.",'']
    if history_training and history_training['status']=='PASS_BOUNDED_TRAIN_DIAGNOSTIC':
        lines[6:6]=[f"Bounded history-ranking TRAIN recovery completes three fresh sparse200-frame collectors, real neutral OFF40-prefix checks and9 matched heads (3 backbones ×3 seeds). Primary unique-VGGT gate is {history_training['unique_vggt_primary_gate']}; selected heads make no new corrections/harms against native on1,497 reused TRAIN12 reference-frame clauses ({history_training['native_rank_errors']} native errors from {history_training['native_error_unique_query_observations']} query observations). No new DEV tuning follows this failed gate. Whole command cost {history_training['actual_whole_command_seconds']:.2f}s, target/event artifacts about11.5MB; this is offline TRAIN cost, not inference speed or proof all VGGT features lack useful information.",'']
    if prior_retrieval_parity and prior_retrieval_parity['status']=='PASS':
        prior_row=next((row for row in retrieval_rows if row['method']=='retrieval_native_prior_2streams_full_v1'),None)
        if prior_row:
            lines[6:6]=[f"The complete head-free fixed-prior scene18 publication control preserves the entire standalone prior tracking output, with {prior_row['requests']} queries/{prior_row['eligible_queries']} eligible, Recall1 {prior_row['Recall1']:.6f}, MRR {prior_row['MRR']:.6f}, descriptive whole-producer {prior_row['camera_image_fps']:.4f} images/s. Recall1 equals the native tracker reference; the MRR difference is small. This simple-control integration does not establish VGGT tracking-to-retrieval efficacy.",'']
    if prior_multistream_parity and prior_multistream_parity['status']=='PASS':
        lines[6:6]=['Complete fixed-prior four/ten-camera producers preserve every standalone prior tracking output in their respective scenes; the ten-camera run covers all strict5. Fixed scene18 query/detection cohorts are checked separately in WWW_SCOPE_NOTE.md. Full-suite status on the joint parity report is false because the four-camera run intentionally contains only two scenes; no missing ten-camera scenes are implied. Dense VGGT publication completion is reported separately.','']
    if camera_full_control:
        changed=sum(row.get('changed_track_id_rows') or 0 for row in camera_full_control['rows'])
        live_note=' Its live spatial-MLP hook remains a separate bounded check.'
        if camera_live_hook and camera_live_hook['status']=='PASS':
            live_note=' The actual checkpoint/image hook probe passes all8 first-four-group calls: camera scalars0/0 become0/.5, other10 columns remain exact, and the second-camera frozen MLP output changes. This verifies the declared inference path at that bounded scope.'
        lines[6:6]=[f"Corrected frozen GMT camera-index control completes all strict5 with identical aggregate metrics. Four scenes have exact predictions; scene18 changes {changed} track-ID rows with unchanged detector rows, so full prediction parity is not established."+live_note+' Whole-GMT input-order diagnostics are reported separately.','']
    if whole_camera_order:
        lines[6:6]=['Actual whole-GMT scene1 forty-group reversal changes identity partitions for both OFF and dense B7, even after physical camera IDs/output positions are restored and global ID renaming is allowed. Both canonical variants exactly match their standalone prefixes; detector membership is unchanged; B7 computes80 VGGT calls per variant, with0 valid geometry calls/failures. Thus internal sequential tracker order invariance is contradicted at this scope. An explicit physical-ID canonical input adapter is separately unit-tested and queued for real reverse/alternate caller-order validation; no internal tracker or baseline change is hidden.','']
    if native_seed_stability:
        number=len(native_seed_stability['completed_inference_seeds']);limits=native_seed_stability['paired_cvidf1_delta_range']
        lines[6:6]=[f"Head-free fixed-prior paired inference-RNG trials: {number}/3 complete strict5 pairs audited with identical detector membership. Paired CVIDF1 changes range {limits[0]:+.6f} to {limits[1]:+.6f} percentage points. These use fixed weights and vary actual inference RNG, independently of the VGGT TRAIN-head-seed experiment; pending trials are {native_seed_stability['pending_inference_seeds']}. No VGGT attribution, confidence interval or speed advantage follows.",'']
    if relation_matrix:
        lines[6:6]=['Complete available primary ablation suites now include independently verified cross-camera output-pair fixes/harms in ablation_results.csv; detailed protocol-separated counts and pending suites are in [GEO_STREAM_IDENTITY_RELATIONS.md](GEO_STREAM_IDENTITY_RELATIONS.md). These conditional unordered GT-matched identity relations can scale quadratically with track length and are not internal Hungarian decision counts.','']
    if dense_geometry_attribution and dense_geometry_attribution['status']=='EXACT_FEATURE_JOINT_FULL_STRICT_OUTPUTS':
        lines[6:6]=['Complete primary-head dense B5/B7 geometry attribution is now audited over all strict5: every serialized feature-only prediction exactly equals joint B7, with matched encoder calls and all other effective GEO options verified from their captured source defaults (calibration content checked, inactive default-off cache normalized). Joint geometry passes its numerical/background gate3 times, all in scene1; none changes the final outputs. This extends the earlier scene1 result to the complete primary-head suite; remaining head seeds and valid-pose accuracy are separate.','']
    if geometry_failures:
        count=sum(row['diagnostic_rows'] for row in geometry_failures['summaries'])
        accepted=sum(row['valid'] for row in geometry_failures['summaries'])
        lines[6:6]=[f"Geometry failure recovery is consolidated across {count} fixed screening rows ({accepted} accepted), separating fit support, logged coverage, measured residual violations and unavailable early-return statistics. Mask/feature variants repeat samples; these counts do not replace the separate complete-run3 numerical gate passes. Captured dirty source-file bytes are matched to Git blobs where available, while original tree dirtiness and weaker legacy provenance remain explicit. See [geometry gate diagnostics](GEO_STREAM_GEOMETRY_FAILURES.md); no physical-overlap, texture, rotation or GT-pose cause is inferred.",'']
    if dense_full2_retrieval and dense_full2_retrieval['status']=='PASS' and retrieval_query_pairs:
        pair=next((row for row in retrieval_query_pairs['rows'] if row['method']=='VGGT2'),None)
        producer=next((row for row in retrieval_rows if row['method']=='retrieval_vggt_dense_tracking_2streams_full_v1'),None)
        if pair and producer:
            lines[6:6]=[f"Complete dense-B7 two-camera tracking-to-retrieval preserves the standalone B7 tracking outputs. On the exact fixed detector cohort, native ranking has Recall1 {pair['method_conditional_recall1']:.6f} versus OFF {pair['baseline_conditional_recall1']:.6f}: {pair['common_eligible_rank1_fixes']} corrected and {pair['common_eligible_rank1_harms']} harmed Recall1 queries among {pair['common_eligible']} common eligible, with eligibility gained/lost {pair['eligible_gained']}/{pair['eligible_lost']}. Actual cost is {producer['vggt_calls']} VGGT calls and descriptive {producer['camera_image_fps']:.6f} camera images/s (static setup excluded). This is a small single-primary-head signal; four-camera publication and matched-head producer controls remain pending. See [paired query audit](GEO_STREAM_RETRIEVAL_QUERY_AUDIT.md); aggregate equality in other controls does not imply query equality.",'']
    for row in metric_rows:
        lines.append(f"| {row['method']} | {row['video_id']} | {'causal' if row['causal_online'] else 'offline'} | {'complete' if row['full_video'] else 'prefix'} | {'no: raw diagnostic' if row['raw_gt_diagnostic'] else 'yes'} | {row['HOTA']:.4f} | {row['AssA']:.4f} | {row['IDF1']:.4f} | {row['CVIDF1']:.4f} | {row['IDSW']} |")
    lines+=['','Causal retrieval (local replay, GT only for post-run evaluation; conditional retrieval quality and coverage reported separately):','',
        '| Method | Cameras | Full video | Queries | Eligible | Conditional Recall@1 | All-request success@1 | MRR | Query P95 (ms) | Complete images/s |',
        '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in retrieval_rows:
        speed=f"{row['camera_image_fps']:.4f}" if row['speed_eligible'] else 'diagnostic cost only'
        all_requests=f"{row['all_requests_Recall1']:.6f}" if row['all_requests_Recall1'] is not None else 'NA'
        lines.append(f"| {row['method']} | {row['camera_streams']} | {row['full_video']} | {row['requests']} | {row['eligible_queries']} | {row['Recall1'] if row['Recall1'] is not None else 'NA'} | {all_requests} | {row['MRR'] if row['MRR'] is not None else 'NA'} | {row['query_p95']*1000:.4f} | {speed} |")
    if scoring_rows:
        lines+=['','Same-query retrieval scoring diagnostics (identical feature-source observations/candidates for encoded scorers; full native-gallery denominator retained):','',
            '| Method | Scorer | Full video | Positive candidate coverage | Recall@1 | MRR |', '|---|---|---|---:|---:|---:|']
        for row in scoring_rows:
            lines.append(f"| {row['method']} | {row['mode']} | {row['full_video']} | {row['candidate_positive_coverage']} | {row['Recall1']} | {row['MRR']} |")
    lines+=['','Retrieval latency is local scoring and serialization, not Internet transport latency. Full pipeline throughput charges image reads, frozen detection/GTA, feature extraction, publication, queries and query-log I/O. Original GMT history is unchanged; the index payload cap does not imply bounded total system memory. Dataset replay and a functioning query interface do not establish WWW novelty or live Web application evidence.','',
        'Required remaining work: finish the declared tracking/scheduler matrix and repeated stability checks; complete retrieval comparisons across stream counts/keyframe budgets; report geometry failure recovery and actual timing without unsupported efficacy or Pareto claims. Completed feature tests, source/pose diagnostics and causal results are evidence with limited scope, not completion of the requested experiment matrix.','',
        'See results/experiment_manifest.json for exact commands, seeds, checkpoint SHA, environment, cache status and external artifact paths. See docs/GEO_STREAM_RESEARCH_LOG.md for commits and failed runs.','']
    (ROOT/'docs/GEO_STREAM_FINAL_REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(dict(strict_baseline_videos=summary['complete_strict_baseline_videos'],metric_rows=len(metric_rows),runtime_rows=len(runtime_rows)),indent=2))
if __name__=='__main__':main()
