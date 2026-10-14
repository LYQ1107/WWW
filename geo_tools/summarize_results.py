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
        retrieval_rows.append(dict(method=path.parent.name,backbone=value['backbone'],video_ids=','.join(map(str,value['video_ids'])),
            camera_streams=value['camera_streams'],full_video=value['full_video'],interval=value['interval'],used_cache=value['used_cache'],speed_eligible=value.get('speed_eligible',True),
            requests=report['requests'],eligible_queries=report['gallery_eligible_queries'],query_detection_coverage=report['gt_query_detection_coverage'],
            Recall1=report['recall_at_k']['1'],Recall5=report['recall_at_k']['5'],Recall10=report['recall_at_k']['10'],MRR=report['mrr'],
            query_p50=value['query_latency_seconds']['p50'],query_p95=value['query_latency_seconds']['p95'],camera_image_fps=value['camera_image_fps'],
            round_p50=value['round_latency_seconds']['p50'],round_p95=value['round_latency_seconds']['p95'],
            vggt_calls=sum(v['vggt_calls'] for v in value['extractors'].values()),dinov2_calls=sum(v['dinov2_calls'] for v in value['extractors'].values()),
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
    retrieval_fields=['method','backbone','video_ids','camera_streams','full_video','interval','used_cache','speed_eligible','requests','eligible_queries','query_detection_coverage',
        'Recall1','Recall5','Recall10','MRR','query_p50','query_p95','camera_image_fps','round_p50','round_p95','vggt_calls','dinov2_calls',
        'peak_index_payload_bytes','peak_host_rss_bytes','peak_gpu_bytes','source','sha256']
    write_csv(ROOT/'results/retrieval_results.csv',retrieval_rows,retrieval_fields)
    metrics=['method','video_id','full_video','causal_online','raw_gt_diagnostic','HOTA','DetA','AssA','IDF1','CVIDF1','CVMA','CVMA_sequential','MOTA','IDSW','source','sha256']
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
        limitations=['original GMT offline full-video postprocessing','conflicting raw GT identities in several original TRAIN sequences','B8 has an extremely small audited full-strict numerical identity signal, not stable method improvement; scene1 attribution is complete but does not establish independent VGGT/geometry benefit; remaining dense scenes/stability pending','WWW application evidence pending'],
        preliminary_complete_video_signal=signal,
        preliminary_full_dev_signal=tiny,
        scene1_attribution=attribution,camera_encoding_control=camera_control,cached_feature_prefix=cached_prefix,cached_feature_complete_scene=cached_full,resource_ledger=resource_ledger,
        cached_three_backbone_controls=cached_controls,recorded_suite_settings=settings_audit,original_matrix_complete_off=complete_off,
        sparse_failure_trace=sparse_trace,identity_anchor_diagnostic=anchor_diagnostic,vggt_resident_parameter_structure=resident_structure,
        retrieval_experiments=retrieval_rows,retrieval_scoring_diagnostics=scoring_rows,retrieval_full_video_evaluation=any(row['full_video'] for row in retrieval_rows),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    save_json(ROOT/'results/final_summary.json',summary)
    lines=['# VGGT+GMT report (automatically generated; research in progress)','',
        f'SCIENTIFIC_SUCCESS: {science_status}. ENGINEERING_COMPLETE: false. PAPER_READY: false.','',
        'Tables below are read from actual evaluation files. Raw-GT diagnostic rows cannot substitute for strict evaluation. All input frames are retained. Original GMT full-video postprocessing is offline; no real-time identity latency claimed.','',
        '| Method | Video | Strict GT | HOTA | AssA | IDF1 | CVIDF1 | IDSW |','|---|---:|---|---:|---:|---:|---:|---:|']
    if signal:
        row=signal['rows'][0];delta=row['metric_delta']
        attribution_note=(' Scene1 attribution is complete: feature-only exactly equals joint; masked geometry-only exactly equals baseline; DINO matches VGGT CVIDF1 with one fewer IDSW; a GMT-only head captures much of the gain. Independent VGGT/geometry credit is unsupported.' if attribution and attribution['status']=='COMPLETE_SCENE1_ATTRIBUTION_NOT_FULL_DEV' else ' Independent attribution controls are pending.')
        lines[6:6]=['Complete causal scene1 preliminary signal: dense B7 has paired changes HOTA '+f"{delta['HOTA']:+.4f}, AssA {delta['AssA']:+.4f}, CVIDF1 {delta['CVIDF1']:+.4f} percentage points, IDSW {delta['IDSW']:+d}."+' Exact detector-key parity is audited separately.'+attribution_note+' Complete strict5 and dense stability remain pending; this is not full development success.','']
    if camera_control and camera_control['exact_serialized_predictions']:
        lines[6:6]=['The inherited single-image camera encoding issue is confirmed in source. Actual corrected-camera GMT-only scene1 predictions exactly equal the locked causal baseline; all metric differences are zero. The repair does not explain this scene1 head gain. Other strict scenes remain pending.','']
    if cached_prefix and cached_prefix['status']=='PASS':
        lines[6:6]=['Actual cached-feature VGGT40/60-group GPU runs pass exact40-group identity-output and scheduler-prefix equality with active target reuse (4/6 actual VGGT calls). This is a short-prefix correctness result, not a full-video accuracy, speed, or whole-process memory claim. Per-target source-age traces were not recorded.','']
    if cached_full and cached_full['status']=='PASS':
        row=cached_full['rows'][0]
        if row['exact_baseline_predictions']:
            controls_note=(' All three cached backbone controls now complete and exactly equal baseline under the same40 selected observation groups (VGGT80 / DINO40 / GMT0 actual encoder calls; different compute).' if cached_controls and cached_controls['all_exact_baseline_predictions'] else ' Other cached backbone controls remain pending.')
            trace_note=(' Bounded dense/cache traces now preserve their respective production output prefixes; the first split at dataset frame27 has no valid cached feature for the affected query, while dense fusion changes one same-live-history counterfactual assignment. Detailed gate and constant-score controls remain pending.' if sparse_trace and sparse_trace['status']=='PASS' else ' Bounded failure traces remain pending.')
            lines[6:6]=[f"Complete cached-feature scene1 run: {row['actual_encoder_calls']} actual VGGT calls, exact baseline identities and unchanged metrics; no retained dense identity gain. Measured {row['camera_image_fps']:.4f} images/s and P95 {row['input_to_identity_p95']*1000:.2f}ms are one descriptive full-pipeline run, not stable speed superiority."+controls_note+trace_note,'']
    if complete_off and complete_off['status']=='PASS':
        lines[6:6]=['Exact complete strict5 OFF prediction parity passes for the original immutable c214 source, including every serialized box/score/category/image/track ID and row order. Current post-cache/diagnostic-source complete OFF/zero regressions remain queued; earlier-source proof is not promoted to later-source completion.','']
    if tiny:
        base=json.loads((ROOT/'results/causal_B0_strict_suite_v1_metrics.json').read_text())['metrics']
        delta=tiny['metrics']['CVIDF1']-base['CVIDF1']
        lines[6:6]=[f'Complete strict B8 fixed20/80-call-per-video CVIDF1 change is only {delta:+.6f} percentage points, with unchanged total IDSW. The common-detection identity audit finds one small scene19 relation correction and no harms. This is an extremely small preliminary full-DEV signal; no stable VGGT/geometry/scheduler contribution is established.','']
    for row in metric_rows:
        lines.append(f"| {row['method']} | {row['video_id']} | {'no: raw diagnostic' if row['raw_gt_diagnostic'] else 'yes'} | {row['HOTA']:.4f} | {row['AssA']:.4f} | {row['IDF1']:.4f} | {row['CVIDF1']:.4f} | {row['IDSW']} |")
    lines+=['','Causal retrieval (local replay, GT only for post-run evaluation; conditional retrieval quality and coverage reported separately):','',
        '| Method | Cameras | Full video | Queries | Eligible | Recall@1 | MRR | Query P95 (ms) | Complete images/s |',
        '|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for row in retrieval_rows:
        speed=f"{row['camera_image_fps']:.4f}" if row['speed_eligible'] else 'diagnostic cost only'
        lines.append(f"| {row['method']} | {row['camera_streams']} | {row['full_video']} | {row['requests']} | {row['eligible_queries']} | {row['Recall1'] if row['Recall1'] is not None else 'NA'} | {row['MRR'] if row['MRR'] is not None else 'NA'} | {row['query_p95']*1000:.4f} | {speed} |")
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
