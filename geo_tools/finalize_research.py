"""Audit the declared research scope before publishing a final scoped report."""
import ast
import csv
import math
import re
from common import *
from run_research_queue import process_identity


def require_complete_factors(value):
    expected={('B5','training_head_seed'),('B7','training_head_seed'),('B7','inference_rng_seed')}
    if {(row['method'],row['factor']) for row in value['groups']}!=expected:raise ValueError('missing seed factor')
    for row in value['groups']:
        if not row['complete'] or row['completed_trials']!=3 or row['pending'] or len(row['rows'])!=3:
            raise ValueError('incomplete declared seed factor')
        if not all(trial['exact_detector_membership'] for trial in row['rows']):raise ValueError('unpaired detector cohort')


def require_closed_resources(reports):
    if len(reports)!=2 or any(row['status']!='FINISHED_PROBE_SAMPLED_RESOURCES' or row['sample_groups']<=0 for row in reports):
        raise ValueError('both finite probes must be finalized')


def audit():
    artifacts={};cache={};checks=[]
    def digest(path):
        path=Path(path);key=(str(path),path.stat().st_mtime_ns,path.stat().st_size)
        if key not in cache:cache[key]=sha256(path)
        return cache[key]
    def retain(path,expected=None):
        path=Path(path);value=digest(path)
        if expected is not None and value!=expected:raise ValueError('changed evidence: '+str(path))
        artifacts[str(path)]=value
    def load(name,states=None):
        path=ROOT/'results'/name;value=json.loads(path.read_text());retain(path)
        if states and value.get('status') not in states:raise ValueError('unexpected evidence status: '+name)
        for file,digest_value in value.get('source_artifacts',{}).items():
            if not isinstance(digest_value,str) or len(digest_value)!=64:raise ValueError('unmeasured source hash: '+file)
            retain(file,digest_value)
        return value
    def check(identifier,label,evidence,notes):
        checks.append(dict(id=identifier,requirement=label,verified=True,evidence=evidence,scope_and_limits=notes))
    attachment=Path('/home/liuyeqiang/.codex/attachments/b2cd0a98-e4ad-4cf1-83fc-23a25eed5e30/Pasted text.txt');retain(attachment)
    split=json.loads((ROOT/'manifests/geo_split_v1.json').read_text());retain(ROOT/'manifests/geo_split_v1.json')
    strict=sorted(split['strict_development_video_ids'])
    pre=load('preflight_readonly_locked_inputs_final_v4.json',{'LOCKED_INPUTS_VERIFIED_MODEL_LOAD_NOT_RUN'})
    assert pre['locked_files_unchanged'] and pre['strict_video_ids']==strict and pre['raw_gt_files_verified']==10
    for path,value in pre['locked_file_sha256'].items():retain(path,value)
    lock=load('baseline_manifest.json',{'LOCKED_FULL_STRICT_DEVELOPMENT_BASELINE'})
    for key,path in [('checkpoint_sha256',CHECKPOINT),('config_sha256',ROOT/'configs/VISION_test.yaml'),('annotation_sha256',DATA/'annotations/train.json')]:retain(path,lock[key])
    assert lock['gmt_state_keys']==411 and not split['official_test_used_for_this_project'] and split['historical_test_exposure']
    assert not set(split['association_train_video_ids'])&set(split['full_development_video_ids'])
    check('inputs','Checkpoint/config/data, lawful split, TEST exposure',['baseline_manifest.json','preflight_readonly_locked_inputs_final_v4.json'],
          'Fixed411-key GMT and original TRAIN; strict5 plus seven conflicting-GT diagnostics. Backbone saw original TRAIN including DEV; historical TEST exposure disclosed, no new TEST selection.')
    imported=[]
    for directory in ('geo_tools','gtr/geometry','gtr/modeling'):
        for path in (ROOT/directory).rglob('*.py'):
            for node in ast.walk(ast.parse(path.read_text())):
                names=[alias.name for alias in node.names] if isinstance(node,ast.Import) else [node.module or ''] if isinstance(node,ast.ImportFrom) else []
                if any('jev' in name.lower() for name in names):imported.append(str(path))
    assert not imported
    unchanged=[]
    for file in ('gtr/modeling/roi_heads/gtr_roi_heads.py','gtr/modeling/roi_heads/association_head.py','gtr/modeling/roi_heads/transformer.py'):
        assert (ROOT/file).read_bytes()==subprocess.check_output(['git','show','8fa96705:'+file],cwd=ROOT);retain(ROOT/file);unchanged.append(file)
    check('isolation','Independent VGGT+GMT and preserved frozen GMT architecture',unchanged,
          'No direct JEV imports in new tools/geometry/modeling. GTRRCNN has explicit GEO hooks and empty-association/RNG repair; GTA/Hungarian/default memory are preserved, memory gets neutral GEO context.')
    for name in ('GEOMETRY_OPEN_SOURCE_AUDIT.md','GEO_STREAM_REPRODUCIBILITY.md','WWW_SCOPE_NOTE.md'):
        path=ROOT/'docs'/name;assert path.stat().st_size>100;retain(path)
    source=(ROOT/'docs/GEOMETRY_OPEN_SOURCE_AUDIT.md').read_text()
    assert all(word in source for word in ('VGGT-Det','StreamVGGT','FastVGGT','2607.22731','LICENSE'))
    check('source_audit','GMT/VGGT/Det/Stream/Fast/prior work, interfaces and licenses',['GEOMETRY_OPEN_SOURCE_AUDIT.md'],
          'Pinned source decisions, integration boundaries, noncommercial research weights and novelty risks retained; no commercial/Web-deployment claim.')
    named=['feature_screening.csv','geometry_validation.csv','ablation_results.csv','runtime_results.csv','experiment_manifest.json','final_summary.json']
    for name in named:assert (ROOT/'results'/name).stat().st_size>0
    for name in named:
        if name.endswith('.csv'):retain(ROOT/'results'/name)
    for name in ('finalize_research.py','summarize_results.py','update_reviewer_questions.py','plot_research_results.py','summarize_resource_samples.py','test_finalization_guards.py'):
        retain(ROOT/'geo_tools'/name)
    diagnostic=RUNTIME/'baseline/B0_raw_diagnostic_remaining7_full_v1'
    diagnostic_ids=sorted(split['raw_gt_diagnostic_development_video_ids'])
    assert set(strict)|set(diagnostic_ids)==set(split['full_development_video_ids']) and not set(strict)&set(diagnostic_ids)
    for scene in diagnostic_ids:
        metric_path=diagnostic/f'video_{scene:02d}.raw_diagnostic.metrics.json'
        value=json.loads(metric_path.read_text());retain(metric_path)
        assert value['status']=='PASS' and value['full_video'] and value['allow_duplicate_gt'] and not value['causal_online']
        assert value['video_id']==scene and value['gt_policy']=='all raw rows retained' and sum(value['duplicate_gt_rows'].values())>0
        retain(diagnostic/f'video_{scene:02d}.json',value['prediction_sha256'])
        run_path=diagnostic/f'video_{scene:02d}_manifest.json';run=json.loads(run_path.read_text());retain(run_path)
        assert run['full_video'] and run['video_id']==scene and run['checkpoint_sha256']==lock['checkpoint_sha256'] and run['annotation_sha256']==lock['annotation_sha256']
        for name,digest_value in value['raw_gt_sha256'].items():retain(DATA/'train'/name/'gt/gt.txt',digest_value)
    check('diagnostic_coverage','All original12 DEV scenes accounted for without GT repair',['B0_raw_diagnostic_remaining7_full_v1','geo_split_v1.json'],
          'Five strict scenes and seven full permissive raw-GT diagnostic scenes; every conflicting-GT row is retained. Diagnostics are offline and cannot be pooled as strict or causal evidence.')
    with (ROOT/'results/feature_screening.csv').open() as f:features=list(csv.DictReader(f))
    assert len(features)>=12 and all(key in features[0] for key in ('recall1','recall5','mean_rank','similarity_margin','hard_recall1','pair_auc','pair_ap'))
    text=json.dumps(features)
    assert all('layer_'+str(layer) in text for layer in (4,11,17,23)) and all(pool in text for pool in ('roi_align','mean','center'))
    retain(ROOT/'results/feature_screening.csv');load('feature_screening_manifest.json');load('dinov2_screening_manifest.json')
    check('features','Frozen multilevel/DINO/GMT screening, ROI and small-target diagnostics',['feature_screening.csv','feature_screening_manifest.json'],
          'Same-time sample/hard-query metrics remain separate from complete tracking/clip retrieval;139 CPU tests cover token/pad/ROI/normalization contracts.')
    heads=load('low_rank_three_backbones_train_v1_manifest.json')
    assert heads['steps']==2000 and heads['seeds']==[20261014,20261015,20261016] and len(heads['results'])==9
    assert not set(heads['fit_video_ids'])&set(heads['heldout_video_ids'])
    assert set(heads['fit_video_ids']+heads['heldout_video_ids'])<=set(split['association_train_video_ids'])
    for row in heads['results']:retain(row['checkpoint'],row['checkpoint_sha256'])
    check('training','Nine matched light heads, three TRAIN seeds, frozen upstream',['low_rank_three_backbones_train_v1_manifest.json'],
          'Fit scenes5/8, held-out TRAIN12,2000 updates; no GMT Stage1/2 retraining. Historical training-only status is supplemented by complete native closed-loop audits.')
    relation=load('identity_relations_matrix_v2.json',{'COMPLETE_DECLARED_MATRIX'});assert len(relation['rows'])==35 and not relation['pending']
    settings=load('suite_settings_audit_v1.json');assert len(settings['rows'])==42 and not settings['inconsistent_suites'] and not settings['unverified_default_suites']
    suites={}
    for path in sorted((ROOT/'results').glob('*_metrics.json')):
        value=json.loads(path.read_text())
        if value.get('video_ids')!=strict or not value.get('full_video'):continue
        assert value['status']=='PASS' and all(key in value['metrics'] and math.isfinite(value['metrics'][key]) for key in ('HOTA','DetA','AssA','IDF1','CVIDF1','CVMA','MOTA','IDSW'))
        retain(path)
        for captured in value['run_manifests'].values():retain(captured['path'],captured['sha256'])
        for file,digest_value in value['prediction_sha256'].items():retain(file,digest_value)
        for file,digest_value in value['raw_gt_sha256'].items():retain(file,digest_value)
        suites[path.stem.removesuffix('_metrics')]=value
    assert len(suites)==42
    for row in relation['rows']:
        assert row['full_strict_suite']
        retain(ROOT/row['source'],row['source_sha256'])
        if suites[row['suite']]['causal_online']:
            assert not row['added_detections'] and not row['removed_detections']
    check('matrix','B0–B9, matched capacity/fusion, complete paired output relations',['identity_relations_matrix_v2.json','suite_settings_audit_v1.json'],
          '42 native strict5 suites;35 declared paired comparisons. Causal pairs have exact detector membership; offline full-video short-track filtering changes some exported rows, explicitly counted rather than claimed identical. Offline/causal protocols remain separate. Seven conflicting-GT DEV scenes have separate permissive diagnostics, not repaired strict labels.')
    seeds=load('causal_dense_seed_stability_v1.json',{'COMPLETE_DECLARED_SEED_FACTORS'});require_complete_factors(seeds)
    prior=load('native_prior_inference_seed_stability_v1.json',{'THREE_PAIRED_TRIALS_AUDITED'});assert len(prior['completed_inference_seeds'])==3 and not prior['pending_inference_seeds']
    check('stability','Three TRAIN heads and fixed-head inference-RNG trials',['causal_dense_seed_stability_v1.json','native_prior_inference_seed_stability_v1.json'],
          'All declared trials complete; shared primary B7 is not independent repetition. VGGT deltas have mixed signs in both factors; same five scenes, no inferential CI or stable unique-gain claim.')
    gates=[('OFF_c214_full_strict_prediction_parity_v1.json',{'PASS'}),('ZERO_c214_full_strict_prediction_parity_v1.json',{'PASS'}),
           ('OFF_post_cache_trace_strict_v1_parity.json',{'PASS'}),('ZERO_post_cache_trace_strict_v1_parity.json',{'PASS'}),
           ('current_default_source_consistency_v1.json',{'RECORDED_DEFAULT_SOURCE_MATCHES'}),
           ('compute_neutral_parity.json',{'PASS'}),('current_group_reading_parity.json',{'PASS'})]
    for name,status in gates:load(name,status)
    check('neutral','OFF/zero/computed-neutral exact outputs and current default source',[name for name,_ in gates],
          'Original protocols retained; ten retained full OFF/zero scenes and44 recorded default files per scene match current bytes. Not exhaustive dynamic imports/library binaries; optional retrieval is separately gated.')
    for name,status in [('complete_causal_vggt_view_order_audit_v2.json',{'EXACT_TRACKING_PARITY'}),('physical_camera_input_contract_v1.json',{'PASS_EXPLICIT_CAMERA_INPUT_CONTRACT'}),('live_whole_gmt_camera_order_v1.json',{'DIAGNOSTIC_COMPLETED'}),('live_camera_index_hook_probe_v1.json',{'PASS'}),('camera_index_corrected_full_strict_audit_v2.json',{'PREDICTION_DIFFERENCES_OBSERVED'})]:load(name,status)
    check('camera','Camera order, explicit physical IDs and original camera-MLP semantics',['physical_camera_input_contract_v1.json','complete_causal_vggt_view_order_audit_v2.json','camera_index_corrected_full_strict_audit_v2.json'],
          'Whole-GMT arbitrary internal order fails even OFF; canonical physical-ID adapter passes four real40-group variants. Encoder k20 reverse passes full5 with0 pose gates; no valid-pose/internal invariance claim. Corrected MLP changes5 scene18 rows despite equal metrics.')
    for name in ('causal_prefix_parity.json','causal_scheduler_gpu_prefix_v1.json','async_camera_gpu_prefix_v1.json','retrieval_multistream_gpu_prefix_v1.json','empty_association_gpu_v2.json'):
        value=load(name,{'PASS'})
        if name.startswith('empty'):assert len(value['cases'])==4 and all(row['new_path_success'] and row['legacy_values_parity'] and row['legacy_rng_parity'] for row in value['cases'])
    stress=load('constructed_geometry_degeneracy_v1.json',{'PASS_CONSTRUCTED_CPU_GEOMETRY_STRESS'});assert len(stress['observations'])==5 and stress['encoder_calls']==0
    check('correctness','Prefix causality, empty/texture/tiny/rotation/nonfinite/OOM/timeout/stale fallbacks',['causal_prefix_parity.json','empty_association_gpu_v2.json','constructed_geometry_degeneracy_v1.json','final_cpu_regression_v1.log'],
          'Real GPU/prefix tests plus constructed/injected branches. Old1024 empty fixture is invalid;1152 v2 is used. Rotation flow failure disables old geometry but motion-only misses an update; not natural-UAV/GT-pose efficacy.')
    geometry=[]
    for seed in (20261014,20261015,20261016):
        name='causal_dense_geometry_full_strict_attribution_v1.json' if seed==20261014 else f'causal_dense_geometry_seed{seed}_full_strict_attribution_v1.json'
        value=load(name,{'EXACT_FEATURE_JOINT_FULL_STRICT_OUTPUTS'});assert value['full_strict_suite'] and len(value['rows'])==5 and all(row['exact_serialized_predictions'] for row in value['rows']);geometry.append(name)
    failure=load('geometry_failure_taxonomy_v1.json',{'RECORDED_GATE_DIAGNOSTICS_CONSOLIDATED'});assert len(failure['rows'])==420
    check('dcga','Pose/background/plane/neural/temporal geometry and independent effect',geometry+['geometry_failure_taxonomy_v1.json'],
          'All three heads have exact B5/B7 full5 outputs;3 numerical/background gates per replay have no final output effect.420 screening variants accept0 and are not independent samples. Physical failure causes/GT pose accuracy and valid real propagation remain unproved; legacy dirty-source limits retained.')
    for name in ('causal_B8_B9_complete_budget_runtime_audit_v1.json','causal_complete_interval_identity_budget_v1.json','causal_trigger_full_dev_realized_budget_v1.json','causal_fixed_full18_offset_stability_v1.json','causal_random_full18_seed_stability_v1.json','causal_joint_full18_repeat_stability_v1.json'):
        value=load(name,{'PASS'})
        for method in value.get('methods',[]):
            retain(ROOT/method['source'],method['source_sha256'])
            assert sorted(row['video_id'] for row in method['rows'])==strict
            for row in method['rows']:
                retain(row['manifest'],row['manifest_sha256']);actual_run=json.loads(Path(row['manifest']).read_text())
                assert actual_run['full_video'] and actual_run['total_vggt_calls']==row['actual_vggt_calls'] and not actual_run['geo_failures']
            assert sum(row['actual_vggt_calls'] for row in method['rows'])==method['total_actual_vggt_calls']
    check('isgu','Dense/k5/k10/k20, equal-budget random/joint and single triggers/repeats',['causal_complete_interval_identity_budget_v1.json','causal_B8_B9_complete_budget_runtime_audit_v1.json','causal_trigger_full_dev_realized_budget_v1.json'],
          'Fixed/joint/random400 calls; identity218 and motion0 are not equal realized budgets. B9 equals causal B0; repeats/offsets provide no stable joint gain. Implemented scheduling/propagation remains optional, not proven innovation.')
    recovery=load('history_rank_train_audit_v1.json');assert recovery['unique_vggt_primary_gate'] is False
    fast=load('fast_vggt_short_pair_fp16_v4_comparison.json',{'PASS_MICROBENCHMARK_ONLY'});assert len(fast['rows'])==3 and all(row['original_over_fast_ratio']<1 for row in fast['rows'])
    check('recovery','Bounded alternative heads/cache/geometry and Fast/Stream references',['history_rank_train_audit_v1.json','fast_vggt_short_pair_fp16_v4_comparison.json'],
          'History TRAIN recovery has no native error corrections; cache controls lose dense gain. Actual short-pair V100 Fast is slower after isolated FP16 repair; Stream is a different trained KV architecture. Original failures retained; no backend/A100/long-sequence multiplier inferred.')
    retrieval=load('retrieval_paired_query_outcomes_v1.json',{'COMPLETE_DECLARED_COMPARISONS'});assert len(retrieval['rows'])==12 and not retrieval['pending']
    for name in ('retrieval_10streams_full_tracking_output_parity_v1.json','native_prior_retrieval_full4_10_tracking_parity_v1.json','vggt_dense_retrieval_full2_tracking_parity_v1.json','vggt_dense_retrieval_full4_tracking_parity_v1.json','gmt_dense_head_retrieval_full2_tracking_parity_v1.json','gmt_dense_head_retrieval_full4_tracking_parity_v1.json','dinov2_dense_head_retrieval_full2_tracking_parity_v1.json','dinov2_dense_head_retrieval_full4_tracking_parity_v1.json'):load(name,{'PASS'})
    check('retrieval','Real multi-stream other-camera clip retrieval and WWW scope',['retrieval_paired_query_outcomes_v1.json','WWW_SCOPE_NOTE.md'],
          'Real image reads/detection/GTA/feature/index/query,2/4/10 streams, fixed cohorts and GT post-run only. Conditional quality/coverage/all-request success are separate. Local replay/query timing is not capture/network latency or Web deployment; WWW scientific fit and PAPER_READY unsupported.')
    actual=load('actual_decision_full_scene1_cohort_verified_v2.json',{'PASS_ACTUAL_SOLVER_DIAGNOSTIC'})
    assert actual['full_video'] and not actual['full_strict_suite'] and actual['identical_physical_detector_proposal_cohort'] and actual['identical_common_primary_query_gt_labels']
    check('decisions','Actual primary solver counts with GT/query cohort control',['actual_decision_full_scene1_cohort_verified_v2.json'],
          'Complete scene1 only,32 known fixes/1 harm/386 uncertain, including75 correct→ambiguous. Own histories and pre-memory scope; no global-final-ID or all-strict Hungarian correctness claim.')
    sampled=[load(name,{'FINISHED_PROBE_SAMPLED_RESOURCES'}) for name in ('remaining_job_sampled_resources_v1.json','balanced_handoff_sampled_resources_v1.json')];require_closed_resources(sampled)
    for value in sampled:
        assert {row['gpu'] for row in value['device_rows']}=={0,4,5,6}
        for path in value['source_artifacts']:
            if Path(path).name=='manifest.json':assert json.loads(Path(path).read_text())['status']=='FINISHED_EXISTING_JOBS'
        from summarize_resource_samples import summarize
        folder=next(Path(path).parent for path in value['source_artifacts'] if Path(path).name=='manifest.json')
        with (folder/'devices.csv').open() as f:devices=list(csv.DictReader(f))
        with (folder/'processes.csv').open() as f:processes=list(csv.DictReader(f))
        rebuilt=summarize(json.loads((folder/'manifest.json').read_text()),devices,processes)
        assert all(value[key]==rebuilt[key] for key in rebuilt)
    ledger=load('resource_ledger_v1.json');assert ledger['parameters']['gmt']['strict_checkpoint_keys']==411 and ledger['parameters']['gmt']['total']==53515309
    for file,digest_value in ledger['parameter_sources_sha256'].items():retain(file,digest_value)
    cost=load('process_wall_runtime_v1.json',{'RECORDED_UNHELD_PROCESS_WALL_COSTS'});assert len(cost['suites'])>=17 and len(cost['excluded_held_stages'])==8
    storage=load('storage_budget_v1.json',{'MEASURED_CURRENT_STORAGE_SNAPSHOT'});assert storage['total_apparent_bytes']==sum(row['apparent_bytes'] for row in storage['rows'])
    check('systems','Complete system cost, latency, model sizes, GPU/host/cache/disk scopes',['process_wall_runtime_v1.json','resource_ledger_v1.json','remaining_job_sampled_resources_v1.json','balanced_handoff_sampled_resources_v1.json','storage_budget_v1.json'],
          'Reads/transfer/model/publication included in replay timing; imports/static setup separately measured in unheld process costs.8 held/superseded outer costs unavailable, not zero. Sample maxima/means, allocator/RSS/index/final-history/disk have distinct scopes. No peak summation, stable speed, real-time capture or total-memory-cap claim.')
    for tag in ('primary_vggt_inference_balanced_handoff_v1_gpu0','primary_vggt_inference_balanced_handoff_v1_gpu4','actual_decision_scene1_v1_gpu6','matched_head_tracking_retrieval_resume_v2_gpu5'):
        path=RUNTIME/'queues'/tag/'manifest.json';value=json.loads(path.read_text());assert value['status']=='FINISHED' and all(row['status']=='PASS' for row in value['stages']);retain(path)
    handoff=load('primary_vggt_inference_balanced_handoff_v1.json',{'ORIGINAL_DISPATCHER_SUPERSEDED_AFTER_CHILD_EXIT'})
    for pid,ticks in [(handoff['controller_pid'],handoff['controller_starttime']),*[(int(pid),ticks) for pid,ticks in handoff['existing_children'].items()]]:
        current=process_identity(pid);assert current is None or current[0]!=ticks
    check('lifecycle','Finite queues complete, legal handoff and retained failures',['primary_vggt_inference_balanced_handoff_v1.json'],
          'Original child completes untouched; dispatcher snapshot explicitly superseded, not relabeled PASS. Current replacement/solver/retrieval recovery queues end PASS. Earlier invalid fixtures/interface/checker failures remain archived with corrected evidence.')
    log=RUNTIME/'logs/final_cpu_regression_v1.log';text=log.read_text();assert re.search(r'Ran 139 tests in .*\n\nOK',text);retain(log)
    guard_log=RUNTIME/'logs/finalization_guards_v1.log';assert re.search(r'Ran 2 tests in .*\n\nOK',guard_log.read_text());retain(guard_log)
    check('tests','Full CPU regression plus actual GPU/whole-video gates',['final_cpu_regression_v1.log'],
          '139 full core tests and2 finalization guards pass; source proofs, actual prefix/full outputs, native metrics and resource hashes are independently verified. Synthetic failures are distinct from natural hardware failures.')
    plot=load('plots/manifest.json')
    for name,value in plot['outputs'].items():retain(ROOT/'results/plots'/name,value)
    check('deliverables','Named automatic reports/tables/manifests and figures',named,
          'Metric tables read native artifacts; exact commands, config/head/model/data hashes, source versions, environment, cache/causal flags and raw logs remain linked. Final report/scientific/engineering/paper states are separate.')
    primary_hashes={row['checkpoint_sha256'] for row in heads['results'] if row['seed']==20261014}
    eligible=[]
    for row in relation['rows']:
        suite=suites[row['suite']]
        if not suite['causal_online']:continue
        runs=[json.loads(Path(item['path']).read_text()) for item in suite['run_manifests'].values()]
        if any(run['seed']!=20261014 or run.get('head_checkpoint_sha256') not in primary_hashes|{None} for run in runs):continue
        eligible.append(dict(suite=row['suite'],CVIDF1=suite['metrics']['CVIDF1'],IDSW=suite['metrics']['IDSW']))
    chosen=max(eligible,key=lambda row:(row['CVIDF1'],-row['IDSW']))
    selection=dict(status='DEV_ACCURACY_REFERENCE_SELECTED_WITHOUT_TEST',selected=chosen,eligible_primary_candidates=eligible,
        rule='highest complete causal strict5 CVIDF1 among matched actual-inference20261014 and preselected TRAIN-head20261014 candidates; IDSW breaks ties; no reselection of stability-head seeds',
        frozen_gmt_checkpoint=str(CHECKPOINT),checkpoint_sha256=lock['checkpoint_sha256'],retained_primary_vggt_head=str(RUNTIME/'heads/low_rank_three_backbones_train_v1/vggt_seed20261014.pt'),
        scope='DEV accuracy reference, not novel VGGT contribution, unseen-backbone generalization or deployment default change; no new training/threshold tuning follows; all head checkpoints retained',provenance=provenance())
    save_json(ROOT/'results/final_candidate_selection_v1.json',selection);retain(ROOT/'results/final_candidate_selection_v1.json')
    check('selection','Retained candidate/checkpoints and distinct scientific/engineering/paper states',['final_candidate_selection_v1.json'],
          'Ordinary native-prior control is the strongest matched DEV accuracy reference; primary VGGT head remains fixed. Paired positive signals exist but stable VGGT/three-module/WWW-paper success is not established.')
    value=dict(status='VERIFIED_SCOPED_RESEARCH_DELIVERABLES',requirements=checks,source_artifacts=artifacts,strict_video_ids=strict,
        native_full_strict_suites=len(suites),paired_comparisons=len(relation['rows']),cpu_tests=141,
        engineering_complete=True,stable_vggt_specific_scientific_success=False,paper_ready=False,
        scope='completed Phase0–5/B0–B9 and bounded alternatives under the recorded data/hardware constraints; negative/unsupported scientific contributions remain negative, not repaired by claim changes',
        limitations='strict original12-scene evaluation is infeasible with unchanged conflicting raw GT; strict5 and seven diagnostics remain separate; V100 only; no universal physical-geometry, natural-rotation, arbitrary-order or Web-deployment validity claim; legacy provenance/coverage limits remain explicit',provenance=provenance())
    save_json(ROOT/'results/final_completion_audit_v1.json',value)
    lines=['# Final attachment requirement audit','',value['scope']+'.','',value['limitations']+'.','',
           '| Requirement | Verified evidence | Disposition and limits |','|---|---|---|']
    for row in checks:lines.append(f"| {row['requirement']} | {', '.join(row['evidence'])} | {row['scope_and_limits']} |")
    (ROOT/'docs/GEO_STREAM_REQUIREMENT_AUDIT.md').write_text('\n'.join(lines)+'\n')
    return value,selection


def render(summary,audit_value,selection):
    for path,value in audit_value['source_artifacts'].items():
        if sha256(path)!=value:raise ValueError('completion evidence changed: '+path)
    summary.update(status='COMPLETED_SCOPED_RESEARCH_WITH_NEGATIVE_FINDINGS',engineering_complete=True,paper_ready=False,
        scientific_success='PRELIMINARY_PAIRED_SIGNALS_WITHOUT_STABLE_VGGT_SPECIFIC_SUCCESS',
        completion_audit=dict(path=str(ROOT/'results/final_completion_audit_v1.json'),sha256=sha256(ROOT/'results/final_completion_audit_v1.json')),
        final_accuracy_reference=selection['selected'])
    summary['limitations']=['strict5 plus seven conflicting-GT diagnostics; backbone TRAIN/DEV exposure and historical TEST exposure disclosed',
        'mixed signs across complete TRAIN-head and frozen-head inference-RNG factors; no stable VGGT-specific benefit or independent geometry contribution',
        'V100 local replay, scope-limited timing/memory/correctness; no real-time capture/Web deployment or PAPER_READY claim']
    save_json(ROOT/'results/final_summary.json',summary)
    manifest=json.loads((ROOT/'results/experiment_manifest.json').read_text());manifest['status']='COMPLETED_SCOPED_RESEARCH_INDEX';save_json(ROOT/'results/experiment_manifest.json',manifest)
    lines=['# VGGT+GMT — final scoped research report','',
        'The declared experiments and bounded recovery paths are complete under the recorded data/hardware constraints. Stable VGGT-specific improvement, independent geometry benefit and a WWW-ready three-module contribution were not established.','',
        'SCIENTIFIC_SUCCESS: paired preliminary signals exist; stable VGGT-specific success is not established. ENGINEERING_COMPLETE: true within the audited scope. PAPER_READY: false.','',
        'No JEV decision model, learned weights or perception cache is used. New training/selection uses original TRAIN only. Frozen GMT saw these DEV scenes in original training; historical TEST exposure is disclosed. This is not backbone-unseen or blind-TEST generalization. Strict DEV is [1,10,11,18,19]; seven additional original DEV scenes have conflicting raw GT and separate permissive diagnostics. No labels or metric definitions were repaired to obtain gains.','',
        '## Tracking results','',
        'Full strict5 pooled native metrics, actual inference20261014. Offline GMT full-video filtering and causal publication have separate baselines. All metrics including DetA/IDF1/CVMA/MOTA remain in ablation_results.csv.','',
        '| Primary causal candidate | HOTA | AssA | CVIDF1 | Delta (pp) | IDSW |','|---|---:|---:|---:|---:|---:|']
    candidates=selection['eligible_primary_candidates'];base=json.loads((ROOT/'results/causal_B0_strict_suite_v1_metrics.json').read_text())['metrics']
    for item in candidates:
        m=json.loads((ROOT/'results'/f"{item['suite']}_metrics.json").read_text())['metrics']
        lines.append(f"| {item['suite']} | {m['HOTA']:.6f} | {m['AssA']:.6f} | {m['CVIDF1']:.6f} | {m['CVIDF1']-base['CVIDF1']:+.6f} | {m['IDSW']} |")
    lines+=['',f"The DEV accuracy reference is {selection['selected']['suite']} (CVIDF1 {selection['selected']['CVIDF1']:.6f}, IDSW {selection['selected']['IDSW']}). It is an ordinary head-free native camera/time prior, with no VGGT calls or new learned weights. It outperforms the primary learned VGGT candidate; it is retained as a control/reference, not claimed as the three proposed innovations. The preselected VGGT TRAIN head20261014 and all other heads remain retained; no seed reselection, new fitting or TEST tuning follows.",'',
            '## Complete seed factors','',
            '| Factor | Method | TRAIN head | Actual inference RNG | Paired CVIDF1 delta (pp) | IDSW |','|---|---|---:|---:|---:|---:|']
    for group in summary['dense_causal_seed_factor_stability']['groups']:
        for row in group['rows']:lines.append(f"| {group['factor']} | {group['method']} | {row['training_head_seed']} | {row['inference_seed']} | {row['paired_metric_delta']['CVIDF1']:+.6f} | {row['candidate_metrics']['IDSW']} |")
    lines+=['','Every trial is paired to its own same-inference-seed causal baseline, with exact detector membership and fixed effective options/head hashes. Primary B7 is shared across factors. These repeat the same five scenes, not independent datasets; no confidence interval or stable-effect claim. Both factors include a negative CVIDF1 result despite fewer ID switches. The native transformer randomly assigns trajectory embedding indices in evaluation, motivating separate inference-RNG controls.','',
            '![TRAIN-head trials](../results/plots/causal_B5_B7_training_head_deltas.png)','',
            '![Fixed-head inference trials](../results/plots/causal_B7_fixed_head_inference_deltas.png)','',
            '## Findings for the three ideas','',
            '- GCIR: two TRAIN heads improve full-DEV CVIDF1 and one regresses; frozen-head inference trials also change sign. Matched DINO/GMT heads and ordinary native priors prevent unique VGGT attribution. Layer/pool/native/compressed/hard-candidate diagnostics and bounded history TRAIN recovery are retained; this is not a proof that all foundation features are useless.',
            '- DCGA: complete B5/B7 serialized outputs are identical for all three TRAIN heads over strict5. Each joint replay has3 numerical/background gate passes, with no independent final-output effect.420 fixed screening variants accept0 geometry states; these repeat samples and do not establish physical causes. Pose/background/plane/neural and legal temporal branches were exercised; GT-pose accuracy and valid real propagation remain unproved.',
            '- ISGU: fixed/random/joint400-call budgets are measured. B9 equals causal GMT, while B8 gains only+.001627 CVIDF1pp. Identity-only uses218 calls and motion-only0, so they cannot be claimed superior at the same realized400-call budget. Fixed offsets, random seeds and joint repeats on complete scene18 retain baseline outputs. The motion estimator can return0 on failed rotation flow and miss an update.','',
            '## Actual decisions and failure analysis','',
            'Complete scene1 diagnostic preserves standalone predictions and verifies all4,527 physical detector proposals/common GT labels. There are4,524 common primary queries,32 conditional known corrections,1 known harm and386 uncertain cases. All32 recover missed known positive links;75 correct-link queries become ambiguous-reference. Averaged-descriptor memory decisions are excluded, and each run has its own history. These are pre-memory primary decisions, distinct from final ID correctness and quadratic all-time output-pair counts. [Complete state/transition table](GEO_STREAM_ACTUAL_DECISIONS.md).','',
            'Correctness covers full OFF/zero raw parity, explicit camera-input canonicalization, real GPU empty cases, actual prefix causality and constructed/injected texture/nonoverlap/tiny/nonfinite/OOM/timeout/stale branches. Whole-GMT internal order sensitivity remains a measured baseline limitation; canonical physical-ID input passes four40-group variants. Blank/unrelated background and60-degree rotation stress is constructed CPU evidence, not natural UAV pose/overlap truth. The original camera-MLP scalar collapse is documented; its corrected full5 control has equal metrics but5 changed scene18 rows. [Requirement audit](GEO_STREAM_REQUIREMENT_AUDIT.md).','',
            '## Runtime, resources and storage','',
            'Replay body timing charges image reads, preprocessing/transfers, detection/GTA, feature/geometry work and identity publication; retrieval additionally charges index/query/log I/O. Static setup/import/export costs are separately available from unheld parent-observed process durations. They are not camera capture or HTTP/network latency. For two cameras, per-camera delivered FPS equals synchronized-group FPS, half aggregate camera-image FPS.','',
            '| Strict5 setting | Actual VGGT calls | Body camera images/s | Group FPS |','|---|---:|---:|---:|']
    interval=json.loads((ROOT/'results/causal_complete_interval_identity_budget_v1.json').read_text())
    for row in interval['methods']:lines.append(f"| {Path(row['source']).stem.removesuffix('_sources')} | {row['total_actual_vggt_calls']} | {row['cumulative_camera_image_fps']:.6f} | {row['cumulative_group_fps']:.6f} |")
    cost=json.loads((ROOT/'results/process_wall_runtime_v1.json').read_text());primary=next(row for row in cost['suites'] if row['suite']=='causal_B7_fixed_k1_strict_suite_v1')
    ledger=json.loads((ROOT/'results/resource_ledger_v1.json').read_text())
    lines+=['',f"Primary dense joint process throughput is {primary['process_camera_image_fps']:.6f} images/s versus body {primary['reported_loop_camera_image_fps']:.6f}. The audit has {len(cost['rows'])} complete unheld processes/{len(cost['suites'])} strict5 process suites, with {len(cost['excluded_held_stages'])} held/overlap/superseded-child outer costs explicitly unavailable, not zero. These descriptive replays do not establish stable speed or real-time operation. [Process cost](GEO_STREAM_PROCESS_WALL_RUNTIME.md).",'',
            f"Frozen GMT has {ledger['parameters']['gmt']['total']:,} parameters, strict411 checkpoint keys. Official full VGGT has1,256,537,516 parameters; the pruned resident constructor has1,125,286,930, a structural count distinct from measured GPU memory. Matched head counts309,319/309,849/310,402. Per-process CUDA allocator/RSS, whole-device sampled NVML, final history and bounded index/vector payloads are separate; neither noncoincident peaks nor RSS/probe overlaps are summed. Original GMT history is unbounded, and index caps are not total-system memory caps. [Sampled resources](GEO_STREAM_SAMPLED_RESOURCES.md), [storage](GEO_STREAM_STORAGE_BUDGET.md).",'',
            'Hardware is the observed V100 system, at most project GPUs0/4/5/6, not A100. The available FastVGGT short-pair FP16 reference is about2.12–2.15times slower on this setup; original BF16 failures/bridges remain separate. StreamVGGT is a differently trained KV architecture, not a lossless frozen-weight switch. No production integration, long-sequence or other-hardware multiplier is inferred.','',
            '## Multi-stream retrieval and WWW suitability','',
            'Real local image/detector/GTA/feature/index/query producers cover2/4/10 streams. All12 declared physical-query comparisons complete; positive gallery labels are strictly post-run, scene-qualified, majority over all clip observations including unmatched ones. Conditional Recall/MRR, detector coverage, gallery eligibility and all-request success remain separate.','',
            'Dense VGGT tracking plus native ranking corrects one scene18 query among309 eligible/598 requests; Recall1 rises .977346→.980583. DINO tracking has the same aggregate Recall1 and higher MRR, while correcting a different query, so equal scores do not establish the same correction or unique foundation evidence. Four-camera scene1 DINO has1 correction/1 harm despite unchanged aggregate Recall1; uncertainty and rank harms remain retained. Raw VGGT-feature ten-camera ranking is worse than native ranking on the fixed scene18 cohort (.705502 versus .841424). [Query audit](GEO_STREAM_RETRIEVAL_QUERY_AUDIT.md).','',
            'Local replay and local query latency do not establish Internet-source provenance, live Web deployment, real-user traffic or an independent WWW scientific contribution. PAPER_READY remains false. [WWW scope](WWW_SCOPE_NOTE.md), [all15 reviewer answers](REVIEWER_QUESTIONS.md).','',
            '## Reproduction and retained limitations','',
            '[Execution/source instructions](GEO_STREAM_REPRODUCIBILITY.md), [source/license audit](GEOMETRY_OPEN_SOURCE_AUDIT.md), [machine completion audit](../results/final_completion_audit_v1.json), [candidate record](../results/final_candidate_selection_v1.json). Named CSVs, native metrics and experiment manifests are generated from retained artifacts; hashes and source commits bind inputs. Original failed/invalid fixtures, old checker failures, weaker legacy dirty-source provenance and unavailable loaded-file/library/coverage facts are retained rather than relabeled. No raw dataset, large checkpoint, dense token cache or log bulk is added to Git.','']
    (ROOT/'docs/GEO_STREAM_FINAL_REPORT.md').write_text('\n'.join(lines))


def main():
    value,selection=audit();summary=json.loads((ROOT/'results/final_summary.json').read_text());render(summary,value,selection)
    print(json.dumps(dict(status=value['status'],requirements=len(value['requirements']),native_suites=value['native_full_strict_suites'],paired_comparisons=value['paired_comparisons'],selected=selection['selected'])))


if __name__=='__main__':main()
