#!/usr/bin/env python3
"""Publish actual full-train/test artifacts; missing runs never become zero scores."""
from __future__ import annotations
import argparse
import collections
import csv
import hashlib
import json
import statistics
import time
from pathlib import Path
import numpy as np
from cache_fulltrain import atomic_json, sha

REPO=Path(__file__).resolve().parents[1]
METRICS=['HOTA','DetA','AssA','IDF1','IDSW','CVIDF1','CVMA','MOTA','Frag','CLR_TP','CLR_FN','CLR_FP']


def read(path,default=None):
    path=Path(path)
    for attempt in range(3):
        try:return json.loads(path.read_text())
        except (FileNotFoundError,json.JSONDecodeError):
            if attempt<2:time.sleep(.1)
    return default


def csv_file(path,rows,fields):
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)


def summary(values):
    a=np.asarray(values,np.float64)
    return {'count':len(a),'total_s':float(a.sum()),'p50_s':float(np.quantile(a,.5)) if len(a) else None,
        'p95_s':float(np.quantile(a,.95)) if len(a) else None}


def render_report(path,report,coverage,fps):
    rows=report['metrics'];lookup={r['method']:r for r in rows}
    fields=['HOTA','DetA','AssA','IDF1','IDSW','CVIDF1','CVMA','MOTA','real_e2e_fps']
    def number(value):return 'pending' if value is None else f'{value:.3f}'
    lines=['## Actual full TRAIN and independent TEST results','',
        'Status: '+report['status']+'. All values below come from saved complete-video predictions and official evaluators.',
        '',f"Final fitting: {len(coverage['final_training'])} trials on all 24 TRAIN scenes / 28,754 timestamps per epoch. Internal epoch/threshold selection used TRAIN only. Main seeds are 20261022/23/24; matched-seed ablation comparisons use 20261022.",
        '',f"Each completed method inferred all 22 TEST videos / 29,019 synchronized timestamps. Official metrics use the same {len(report['evaluation_scope'])} original-GT-valid scenes; eight raw-GT-invalid scenes have submission files but no fabricated metrics.",
        '', '| Method | '+' | '.join(fields)+' |','|---|'+'---:|'*len(fields)]
    for row in rows:lines.append('| '+row['method']+' | '+' | '.join(number(row.get(k)) for k in fields)+' |')
    lines+=['','FPS means synchronized dual-camera timestamps per second. Four-card aggregate throughput is never substituted for a video stream. Native GMT returns its postprocessed video at the end; its throughput does not establish per-timestamp final-ID latency. WorldMOT measures RGB read through flushed ID delivery, including gaps for telemetry between frames; initialization and offline scoring are excluded.','',
        'The `no_omega` and `no_geometry` quality controls still execute the same perception backends for a matched-input comparison; these measurements are not a claim of removing Omega computation.','']
    if 'joint_three_seed_summary' in report:
        lines+=['Joint PWQT mean ± sample standard deviation across three seeds: '+', '.join(
            f'{k} {v["mean"]:.3f} ± {v["sample_std"]:.3f}' for k,v in report['joint_three_seed_summary'].items() if k in ['HOTA','AssA','IDF1','CVIDF1','IDSW'])+'.','']
    main=lookup.get('v2_joint_seed20261022')
    if main:
        lines+=['Matched-seed differences (joint minus the named comparator; lower IDSW is better):','',
            '| Comparator | ΔHOTA | ΔAssA | ΔIDF1 | ΔCVIDF1 | ΔIDSW |','|---|---:|---:|---:|---:|---:|']
        for method in ['v1','v1_observed_bbox','v2_two_stage_seed20261022','v2_no_geometry_seed20261022','v2_no_omega_seed20261022','gmt_native']:
            if method in lookup:lines.append('| '+method+' | '+' | '.join(f'{main[k]-lookup[method][k]:+.3f}' if k in main and k in lookup[method] else 'pending' for k in ['HOTA','AssA','IDF1','CVIDF1','IDSW'])+' |')
        lines+=['','Joint versus `no_geometry` isolates geometry with all visual branches retained. Joint versus `no_omega` removes both local Omega appearance and geometry and cannot by itself establish an independent geometry contribution. Joint versus learned two-stage uses the same detector boxes and feature sources. GMT has its native detector; that comparison measures complete systems.','']
    lines+=['Checkpoint provenance:','', '| Trial | Seed | SHA-256 |','|---|---:|---|']
    for row in rows:lines.append(f'| {row["method"]} | {row.get("seed") or "frozen baseline"} | {row.get("checkpoint_sha256")} |')
    lines+=['','Official tool fingerprints and source reports: '+json.dumps(report.get('official_tool_fingerprints',{}),sort_keys=True),
        '', 'Stage timings and P50/P95 are in `results/worldmot_v2_test_fps.json`; full failure-event counts, stage evidence, and fixed-rule selected cases are in `results/worldmot_v2_failure_analysis.json`. Custom diagnostic matches are never substituted for official IDSW/HOTA/IDF1.','']
    marker='<!-- BEGIN ACTUAL FULLTRAIN RESULTS -->';end='<!-- END ACTUAL FULLTRAIN RESULTS -->'
    path=Path(path);text=path.read_text()
    if report['status']=='COMPLETE_OFFICIAL_INDEPENDENT_TEST':
        text=text.replace('Status: IN_PROGRESS. This report is updated from actual artifacts; neither a configuration nor a pilot is a completion claim.',
            'Status: COMPLETE_OFFICIAL_INDEPENDENT_TEST. Final results appear below; earlier stage diagnostics and invalidations are retained as history.')
    if marker in text:text=text[:text.index(marker)].rstrip()+'\n'
    path.write_text(text+'\n'+marker+'\n'+'\n'.join(lines)+end+'\n')


def collect(root,output,require_complete=False):
    root=Path(root);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    state=read(root/'experiment_status.json',{'stage':'NOT_STARTED'})
    verification=read(root/'cache_verification/coverage.json')
    progress=[r for p in sorted((root/'train_cache').glob('scene_*/progress.json')) if (r:=read(p)) is not None]
    training=[]
    for phase in ['internal','final']:
        for path in sorted((root/phase).glob('*.json')):
            if '.frozen.' in path.name:continue
            row=read(path)
            training.append({'phase':phase,'trial':path.stem,'seed':row['seed'],'status':row['status'],
                'train_scenes':row['train_scenes'],'timestamps_per_epoch':row['total_timestamps'],
                'full_train_coverage':row['full_train_coverage'],'epochs_completed':row.get('epochs_completed',0),
                'source_sha256':row.get('source_provenance',{}).get('sha256'),'history':row.get('history',[]),
                'manifest':str(path),'manifest_sha256':sha(path),
                'checkpoint_sha256':sha(path.with_suffix('.pt')) if path.with_suffix('.pt').is_file() else None})
    coverage={'status':verification['status'] if verification else 'CORRECTED_TRAIN_CACHE_RUNNING',
        'updated_utc':state.get('updated_utc'),'expected_scenes':24,'expected_timestamps':28754,
        'cached_timestamps':sum(r['done'] for r in progress),'completed_cache_scenes':sum(r['done']==r['expected'] for r in progress),
        'detector_input_verified':read(root/'detector_channel_verification.json',{}).get('status')=='PASS_OFFICIAL_FILENAME_EQUIVALENCE',
        'invalidated_cache':read(root/'channel_cache_invalidation.json'),
        'per_scene':[{k:r[k] for k in ['scene','done','expected','counts']} for r in progress],
        'final_training':[ {k:r[k] for k in ['trial','seed','status','train_scenes','timestamps_per_epoch','full_train_coverage','epochs_completed','checkpoint_sha256']}
                          for r in training if r['phase']=='final'],
        'cache_verification_sha256':sha(root/'cache_verification/coverage.json') if verification else None,
        'raw_GT_modified':False}
    coverage['full_train_completed']=len(coverage['final_training'])==7 and all(r['status']=='FULLTRAIN_COMPLETE' and r['full_train_coverage'] for r in coverage['final_training'])
    atomic_json(out/'worldmot_v2_fulltrain_coverage.json',coverage)
    atomic_json(out/'worldmot_v2_training_curves.json',{'status':state['stage'],'trials':training,'no_TEST_selection':True})
    choice=read(root/'selection/selection.json')
    if choice:
        validation=[]
        for row in choice['epoch_validation']:validation.append({'type':'epoch_loss','method':'joint',**row})
        for method,selection in choice['methods'].items():
            for row in selection['candidates']:validation.append({'type':'TRAIN_cached_identity_surrogate','method':method,**row})
        csv_file(out/'worldmot_v2_train_internal_validation.csv',validation,['type','method','epoch','total','assignment','birth','pair',
            'null_margin','birth_threshold','pair_threshold','identity_f1_surrogate','assigned','births','predictions','gt_cached_positive_observations'])
    lock=read(root/'final_test_lock.json',{})
    checkpoints={f'v2_{r["method"]}_seed{r["seed"]}':r for r in lock.get('checkpoints',[])}
    evaluations={p.parent.name:read(p) for p in sorted((root/'test_evaluation').glob('*/metrics.json'))}
    all_fps={};combined=[];per_scene=[]
    for label,evaluation in evaluations.items():
        if evaluation.get('status')!='COMPLETE_VALID_SCOPE':continue
        actual='v1' if label=='v1_observed_bbox' else label
        inference=root/'test'/actual;manifests=[read(p) for p in sorted(inference.glob('scene_*/manifest.json'))]
        complete=[m for m in manifests if m.get('status')=='COMPLETE' and m.get('full_video')]
        if len(complete)!=22 or sum(m['frames'] for m in complete)!=29019:
            raise RuntimeError('refusing official comparison from partial TEST inference: '+label)
        continuity={'delivered_boxes':sum(m.get('delivered_boxes',0) for m in complete),
            'IDs_with_at_least_two_timestamps':sum(m.get('IDs_with_at_least_two_timestamps',0) for m in complete),
            'max_ID_distinct_timestamps':max(m.get('max_ID_distinct_timestamps',0) for m in complete)}
        values=dict(evaluation['TrackEval']['combined'])
        native=evaluation.get('VisionTrack_MATLAB',{})
        if 'combined' in native:values.update({k:native['combined'][k] for k in ['CVIDF1','CVMA']})
        elapsed=0.;samples=[];stages=collections.defaultdict(list);scene_fps={};gids=0
        for m in complete:
            elapsed+=m.get('real_e2e_seconds',m.get('e2e',{}).get('total_s',0.));scene_fps[m['scene']]=m['real_e2e_fps'];gids+=m.get('global_ids',0)
            path=inference/f'scene_{m["scene"]:02d}'/'timings.jsonl'
            if path.exists():
                for line in path.open():
                    r=json.loads(line);samples.append(r['rgb_to_delivered_ids_s']);stages['rgb_read'].append(r['rgb_read_s'])
                    for key,value in r.get('stages_s',{}).items():stages[key].append(value)
        fps=29019/elapsed if elapsed else None
        all_fps[label]={'real_e2e_fps':fps,'timestamps':29019,'scenes':22,'total_s':elapsed,
            'unit':'synchronized dual-camera timestamps/second','e2e_latency':summary(samples),
            'stages':{k:summary(v) for k,v in stages.items()},'per_scene':scene_fps,
            'shared_inference_with':actual if label!=actual else None,
            'identity_delivery':continuity,
            'timing_scope':complete[0]['timing_scope'],'perception_cache_used':False,
            'gpu_models':sorted({m['gpu'] for m in complete}),
            'native_GMT_per_timestamp_delivery_latency_available':False if actual=='gmt_native' else None}
        omega=stages.get('omega_forward',[]);detector=stages.get('detector_forward',[])
        if omega:
            all_fps[label]['omega_model_only_fps']=len(omega)/sum(omega)
            all_fps[label]['omega_model_only_scope']='H2D + frozen Omega aggregator/heads + camera decoding + CUDA synchronization; excludes RGB resize and pointmap postprocess'
        if detector:
            all_fps[label]['detector_backend_only_fps']=len(detector)/sum(detector)
            all_fps[label]['detector_backend_scope']='Both images: MMDetection test pipeline + model + detector output/feature conversion; not a pure GPU-kernel measurement'
        cp=checkpoints.get(label,{})
        row={'method':label,'seed':cp.get('seed'),'checkpoint_sha256':cp.get('sha256',complete[0].get('checkpoint_sha256')),
            'TEST_scenes':len(evaluation['evaluated_scenes']),'TEST_inference_scenes':22,
            'real_e2e_fps':fps,'status':evaluation['status'],**values}
        combined.append(row)
        cv=native.get('per_scene',{})
        tracks=evaluation['TrackEval'].get('per_scene',{})
        for scene,metrics in tracks.items():
            per_scene.append({'method':label,'scene':scene,'status':'OFFICIAL_VALID_GT',**metrics,
                **{k:cv.get(scene,{}).get(k) for k in ['CVIDF1','CVMA']},'checkpoint_sha256':row['checkpoint_sha256']})
        for blocked in evaluation['blocked']:
            per_scene.append({'method':label,'scene':blocked['scene'],'status':blocked['reason'],'checkpoint_sha256':row['checkpoint_sha256']})
    expected=['v1','v1_observed_bbox','gmt_native']+[f'v2_joint_seed{s}' for s in [20261022,20261023,20261024]]+[f'v2_{m}_seed20261022' for m in ['no_omega','no_appearance','two_stage','no_geometry']]
    missing=sorted(set(expected)-{r['method'] for r in combined})
    persistent_main=all(all_fps.get(f'v2_joint_seed{s}',{}).get('identity_delivery',{}).get('IDs_with_at_least_two_timestamps',0)>0 for s in [20261022,20261023,20261024])
    complete=coverage['full_train_completed'] and not missing and persistent_main and all('CVIDF1' in r and 'CVMA' in r for r in combined)
    if evaluations:
        scopes={tuple(sorted(e['evaluated_scenes'])) for e in evaluations.values() if e.get('status')=='COMPLETE_VALID_SCOPE'}
        if len(scopes)>1:raise RuntimeError('official methods have different TEST evaluation scopes')
        report={'status':'COMPLETE_OFFICIAL_INDEPENDENT_TEST' if complete else 'IN_PROGRESS','metrics':combined,'missing_methods':missing,
            'evaluation_scope':list(next(iter(scopes))) if scopes else [],'all_image_sequences_inferred':22,
            'original_GT_modified':False,'official_only':True,'frozen_lock_sha256':sha(root/'final_test_lock.json') if lock else None,
            'detailed_official_reports':{k:str(root/'test_evaluation'/k/'metrics.json') for k in evaluations}}
        report['official_tool_fingerprints']={}
        for name,e in evaluations.items():
            if e.get('status')!='COMPLETE_VALID_SCOPE':continue
            native=e.get('VisionTrack_MATLAB',{})
            tool=e['TrackEval']['tool_sha256']
            report['official_tool_fingerprints'][name]={
                'TrackEval_python_source_set_sha256':hashlib.sha256(json.dumps(tool,sort_keys=True).encode()).hexdigest(),
                'MATLAB_version':native.get('MATLAB_version'),'MATLAB_adapter_sha256':native.get('adapter_sha256'),
                'MATLAB_toolkit_source_set_sha256':hashlib.sha256(json.dumps(native['toolkit_sha256'],sort_keys=True).encode()).hexdigest() if 'toolkit_sha256' in native else None}
        joint=[r for r in combined if r['method'].startswith('v2_joint_seed')]
        if len(joint)==3:
            report['joint_three_seed_summary']={key:{'mean':statistics.mean(r[key] for r in joint),
                'sample_std':statistics.stdev(r[key] for r in joint)} for key in METRICS+['real_e2e_fps'] if all(key in r for r in joint)}
        atomic_json(out/'worldmot_v2_test_official_metrics.json',report)
        atomic_json(out/'worldmot_v2_test_fps.json',{'status':report['status'],'methods':all_fps,'four_GPU_aggregate_is_not_single_video_FPS':True})
        csv_file(out/'worldmot_v2_vs_v1_vs_gmt.csv',combined,['method','seed','status','TEST_scenes','TEST_inference_scenes',*METRICS,'real_e2e_fps','checkpoint_sha256'])
        csv_file(out/'worldmot_v2_test_per_scene.csv',per_scene,['method','scene','status',*METRICS,'checkpoint_sha256'])
        render_report(REPO/'docs/WORLDMOT_V2_FULLTRAIN_TEST_REPORT.md',report,coverage,all_fps)
    if require_complete and not complete:raise RuntimeError('final artifacts incomplete: '+str(missing))
    return {'complete':complete,'stage':state['stage'],'cached_timestamps':coverage['cached_timestamps'],'trained_final_trials':len(coverage['final_training']),'official_methods':len(combined),'missing_official_methods':missing}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='/data1/liuyeqiang/nas/A100/WorldMOT/fulltrain_20261022')
    p.add_argument('--output',default=str(REPO/'results'));p.add_argument('--require-complete',action='store_true');a=p.parse_args()
    print(json.dumps(collect(a.root,a.output,a.require_complete),indent=2))
