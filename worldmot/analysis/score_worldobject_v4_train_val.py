#!/usr/bin/env python3
"""Aggregate all six complete TRAIN videos and score original legal 4/8/12."""
import argparse
from collections import Counter,defaultdict
import csv
import json
import os
from pathlib import Path
import subprocess
import time
from worldobject_v4_audit_common import VAL_SCENES,LEGAL_VAL,write_json,code_fingerprint


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path)
    p.add_argument('--r2-root',required=True,type=Path);a=p.parse_args()
    repo=Path(__file__).resolve().parents[2];started=time.perf_counter();fp=code_fingerprint(repo)
    from worldmot.runtime.experiment_authorization import source_fingerprint,source_commit,APPROVAL_SCHEMA
    from worldmot.data.worldobject_cache import sha256
    reports={sid:json.loads((a.root/f'scene_{sid:02d}/replay_report.json').read_text()) for sid in VAL_SCENES}
    if any(r['status']!='COMPLETE' or r['source_sha256']!=fp for r in reports.values()):
        raise ValueError('ALL_SIX_COMPLETE_SOURCE_BOUND_TRAIN_VIDEOS_REQUIRED')
    if sum(r['timestamps'] for r in reports.values())!=5860:raise ValueError('EXACT_5860_TRAIN_TIMESTAMPS_REQUIRED')
    if any(r['C0_R2_exact_frames']!=r['timestamps'] for r in reports.values()):raise ValueError('C0_REAL_R2_EXACTNESS_REQUIRED')
    variants=list(reports[VAL_SCENES[0]]['closed_loop'])
    if any(set(r['closed_loop'])!=set(variants) for r in reports.values()):raise ValueError('VARIANT_SCOPE_DIFFERENCE')
    approval=a.root/'authorization_TRAIN_native_scoring.json'
    user_attachment=Path('/home/liuyeqiang/.codex/attachments/ff57cea5-0ab1-48bf-8828-4a9ad6561eac/已粘贴的文本.txt')
    write_json(approval,{'schema':APPROVAL_SCHEMA,'approved':True,'user_authorization_reference':
        'WorldMOT V4 explicit TRAIN-only replay and legal TRAIN VAL native scoring; attachment SHA '+sha256(user_attachment),
        'source_commit':source_commit(),'source_sha256':source_fingerprint(),'operations':['evaluate'],
        'splits':['train'],'method_ids':['FULL_WORLD_OBJECT_V3'],'no_new_training':True,'V4_TEST_authorized':False})
    aggregate={};factorial=defaultdict(Counter);runtime={};b1=defaultdict(Counter);b1_decisions=defaultdict(Counter)
    evaluation=a.root/'native_TRAIN_VAL';evaluation.mkdir(exist_ok=True)
    for variant in variants:
        prediction_root=a.root/'combined_predictions'/variant;prediction_root.mkdir(parents=True,exist_ok=True)
        totals=Counter();legal_totals=Counter();contrasts=Counter()
        for sid,r in reports.items():
            origin=a.root/f'scene_{sid:02d}'/variant/f'scene_{sid:02d}'
            link=prediction_root/f'scene_{sid:02d}'
            if not link.exists():link.symlink_to(origin.resolve(),target_is_directory=True)
            manifest=json.loads((origin/'manifest.json').read_text())
            if sha256(origin/'tracks.jsonl')!=manifest['tracks_sha256']:raise ValueError('PREDICTION_CHANGED_BEFORE_NATIVE_SCORING')
            totals.update(r['closed_loop'][variant]['counts'])
            if sid in LEGAL_VAL:legal_totals.update(r['closed_loop'][variant]['counts'])
            if variant!='C0':contrasts.update(r['corrections_and_regressions'][variant])
        target=evaluation/variant
        if not (target/'metrics.json').exists():
            env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
                 'PYTHONPATH':str(repo/'worldmot')}
            command=['/home/liuyeqiang/anaconda3/bin/python',str(repo/'worldmot/scripts/evaluate_fullvideo.py'),
                '--authorization',str(approval),'--dataset-root','/data/DATASETS/TRACKING/JDE/VisionTrack','--split','train',
                '--audit',str(a.r2_root.parent/'fulltrain_20261022/audit/data_audit.json'),
                '--predictions',str(prediction_root),'--output',str(target),'--scenes',*map(str,VAL_SCENES),
                '--method-id','FULL_WORLD_OBJECT_V3','--skip-matlab']
            with (evaluation/f'{variant}.log').open('x') as log:
                subprocess.run(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
        native=json.loads((target/'metrics.json').read_text())
        if 'TrackEval' not in native:raise ValueError('NATIVE_TRAIN_EVALUATION_INCOMPLETE')
        aggregate[variant]={'all6_clean_subset_diagnostic':dict(totals),'legal3_clean_subset_diagnostic':dict(legal_totals),
            'original_legal_TRAIN_VAL_TrackEval':native['TrackEval']['combined'],'paired_against_C0':dict(contrasts),
            'native_scope_scenes':list(LEGAL_VAL),'previously_seen_by_final_all24_head':True}
        runtime[variant]={'per_scene':{str(s):r['timings'][variant] for s,r in reports.items()},
            'search_totals':dict(sum((Counter(r['search'][variant]) for r in reports.values()),Counter())),
            'FPS_type':'CACHE_REPLAY_ONLY_NOT_LIVE'}
        print(json.dumps({'native_TRAIN_VAL_complete':variant,'metrics':native['TrackEval']['combined']}),flush=True)
    for r in reports.values():
        for key,c in r['B1_fixed_C0_past']['counts'].items():b1[key].update(c)
        for key,c in r['B1_fixed_C0_past']['conditional_current_decision_diagnostics'].items():b1_decisions[key].update(c)
        for key,c in r['factorial_single_factor_edges'].items():factorial[key].update(c)
    write_json(a.root/'v4_frozen_geometry_factorial.json',{'status':'COMPLETE','source_sha256':fp,'checkpoint_sha256':reports[4]['checkpoint']['checkpoint_sha256'],
        'B1_fixed_history':{k:dict(v) for k,v in b1.items()},'B1_counterfactual_state_propagated':False,
        'B1_conditional_current_decision_diagnostics':{k:dict(v) for k,v in b1_decisions.items()},
        'B2_closed_loop':{k:v for k,v in aggregate.items() if k.startswith('G') or k=='C0'},
        'B2_single_factor_paired_edges':{k:dict(v) for k,v in factorial.items()},
        'G4_interpretation':'zeroing geometry inputs of a fixed geometry-trained head measures sensitivity, not independent learned-geometry benefit',
        'TRAIN_only':True,'TEST_read':False})
    write_json(a.root/'v4_closed_loop_identity_attribution.json',{'status':'COMPLETE','source_sha256':fp,'variants':aggregate,
        'TRAIN_only':True,'official_MOT_scope':'full TRAIN scenes 4/8/12; never TEST','runtime':runtime})
    retained_path=a.root/'retained_case/replay_report.json'
    retained=json.loads(retained_path.read_text()) if retained_path.is_file() else None
    write_json(a.root/'v4_error_correction_attribution.json',{'status':'COMPLETE','TRAIN_only':True,
        'C0_contrasts':{k:v['paired_against_C0'] for k,v in aggregate.items() if k!='C0'},
        'per_scene':{str(s):r['corrections_and_regressions'] for s,r in reports.items()},
        'correction_and_regression_examples':[x for r in reports.values() for x in r['paired_correction_and_failure_examples']],
        'mechanism_t704_t705':{str(s):r['mechanism_t704_t705'] for s,r in reports.items() if r['mechanism_t704_t705']},
        'separate_original_retained_epoch1_failure_case':None if retained is None else {
            'checkpoint':retained['checkpoint'],'scope':'705-time TRAIN scene24 prefix, not primary C0-C5 final-head comparison',
            'assertions':retained.get('retained_original_case_assertions'),
            'corrections_and_regressions':retained['corrections_and_regressions'],
            'counts':retained['closed_loop']}})
    with (a.root/'v4_train_val_comparison.csv').open('w',newline='') as f:
        fields=['variant','HOTA','AssA','IDF1','MOTA','IDSW','same_person_splits','different_person_merges','temporal_ID_changes',
                'global_one_to_one_retained_observations','mixed_global_IDs','scope']
        writer=csv.DictWriter(f,fields,lineterminator='\n');writer.writeheader()
        for k,v in aggregate.items():
            native=v['original_legal_TRAIN_VAL_TrackEval'];diag=v['all6_clean_subset_diagnostic']
            writer.writerow({'variant':k,**{n:native[n] for n in ('HOTA','AssA','IDF1','MOTA','IDSW')},
                **{n:diag.get(n,0) for n in fields[6:-1]},'scope':'native full TRAIN 4/8/12; diagnostics all six TRAIN VAL'})
    write_json(a.root/'v4_runtime_analysis.json',{'status':'CACHE_REPLAY_COMPLETE_LIVE_PENDING','variants':runtime,
        'worker_wall_s':sum(r['worker_total_wall_s'] for r in reports.values()),'scoring_wall_s':time.perf_counter()-started,
        'omega_requests':{str(s):{'requests':sum(x['action']=='REQUEST_OMEGA' for x in r['requests']),
            'total':len(r['requests'])} for s,r in reports.items()},
        'true_LIVE_FPS':None,'live_scope_limit':'request counts do not validate sparse live alignment or RGB-to-ID speed'})


if __name__=='__main__':main()
