"""Offline Phase V audits. GT is never supplied to an online policy."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = Path('/home/liuyeqiang/WWW_jev_rng_v4_runtime/segmented_gate_20261007_v4')
OUT = Path('/home/liuyeqiang/WWW_jev_phase5_runtime/20261007')
TRAIN = Path('/data/DATASETS/TRACKING/JDE/VisionTrack/annotations/train.json')
CACHE = Path('/data1/liuyeqiang/WWW/outputs/research_final_v2/off/perception_cache_train')
TRACE = Path('/home/liuyeqiang/WWW_jev_rng_v4_runtime/segmented_native_trace_20261007/video01/trace_video_01.jsonl')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''): h.update(b)
    return h.hexdigest()


def save(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    value = {'official_test_read': False, 'full24_authorized': False, **value}
    p = OUT / name
    p.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    return value


def records(video):
    with (BASE / f'video{video:02d}_records.jsonl').open() as f:
        for l in f: yield json.loads(l)


def key(d):
    c = d.get('context', d.get('state', {}).get('online_context', d))
    return int(c.get('frame',d.get('frame',0))), int(c.get('view',d.get('view',0))), str(d.get('question',d.get('question_type'))), int(c.get('detection_index',d.get('row',0)))


def stats(values):
    a = np.asarray(values, dtype=float)
    return {'n':len(a), 'mean':float(a.mean()) if len(a) else None,
            'min':float(a.min()) if len(a) else None, 'max':float(a.max()) if len(a) else None,
            'nonzero':int(np.count_nonzero(np.abs(a)>1e-8))}


def memory():
    by_video = {}
    for v in (1,6,7):
        rows = [r for r in records(v) if r['question_type']=='MEMORY_DECISION']
        fields = ['utility','memory_contamination','future_identity_switches','future_fragmentation',
                  'window_assa_proxy','future_correct_identity_duration','false_reactivation']
        diffs = {f:[r['action_outcomes']['WRITE_MEMORY'][f]-r['action_outcomes']['SKIP_MEMORY'][f] for r in rows] for f in fields}
        outcome_diffs = sum(r['action_outcomes']['WRITE_MEMORY']!=r['action_outcomes']['SKIP_MEMORY'] for r in rows)
        by_video[str(v)] = {'records':len(rows),'tie_rate':sum(abs(x)<=1e-8 for x in diffs['utility'])/max(1,len(rows)),
                            'write_minus_skip':{f:stats(x) for f,x in diffs.items()},
                            'different_outcomes':outcome_diffs,
                            'nonzero_weight_ties':sum(abs(x)<=1e-8 and r['sample_weight']>0 for r,x in zip(rows,diffs['utility'])),
                            'record_sha256':sha(BASE / f'video{v:02d}_records.jsonl')}
    return save('MEMORY_SUPERVISION_AUDIT.json',{'status':'COMPLETE','scope':'all existing MEMORY records, frozen H8 protocol',
                'videos':by_video,'key_limitations':['Utility resets identity and memory anchors at each branch start.',
                 'Fixed OFF event-map actions are replayed into changed branch states.',
                 'video06 has no traced reactivation; ordinary GMT transformer reads association history, not memory bank.',
                 'Existing score_rollout GT coordinate mapping requires separate audit.']})


def native():
    from gtr.modeling.jev_state import feature_names
    from run_early_pilot_tracking import feature_parity_tolerance, DEFAULT_SCALE_SENSITIVE_RELATIVE_TOLERANCE
    names = list(feature_names(64))
    envelope = {'feature_names':names, 'tolerance':2e-5,
                'relative_tolerance':DEFAULT_SCALE_SENSITIVE_RELATIVE_TOLERANCE}
    legacy = {key(d):d for d in (json.loads(l) for l in TRACE.open())}
    summary = {}
    for question in ['MATCH_DECISION','MEMORY_DECISION','REACTIVATION_DECISION']:
        differences = []
        failures = []
        examples = []
        for r in records(1):
            if r['question_type'] != question: continue
            d = legacy[key(r)]
            diff = np.abs(np.asarray(d['state_feature_vector'])-np.asarray(r['state']['feature_vector']))
            differences.append(diff)
            limits = np.asarray([feature_parity_tolerance(envelope,i,r['state']['feature_vector'][i]) for i in range(64)])
            failed = diff > limits
            failures.append(failed)
            if np.any(failed) and len(examples)<5:
                examples.append({'key':list(key(r)), 'mismatching_features':[
                    {'name':names[i],'native':d['state_feature_vector'][i],'mutable':r['state']['feature_vector'][i]}
                    for i in range(64) if failed[i]]})
        a = np.asarray(differences)
        f = np.asarray(failures)
        summary[question] = {'records':len(a),'records_with_abs_error_above_2e_5':int(np.count_nonzero((a>2e-5).any(axis=1))),
                             'records_outside_frozen_float32_envelope':int(f.any(axis=1).sum()),
                             'per_feature':[{'name':n,'mismatch_count':int(np.count_nonzero(a[:,i]>2e-5)),
                                             'mismatch_count_outside_envelope':int(f[:,i].sum()),
                                             'max_abs_error':float(a[:,i].max())} for i,n in enumerate(names)],'examples':examples}
    return save('NATIVE_CONTROLLER_FEATURE_CONTRACT.json',{
        'status':'DIFFERENT_SEMANTICS_CONFIRMED','native_controller_feature_vector_equivalence_proven':False,
        'trace_sha256':sha(TRACE),'frozen_mutable_record_sha256':sha(BASE/'video01_records.jsonl'),
        'exact_pairing':'canonical mapped detection rows; same OFF events and candidates', 'by_question':summary,
        'native_bank_semantics':{'frame_index':'local bank query k','view':0,'window_length':'local historical/current T'},
        'mutable_research_semantics':{'frame_index':'real zero-based video frame','view':'real zero-based camera','window_length':'ordinary production association window'},
        'deployment_decision':'Use an explicitly versioned canonical online research contract for the WWW policy. A native integration must implement and separately prove that contract; existing native vectors cannot be called equivalent.',
        'native_adapter_changed_in_this_phase':False,'native_deployment_claim_allowed':False})


def coverage():
    directory = Path('/home/liuyeqiang/WWW_jev_full_h8_runtime/partition/trace_by_video')
    report = []
    for p in sorted(directory.glob('video_??.jsonl')):
        v = int(p.stem.split('_')[-1]);counts=Counter();candidates=[];actions=Counter()
        for line in p.open():
            d=json.loads(line);q=d.get('question');counts[q]+=1
            if q=='REACTIVATION_DECISION':
                c=d.get('context',{});candidates.append(int(c.get('native_candidate_count',c.get('candidate_count',len(c.get('candidate_track_ids',[]))))));actions[d.get('committed_action',d.get('off_action'))]+=1
        report.append({'video':v,'trace':str(p),'trace_sha256':sha(p),'question_counts':dict(counts),
                       'reactivation_events':counts['REACTIVATION_DECISION'],'candidate_count':stats(candidates),
                       'actions':dict(actions),'source_contract':'legacy trace metadata, not newly canonical labels',
                       'reactivation_utility_margin':None,'margin_unavailable_reason':'Trace has no paired counterfactual outcomes; do not infer margin from action or score.'})
    candidates=sorted([x for x in report if x['video']!=1 and x['reactivation_events']>0],key=lambda x:(-x['reactivation_events'],x['video']))
    train=[];val=[]
    for x in candidates:
        if sum(y['reactivation_events'] for y in train)<200:train.append(x)
        elif sum(y['reactivation_events'] for y in val)<50:val.append(x)
        if len(train)+len(val)>=5:break
    return save('TRAIN_REACTIVATION_COVERAGE.json',{'status':'METADATA_AUDIT_COMPLETE','videos':report,
        'predeclared_selection':{'train':[x['video'] for x in train],'val':[x['video'] for x in val],'held_out':[1],
          'train_events':sum(x['reactivation_events'] for x in train),'val_events':sum(x['reactivation_events'] for x in val),
          'rule':'excluding video01, descending event count then video ID; smallest full-video sets reaching train>=200 and val>=50, <=5 videos total'},
        'new_build_started':False,'selected_traces_are_training_labels':False})


def coordinates():
    from gtr.modeling.jev_perception_cache import FrozenPerceptionCache
    from build_jev_counterfactual_dataset import load_gt
    from build_jev_counterfactual_v2 import detection_target
    _,images,gt,meta=load_gt(TRAIN)
    # Only offline GT lookup is changed; perception and policy are untouched.
    corrected={(v,w,f-1):image_id for (v,w,f),image_id in images.items()}
    cache=FrozenPerceptionCache(CACHE);report={}
    for v in (1,6,7):
        count=wrong_key=old_missing=changed=old_unmatched=new_unmatched=0;examples=[]
        for k in sorted(x for x in cache.keys() if x[0]==v):
            payload=cache.load(*k);vid,frame,view=k
            expected=images[(vid,view+1,frame+1)];legacy=images.get((vid,view+1,frame),images.get((vid,view,frame)))
            count+=1;wrong_key+=legacy!=expected;old_missing+=legacy is None
            for row,b in enumerate(payload['pred_boxes']):
                kw={'video_id':vid,'frame':frame,'view':view,'box':b.tolist(),'image_size':payload['image_size'],'gt_by_image':gt,'image_meta':meta}
                old=detection_target(images=images,**kw);new=detection_target(images=corrected,**kw)
                assert new == detection_target(images=images,gt_coordinate_contract='cache0_annotation1',**kw)
                changed+=old!=new;old_unmatched+=old is None;new_unmatched+=new is None
                if old!=new and len(examples)<8:examples.append({'key':list(k),'row':row,'legacy_gt':old,'aligned_gt':new,'expected_image_id':expected,'legacy_image_id':legacy})
        report[str(v)]={'cache_keys':count,'incorrect_legacy_image_keys':wrong_key,'missing_legacy_images':old_missing,
                        'changed_detection_targets':changed,'legacy_unmatched':old_unmatched,'aligned_unmatched':new_unmatched,'examples':examples}
    return save('LABEL_COORDINATE_AUDIT.json',{'status':'CONFIRMED_LABEL_GT_FRAME_OFFSET','videos':report,
        'cache_convention':'frame=0 maps to metadata.dataset_frame=1 / annotation frame_id=1; view=0 maps to view_id=1',
        'frozen_labeler_lookup':'(video_id, view+1, frame), fallback (video_id, view, frame)',
        'required_lookup':'(video_id, view+1, frame+1), exact dataset convention',
        'scope':'complete cached keys/detections for videos1/6/7; raw annotations and perception unchanged',
        'closed_loop_tracking_metrics_invalidated':False,'supervision_requires_relabeling_before_architecture_claim':True,
        'new_explicit_contract_matches_normalized_lookup_on_all_detections':True,
        'new_builder_CLI_default_contract':'cache0_annotation1',
        'legacy_programmatic_default_retained_for_frozen_reproduction':True})


def aggregate():
    rows={v:json.loads((OUT/'ablations'/v/'result.json').read_text()) for v in ('A0','A1','A2','A3','A4')}
    off=rows['A0']['metrics']
    for v,d in rows.items():d['delta_vs_A0']={k:d['metrics'][k]-off[k] for k in off}
    questions={'MATCH_only_beats_GMT_on_AssA_and_HOTA':all(rows['A1']['metrics'][k]>off[k] for k in ('AssA','HOTA')),
               'MATCH_REACTIVATION_beats_GMT_on_AssA_and_HOTA':all(rows['A2']['metrics'][k]>off[k] for k in ('AssA','HOTA')),
               'MEMORY_effect_with_GMT_reactivation':{k:rows['A3']['metrics'][k]-rows['A1']['metrics'][k] for k in off},
               'MEMORY_effect_with_JEV_reactivation':{k:rows['A4']['metrics'][k]-rows['A2']['metrics'][k] for k in off},
               'REACTIVATION_effect_with_GMT_memory':{k:rows['A2']['metrics'][k]-rows['A1']['metrics'][k] for k in off},
               'REACTIVATION_effect_with_JEV_memory':{k:rows['A4']['metrics'][k]-rows['A3']['metrics'][k] for k in off}}
    value=save('COMPONENT_ABLATION.json',{'status':'COMPLETE','variants':rows,'answers':questions,
       'attribution_scope':'whole-component intervention, with all subsequent decisions recomputed online; pairwise component effects may interact',
       'checkpoint_unchanged':True,'retrained':False})
    lines=['# 冻结 JEV checkpoint 组件消融','', '| Variant | HOTA | AssA | IDF1 | IDSW | MOTA |','| --- | ---: | ---: | ---: | ---: | ---: |']
    for v,d in rows.items():
        m=d['metrics'];lines.append(f"| {v} | {m['HOTA']:.4f} | {m['AssA']:.4f} | {m['IDF1']:.4f} | {m['IDSW']:.0f} | {m['MOTA']:.4f} |")
    lines+=['','同一个冻结 checkpoint；video01 为 TRAIN 内诊断序列。A0/A4 均要求预测 SHA 和指标复现旧基线。', '', '完整动作计数、条件组件效应及历史 OFF 标签代理计数的限制见 COMPONENT_ABLATION.json。']
    (OUT/'COMPONENT_ABLATION.md').write_text('\n'.join(lines)+'\n')
    return value


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['memory','native','coverage','coordinates','aggregate']);args=p.parse_args()
    for path in (ROOT,ROOT/'third_party/CenterNet2',ROOT/'reproduction_tools'):sys.path.insert(0,str(path))
    result=globals()[args.mode]()
    print(json.dumps({'mode':args.mode,'status':result['status']}))


if __name__=='__main__':main()
