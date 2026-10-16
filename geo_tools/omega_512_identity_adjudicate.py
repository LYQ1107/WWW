"""Posthoc sealed-label certification of actual solver forks; no new GT matching."""
import collections
import copy
import gzip
import json
from pathlib import Path
import numpy as np
from omega_common import ROOT,RUNTIME,NATIVE,sha256,save_new,provenance
from omega_modules import geometry_module


def pure(values):
    values=set(values)
    return next(iter(values)) if len(values)==1 and None not in values else None


def read_gz(path):
    with gzip.open(path,'rt') as f:
        for line in f:yield json.loads(line)


def classify(decisions,labels,prior):
    anchors=copy.deepcopy(prior);rows={}
    banks={tuple(q):t for d in decisions if d['stage']=='run_memory_tracker' for q,t in zip(d['queries'],d['accepted_track_ids'])}
    for d in decisions:
        if d['stage']=='run_memory_tracker':continue
        refs=collections.defaultdict(set)
        for r in d['references']:refs[r['track']].add(labels.get(tuple(r['source'])))
        ref_pure={t:pure(v) for t,v in refs.items()}
        for i,source in enumerate(d['queries']):
            key=tuple(source);gt=labels.get(key);accepted=d['accepted_track_ids'][i];final=d['final_track_ids'][i]
            positive=[t for t,v in ref_pure.items() if gt is not None and v==gt]
            identity=ref_pure.get(accepted)
            if gt is None:primary='unknown_query'
            elif accepted<0:primary='missed_known_positive' if positive else 'no_certified_positive'
            elif identity is None:primary='unknown_or_mixed_reference'
            else:primary='correct' if identity==gt else 'wrong'
            anchor=anchors.get(final,set());anchor_id=pure(anchor)
            if gt is None:state='unknown_query'
            elif anchor:
                state='unknown_or_mixed_anchor' if anchor_id is None else ('correct' if anchor_id==gt else 'wrong')
            else:
                existing=any(pure(v)==gt for v in anchors.values())
                state='certified_fragmentation' if positive and existing else 'new_track_unscored'
            rows[key]=dict(query=list(key),query_gt_identity=gt,primary=primary,final_state=state,
                           accepted_primary_track=accepted,final_track=final,positive_legal_tracks=positive,
                           prior_final_anchor=sorted(anchor,key=lambda x:-1 if x is None else x),
                           memory_accepted_track=banks.get(key),solver_pair=next((r for r in d['hungarian_pairs'] if r['row']==i),None))
        # Publish this completed camera's output before the next actual primary call.
        for source,final in zip(d['queries'],d['final_track_ids']):anchors.setdefault(final,set()).add(labels.get(tuple(source)))
    return rows


def main():
    posthoc_source=provenance();assert not posthoc_source['source_status'].strip()
    run=RUNTIME/'omega512_independent_v1/solver_forks'
    manifest=json.loads((run/'manifest.json').read_text());assert manifest['status']=='FINITE_SOLVER_FORKS_COMPLETE'
    assert len(manifest['parity'])==15 and all(all(r[k] for k in ['native_predictions_exact','zero_predictions_exact','zero_solver_and_Bank_exact']) for r in manifest['parity'])
    receipt_path=RUNTIME/'omega512_independent_v1/logs/solver_forks_v1_receipt.json'
    receipt=json.loads(receipt_path.read_text());assert receipt['exit_code']==0
    assert sha256(receipt['log']['path'])==receipt['log']['sha256']
    assert sha256(manifest['output']['path'])==manifest['output']['sha256']
    plan=json.loads((ROOT/'manifests/omega512_identity_counterfactual_preregistration_v1.json').read_text())
    for p,h in plan['frozen_inputs'].items():assert sha256(ROOT/p)==h
    data=json.loads(Path('/data/DATASETS/TRACKING/JDE/VisionTrack/annotations/train.json').read_text())
    metadata={i['id']:i for i in data['images']}
    labels_by_scene={};events_by_scene={};assignments={};seeds={};receipts=[];certified_recovered=0
    cohort=json.loads((ROOT/'manifests/omega_fixed_pair_cohort_v1.json').read_text())
    for scene in sorted({c['scene'] for c in plan['cases']}):
        directory=NATIVE/f'scene{scene}_full_v1'
        bound=next(r for r in cohort['native_source_receipts'] if r['path']==str(directory/'manifest.json'))
        for fn in ['events.jsonl.gz','actual_decisions.jsonl.gz']:assert sha256(directory/fn)==bound['artifacts'][fn]
        events={tuple(e['key'][1:]):e for e in read_gz(directory/'events.jsonl.gz')}
        labels={s:e['query_gt_identity'] for s,e in events.items()}
        assign={s:e['final_output_track'] for s,e in events.items()};initial={};index=0
        for d in read_gz(directory/'actual_decisions.jsonl.gz'):
            if d['stage']=='run_memory_tracker':continue
            for q in d['queries']:
                e=events[tuple(q)];assert e['source_primary_call']==index
                if e['query_gt_identity'] is None:continue
                for r in d['references']:
                    if r['track'] not in e['positive_tracks']:continue
                    key=tuple(r['source']);value=e['query_gt_identity']
                    if key in labels:assert labels[key]==value,('conflicting sealed source certification',scene,key)
                    else:labels[key]=value;certified_recovered+=1
            for r in d['references']:
                key=tuple(r['source'])
                if key not in assign:
                    assert metadata[key[0]]['frame_id']==1,('unexplained missing native query',key)
                    if key in initial:assert initial[key]==r['track']
                    initial[key]=r['track']
            index+=1
        labels_by_scene[scene]=labels;events_by_scene[scene]=events;assignments[scene]=assign;seeds[scene]=initial
        receipts.append(dict(scene=scene,events_sha256=bound['artifacts']['events.jsonl.gz'],decisions_sha256=bound['artifacts']['actual_decisions.jsonl.gz']))
    rows=[];summaries=collections.defaultdict(collections.Counter);corrected=collections.defaultdict(set);changed=[]
    new_gpu_seconds=receipt['child_process_wall_seconds']
    for name in ['omega','old']:
        r=json.loads((RUNTIME/f'omega512_independent_v1/logs/remaining72_{name}_receipt.json').read_text());new_gpu_seconds+=r['child_process_wall_seconds']
    assert new_gpu_seconds<1800,'preregistered combined GPU-visible child wall budget exceeded'
    import torch
    for line in open(manifest['output']['path']):
        case=json.loads(line);scene=case['scene'];group=case['group'];labels=labels_by_scene[scene]
        prior=collections.defaultdict(set)
        for source,track in {**assignments[scene],**seeds[scene]}.items():
            frame=metadata[source[0]]['frame_id']
            if frame<group or (group==1 and source in seeds[scene]):prior[track].add(labels.get(source))
        baseline=classify(case['baseline_decisions'],labels,prior);branch=classify(case['decisions'],labels,prior)
        assert baseline.keys()==branch.keys()
        expected_states={'correct_link':'correct','wrong_link':'wrong','missed_known_positive':'missed_known_positive'}
        unresolved=0
        for source,b in baseline.items():
            original=events_by_scene[scene][source]['actual_primary_state']
            if original in expected_states and b['primary']!=expected_states[original]:
                assert b['primary'] in ['unknown_query','unknown_or_mixed_reference','no_certified_positive'],('native offline classification conflict',scene,source,original,b['primary'])
                unresolved+=1
        # Serialized current proposal order binds submitted IDs to physical sources.
        public_by_image=collections.defaultdict(list)
        for p in case['public_predictions']:public_by_image[p['image_id']].append(p)
        for source,r in branch.items():assert public_by_image[source[0]][source[1]]['track_id']==r['final_track']
        matrix_changes=0
        for m in case['matrices']:
            raw=torch.tensor(m['raw_logits'],dtype=torch.float32);delta=torch.tensor(m['delta'],dtype=torch.float32)
            if raw.numel():assert torch.isfinite(raw).all() and torch.isfinite(delta).all() and float(delta.abs().max())<=1.
            matrix_changes+=int(bool(delta.numel()) and bool(torch.any(delta!=0)))
            if raw.numel():
                summaries[case['arm'],case['weight']]['raw_argmax_reference_changes']+=int(torch.sum(raw.argmax(1)!=(raw+delta).argmax(1)))
            before_scores=np.array(m['before_track_scores']);after_scores=np.array(m['after_track_scores'])
            if before_scores.size:
                summaries[case['arm'],case['weight']]['activated_track_row_rank_changes']+=int(np.any(np.argsort(-before_scores,axis=1,kind='stable')!=np.argsort(-after_scores,axis=1,kind='stable'),axis=1).sum())
        key=(case['arm'],case['weight']);summary=summaries[key]
        summary['native_primary_certification_unresolved']+=unresolved
        summary['cases']+=1;summary['proposal_matrix_calls']+=len(case['matrices']);summary['nonzero_delta_calls']+=matrix_changes
        for source,b in baseline.items():
            a=branch[source];summary['queries']+=1
            primary_correction=b['primary'] in ['wrong','missed_known_positive'] and a['primary']=='correct'
            primary_harm=b['primary']=='correct' and a['primary'] in ['wrong','missed_known_positive']
            correction=b['final_state'] in ['wrong','certified_fragmentation'] and a['final_state']=='correct'
            harm=b['final_state']=='correct' and a['final_state'] in ['wrong','certified_fragmentation']
            final_changed=b['final_track']!=a['final_track']
            if correction:assert final_changed;corrected[key].add((scene,a['query_gt_identity']))
            summary['primary_corrections']+=primary_correction;summary['primary_harms']+=primary_harm
            summary['known_final_corrections']+=correction;summary['known_final_harms']+=harm
            summary['final_ID_changes']+=final_changed
            summary['primary_accepted_assignment_changes']+=b['accepted_primary_track']!=a['accepted_primary_track']
            summary['Bank_invocation_changes']+=(b['memory_accepted_track'] is None)!=(a['memory_accepted_track'] is None)
            summary['Bank_accepted_assignment_changes']+=(b['memory_accepted_track'] if b['memory_accepted_track'] is not None else -1)!=(a['memory_accepted_track'] if a['memory_accepted_track'] is not None else -1)
            summary['unknown_final_comparisons']+=('unknown' in a['final_state'] or 'unknown' in b['final_state'] or a['final_state']=='new_track_unscored' or b['final_state']=='new_track_unscored')
            row=dict(pair_id=case['pair_id'],scene=scene,group=group,arm=case['arm'],weight=case['weight'],source=list(source),
                     baseline=b,branch=a,primary_correction=primary_correction,primary_harm=primary_harm,
                     known_final_correction=correction,known_final_harm=harm,final_ID_changed=final_changed)
            rows.append(row)
            if final_changed or primary_correction or primary_harm:changed.append(row)
    summary_rows=[]
    for (arm,weight),s in summaries.items():
        people=sorted(corrected[arm,weight]);scenes=sorted({v[0] for v in people})
        passes=arm=='G3_omega_F_gated' and s['known_final_corrections']>s['known_final_harms'] and len(people)>=2 and len(scenes)>=2
        summary_rows.append(dict(arm=arm,weight=weight,**s,corrected_scene_qualified_people=people,corrected_scenes=scenes,passes_online_admission=passes))
    passed=any(s['passes_online_admission'] for s in summary_rows)
    result=dict(schema='omega512-native-identity-counterfactual/v1',status='IDENTITY_GATE_PASSED' if passed else 'NO_CERTIFIED_MULTISCENE_NET_FINAL_IDENTITY_GAIN',
        posthoc_source=posthoc_source,role='INDEPENDENT_512_NOT_416',cases=15,variants_per_case=11,scope='fixed development currentgroup counterfactual, no futurebranch propagation or fullvideo claim',
        recovered_source_labels_by_existing_pure_positive_certification=certified_recovered,sealed_label_receipts=receipts,
        no_new_GT_matching=True,inference_GT_input=False,summary=summary_rows,changed_events=changed,all_queries=rows,
        producer_manifest=manifest,process_clock=receipt,combined_new_GPU_visible_process_wall_seconds=new_gpu_seconds,
        underlying_output=dict(path=manifest['output']['path'],sha256=manifest['output']['sha256']),
        online_fusion='CONDITIONAL_NOT_EXECUTED' if passed else 'NOT_EXECUTED_IDENTITY_GATE_FAILED',full_tracking='NOT_EXECUTED',
        actual_system_FPS='NOT_MEASURED',depth='NOT_EXECUTED_UNCALIBRATED',rules=json.loads((ROOT/'manifests/omega512_identity_adjudication_rules_v1.json').read_text()))
    save_new(ROOT/'results/omega512_identity_counterfactual_v1.json',result)
    print(json.dumps(dict(status=result['status'],summary=summary_rows,changed_events=len(changed))),flush=True)

if __name__=='__main__':main()
