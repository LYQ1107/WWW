"""Read existing sealed native audit and selected caches; never mine GT or run GMT."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import time

FIT = (4,5,6,7,8,9,13,17,20,21,24)
INVENTORY_SHA = '5bfdfcd9971901242fa701494ab59c579e2de84404f20b1ee1d0809374e1c5d4'
V3_SHA = '927965246997778d4efe0ec460ffff8cd6e2c31ee09582cbce8d8c1fed16b6d9'
STAGE2_SHA = 'cd72823824d16c86ed27c2dfc8323de610aa9f6c0c0b29249aa3de609deabce8'


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def read_rows(path):
    with gzip.open(path,'rt') as f:
        for line in f:yield json.loads(line)


def checked(path, expected):
    if sha(path)!=expected:raise ValueError('Source SHA changed: '+str(path))
    return Path(path)


def classify(e):
    """PIRGA labels differ from historical A-F; overlapping by definition."""
    known=e['query_gt_identity'] is not None
    positive=bool(e['positive_tracks'])
    observed=bool(e['observed_positive_including_impure'])
    return dict(A_rejected_legal_positive=known and e['actual_primary_state']=='missed_known_positive',
                B_accepted_wrong_identity=known and e['actual_primary_state']=='wrong_link',
                C_correct_observation_only_impure=known and observed and not positive,
                D_no_pure_legal_positive=known and not positive,
                E_uncertified=not known or e['actual_primary_state'] in ('ambiguous_reference','unknown_candidate_coverage'),
                known_query=known,pure_positive=positive,
                known_wrong_reachable=e['known_wrong'] and positive)


def run(producer, runtime, out):
    started=time.monotonic();out.mkdir(parents=True,exist_ok=False)
    ip=producer/'results/trackvggt_full_train_event_inventory_v1.json'
    checked(ip,INVENTORY_SHA);inventory=json.loads(ip.read_text())
    if inventory['status']!='COMPLETE_12_ASSOCIATION_TRAIN_FULL_CAUSAL_AUDITS':raise ValueError('Incomplete audit')
    source={str(ip):INVENTORY_SHA}
    # Existing sealed artifacts only; no raw GT file opened or GT source recomputed.
    for name,digest in inventory['source_artifacts'].items():
        p=Path(name).resolve()
        if not (p.is_relative_to(runtime.resolve()) or p.is_relative_to(producer.resolve())):raise ValueError('Unapproved artifact root')
        checked(p,digest);source[str(p)]=digest
    cohort=runtime/'cohorts/hard_events_expanded_v3/cohort.json';checked(cohort,V3_SHA)
    c=json.loads(cohort.read_text());source[str(cohort)]=V3_SHA
    selected=Counter(r['scene'] for r in c['fit']);rows=[];totals=Counter();people=set();hard_people=set();keys=set()
    for scene in FIT:
        base=runtime/'full_train_audit_v1'/f'scene{scene}_full_v1'
        mp=base/'manifest.json';m=json.loads(mp.read_text())
        if m['status']!='COMPLETE_FULL_CAUSAL_AUDIT' or m['video_id']!=scene or m['seed']!=20261014 or not m['prefix40_raw_prediction_exact']:raise ValueError('Source parity/scope invalid')
        for name,digest in m['artifacts'].items():checked(base/name,digest)
        summary=json.loads((base/'summary.json').read_text());counts=Counter();scene_people=set();scene_hard=set();physical_queries=set()
        for e in read_rows(base/'events.jsonl.gz'):
            key=tuple(e['key'])
            if key in keys or key[0]!=scene:raise ValueError('Duplicate/out of scope query')
            keys.add(key);physical_queries.add(tuple(key[1:]));counts['actual_queries']+=1
            for k,v in classify(e).items():counts[k]+=bool(v)
            if e['query_gt_identity'] is not None:
                p=(scene,e['query_gt_identity']);people.add(p);scene_people.add(p)
                if e['known_wrong']:hard_people.add(p);scene_hard.add(p)
        if counts['actual_queries']!=summary['counts']['N_total'] or counts['A_rejected_legal_positive']+counts['B_accepted_wrong_identity']!=summary['counts']['N_wrong']:raise ValueError('Summary count differs')
        # Query-only legacy rows do not constitute a complete joint solver context.
        import torch
        smp=runtime/'selected_native_v3'/f'scene{scene}_full_cycle_v1/manifest.json'
        sm=json.loads(smp.read_text());sp=smp.parent/'events.pt'
        checked(sp,sm['artifacts']['events.pt']);source[str(smp)]=sha(smp);source[str(sp)]=sha(sp)
        payload=torch.load(sp,map_location='cpu',weights_only=True)
        obs=payload['observations'];available=set(obs)
        # Do not infer complete inputs merely from a cached query vector.
        cached_query_keys={tuple(e['query_source']) for e in payload['events']}
        counts['selected_cache_rows']=len(payload['events']);counts['query_vectors_in_selected_cache']=len(physical_queries&available)
        counts['selected_query_input_rows']=len(physical_queries&cached_query_keys)
        counts['V3_selected_fit_queries']=selected[scene]
        counts['complete_full_context_queries']=0
        detector_count=sum(1 for _ in read_rows(base/'detector_provenance.jsonl.gz'))
        native_seconds=m['inference_seconds']
        row=dict(scene=scene,role='fit',groups=m['groups'],images=m['images'],counts=dict(counts),known_scene_person_groups=len(scene_people),known_error_person_groups=len(scene_hard),
                 actual_physical_observations=detector_count,raw_native_vectors_FP32_lower_bound_bytes=detector_count*1152*4,
                 raw_native_vectors_FP16_lower_bound_bytes=detector_count*1152*2,
                 historical_full_native_inference_seconds=native_seconds,historical_cost_is_new_measurement=False,
                 selected_cache_native_dimension=payload['feature_dim'],selected_cache_full_vector_archive=sm['full_native_vector_archive'],
                 selected_cache_only_query_rows=sm['only_selected_query_rows_of_matrices'],complete_full_context_ratio=0.,
                 ranking_supervision_eligible=None,prediction_supervision_eligible=None,
                 eligibility_note='Requires full indexed features and independently masked candidate identities; positive presence is not rank/prediction eligibility.')
        rows.append(row);totals.update(counts);print(json.dumps(dict(scene=scene,queries=counts['actual_queries'],selected_rows=counts['selected_query_input_rows'],full_context=0)),flush=True)
    if totals['actual_queries']!=154911 or totals['A_rejected_legal_positive']+totals['B_accepted_wrong_identity']!=1527:raise ValueError('Frozen census differs')
    audit=dict(status='DATA_INCOMPLETE',fit_scenes=list(FIT),development_scenes=[12],official_TEST_opened=False,raw_GT_read_or_recomputed=False,
               full_GMT_audit_repeated=False,full_context_queries=0,totals=dict(totals),known_scene_person_groups=len(people),known_error_person_groups=len(hard_people),
               per_scene=rows,definitions={'A':'GMT rejection with actual pure legal positive','B':'GMT accepted certified wrong identity; reachable subset separate','C':'Known matching observation trapped only in impure candidate history','D':'Known query without pure legal positive; overlaps C','E':'Unknown query or uncertified primary identity decision; overlaps C/D','independence':'Counts are query exposures; consecutive rows are not independent people'},
               incomplete_not_no_signal=True,wall_seconds=time.monotonic()-started,gpu_wall_hours=0,
               producer_current_HEAD=subprocess.check_output(['git','rev-parse','HEAD'],cwd=producer,text=True).strip())
    (out/'DATA_SCOPE_AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')
    (out/'SOURCE_SHA_MANIFEST.json').write_text(json.dumps(dict(artifacts=source),indent=2)+'\n')
    (out/'SUPERVISION_COVERAGE.json').write_text(json.dumps(dict(status='NOT_ADMITTED_FULL_INPUTS_MISSING',actual_queries=totals['actual_queries'],known_queries=totals['known_query'],pure_positive_queries=totals['pure_positive'],
        ranking_supervised_queries=None,crossview_prediction_queries=None,decision_margin_queries=None,trained_queries=0,
        note='No inference from154911 or pure positives to supervision count; full actor/candidate/time masks must pass first.'),indent=2)+'\n')
    (out/'FULLTRAIN_ASSET_MANIFEST.json').write_text(json.dumps(dict(status='DATA_INCOMPLETE',interface='pirga-indexed-native/v1',
        fit_scenes=list(FIT),expected_actual_queries=154911,complete_context_queries=0,trainable_full_scene_release=None,
        selected_cohort_only=3000,source_sha_manifest='SOURCE_SHA_MANIFEST.json',feature_dimension=1152),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--producer',type=Path,required=True);p.add_argument('--runtime',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.producer,a.runtime,a.out)
