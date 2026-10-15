"""Consume a SHA-sealed local miner release on the precommitted asset keys.

No new GT matching, error mining, videos, features, model fitting or cohort
selection. Git synchronization is recorded separately from artifact validity.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime,timezone
import gzip
import json
from pathlib import Path

import torch

from cv_pil_native_handoff import (sha256,get_observation,source_key,load_trackvggt_bundle,
    FROZEN_STAGE2_SHA256,FROZEN_CONFIG_SHA256)


def records(path):
    with gzip.open(path,'rt') as stream:
        for line in stream:yield json.loads(line)


def joined_row(bundle,asset,event,old_decision):
    body=bundle['payload'];call=body['calls'][int(asset['call_index'])];qi=int(asset['query_row'])
    key=(int(asset['image_id']),int(asset['row_id']));query=get_observation(body['observations'],key)
    if source_key(call['queries'][qi])!=key or event['key']!=[int(asset['scene']),*key]:
        raise ValueError('Physical query pointer mismatch')
    if event['frame']!=query['frame'] or event['camera']!=query['camera']:
        raise ValueError('Physical time/camera mismatch')
    scores=call['track_scores'][qi];columns=torch.argsort(scores,descending=True,stable=True)[:8].tolist()
    ids=[call['track_ids'][i] for i in columns]
    if ids!=event['top8_tracks']:raise ValueError('Full-audit/native-cache candidate ranking drift')
    if (event['accepted_primary_track'],event['final_output_track'])!=old_decision:
        raise ValueError('Cached/native-full primary or final commit drift')
    positives=set(event['positive_tracks'])
    if not positives <= set(call['track_ids']):raise ValueError('Producer positive is not a legal original candidate')
    past_cross_positive=False
    for tid in ids:
        for src in call['histories'][tid]['sources']:
            obs=get_observation(body['observations'],source_key(src))
            if obs['frame']>query['frame']:raise ValueError('Future native source')
            if tid in positives and obs['frame']<query['frame'] and obs['camera']!=query['camera']:
                past_cross_positive=True
    wrong=event['actual_primary_state'] in ('wrong_link','missed_known_positive')
    if wrong!=event['known_wrong']:raise ValueError('Inconsistent producer primary-error flag')
    known=event['query_gt_identity'] is not None
    if not known and (positives or wrong):raise ValueError('UNKNOWN treated as certified positive/error')
    # A rejected primary row may be repaired by Bank or allocated a new ID.
    # A new numeric ID is not a GT identity error by literal equality.
    certified_final_wrong=(event['actual_primary_state']=='wrong_link' and
        event['accepted_primary_track']>=0 and event['final_output_track']==event['accepted_primary_track'])
    return dict(scene=int(asset['scene']),image_id=key[0],row_id=key[1],
        GT_person_OFFLINE_ONLY=event['query_gt_identity'],frame=query['frame'],camera=query['camera'],
        actual_primary_state=event['actual_primary_state'],native_top1_state=event['ranking_state'],
        known_primary_wrong_or_missed=wrong,pure_positive_top8=bool(positives & set(ids)),
        strict_past_cross_view_positive=past_cross_positive,
        reachable_primary_error=wrong and bool(positives & set(ids)) and past_cross_positive,
        certified_accepted_final_wrong=certified_final_wrong,
        original_primary_track=event['accepted_primary_track'],original_final_track=event['final_output_track'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--inventory',type=Path,required=True)
    p.add_argument('--inventory-sha',required=True);p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--compatibility',type=Path,required=True);p.add_argument('--p0-audit',type=Path,required=True)
    p.add_argument('--recovery',type=Path)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    if sha256(a.inventory)!=a.inventory_sha:raise ValueError('Local release inventory SHA differs')
    inv=json.loads(a.inventory.read_text());compat=json.loads(a.compatibility.read_text())
    if inv['status']!='COMPLETE_12_ASSOCIATION_TRAIN_FULL_CAUSAL_AUDITS' or inv['pending_scenes']:
        raise ValueError('Local release not complete')
    if sha256(a.assets)!=compat['compact_keys_sha256']:raise ValueError('Precommitted GT-free keys changed')
    assets=list(csv.DictReader(a.assets.open()));expected_count=compat['gt_free_available_physical_queries']
    assert len(assets)==expected_count
    if len({(r['scene'],r['image_id'],r['row_id']) for r in assets})!=expected_count:raise ValueError('Duplicate frozen query')
    recovery=json.loads(a.recovery.read_text()) if a.recovery else None
    if recovery:
        if sha256(a.recovery)!=sha256(a.compatibility) or recovery['p0_audit_sha256']!=sha256(a.p0_audit):
            raise ValueError('Asset recovery provenance differs')
    sources=json.loads(a.p0_audit.read_text())['verified_sources'];rows=[];source_binding={};summary={}
    for path,old_binding in sources.items():
        folder=Path(path).parent;bundle=load_trackvggt_bundle(folder,expected_manifest_sha256=old_binding['manifest_sha256'])
        if recovery:
            from recover_cv_pil_existing_assets import load_augmented
            bundle,_=load_augmented(folder,old_binding,recovery['geo_provenance'])
        if bundle['artifact_sha256']!=old_binding['events_sha256']:raise ValueError('Old feature cache SHA differs')
        scene=bundle['manifest']['video_id'];entry=next(x for x in inv['scenes'] if x['scene']==scene)
        if entry['role']!=('validation_diagnostic' if scene==12 else 'fit'):raise ValueError('Source split differs')
        event_path=Path(entry['events']);native_folder=event_path.parent;mp=native_folder/'manifest.json'
        expected=inv['source_artifacts'][str(mp)]
        if sha256(mp)!=expected:raise ValueError('Source manifest SHA differs')
        m=json.loads(mp.read_text())
        for flag in ('full_video','continuous_from_first_frame','prefix40_raw_prediction_exact','raw_GT_labeling_after_full_inference_and_OFF_parity'):
            if m.get(flag) is not True:raise ValueError('Source causal/parity gate failed: '+flag)
        if m['status']!='COMPLETE_FULL_CAUSAL_AUDIT' or m['GT_features_to_tracker'] or m['future_query_features'] or m['vggt_calls'] or m['dinov2_calls']:
            raise ValueError('Source native-only frozen scope differs')
        if m['seed']!=20261014 or m['provenance']['checkpoint_sha256']!=FROZEN_STAGE2_SHA256 or m['provenance']['config_sha256']!=FROZEN_CONFIG_SHA256:
            raise ValueError('Source frozen GMT checkpoint/config/seed differs')
        prefix=m['existing_prefix_parity']
        if not prefix['exact'] or prefix['groups']!=400 or Path(prefix['reference']).parent!=folder or sha256(prefix['reference'])!=prefix['sha256']:
            raise ValueError('New/old full400 raw prediction parity not certified')
        for name,value in m['artifacts'].items():
            if inv['source_artifacts'][str(native_folder/name)]!=value or sha256(native_folder/name)!=value:
                raise ValueError('Sealed source artifact SHA differs')
        old_trace=folder/'actual_decisions.jsonl.gz'
        if sha256(old_trace)!=bundle['manifest']['artifacts'][old_trace.name]:raise ValueError('Old solver trace SHA differs')
        decisions={}
        for record in records(old_trace):
            if record['stage']=='run_memory_tracker':continue
            for q,accepted,final in zip(record['queries'],record['accepted_track_ids'],record['final_track_ids']):
                key=source_key(q)
                if key in decisions:raise ValueError('Old primary assigned twice')
                decisions[key]=(accepted,final)
        wanted={(int(r['image_id']),int(r['row_id'])):r for r in assets if int(r['scene'])==scene}
        seen=set()
        for event in records(event_path):
            key=source_key(event['key'][1:])
            if key not in wanted:continue
            if key in seen:raise ValueError('Source index contains a repeated physical query')
            seen.add(key);rows.append(joined_row(bundle,wanted[key],event,decisions[key]))
        if seen!=set(wanted):raise ValueError('Frozen actor query missing from completed source index')
        group=[r for r in rows if r['scene']==scene]
        summary[str(scene)]=dict(queries=len(group),known_primary_wrong_or_missed=sum(r['known_primary_wrong_or_missed'] for r in group),
            strict_past_cross_view_reachable_primary_errors=sum(r['reachable_primary_error'] for r in group),
            certified_accepted_final_wrong=sum(r['certified_accepted_final_wrong'] for r in group),
            wrong_person_groups=len({r['GT_person_OFFLINE_ONLY'] for r in group if r['known_primary_wrong_or_missed']}),
            full_scene_producer_counts=json.loads((native_folder/'summary.json').read_text())['counts'])
        source_binding[str(scene)]=dict(manifest=str(mp),manifest_sha256=expected,event_sha256=sha256(event_path),old_events_sha256=old_binding['events_sha256'],
            old_full400_prediction_parity=True,all_frozen_actor_top8_and_actual_commit_IDs_equal=True)
    if sha256(a.inventory)!=a.inventory_sha or sha256(a.assets)!=compat['compact_keys_sha256']:raise ValueError('Input changed during read-only join')
    a.out.mkdir(parents=True,exist_ok=False)
    csv_path=a.out/'JOINED_QUERY_STATUS_OFFLINE_ONLY.csv'
    with csv_path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    result=dict(status='PASS_LOCAL_SHA_SEALED_HANDOFF_AND_FROZEN_ASSET_JOIN',
        written_utc=datetime.now(timezone.utc).isoformat(),inventory_sha256=a.inventory_sha,
        local_release_accepted_after_full_SHA_validation=True,remote_git_synchronization='NOT_CLAIMED',
        selection_frozen_before_reading_labels=True,selection_rule=f'All{expected_count} previously committed feature-available physical queries; no GT selection.',
        asset_keys_sha256=sha256(a.assets),joined_query_status_sha256=sha256(csv_path),source_binding=source_binding,by_scene=summary,
        development_queries=summary['12']['queries'],development_reachable_primary_errors=summary['12']['strict_past_cross_view_reachable_primary_errors'],
        development_certified_accepted_final_wrong=summary['12']['certified_accepted_final_wrong'],
        new_model_training=False,error_mining_performed=False,gpu_wall_hours=0,
        scientific_success=False,full_online_HOTA=None,
        interpretation='Source labels/outcomes consumed without raw GT recomputation; no new candidate or feature. A known rejected primary does not certify final Bank/birth error.')
    (a.out/'JOIN_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
