"""Inspect published metadata and existing actor assets; never mine errors.

Availability is frozen without reading event labels. This is not permission to
train a new cohort, nor a replacement for the independent miner's release.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import subprocess

import torch

from cv_pil_native_handoff import (sha256,source_key,get_observation,
                                  load_trackvggt_bundle)


def publication_readiness(inventory):
    reasons=[]
    if inventory.get('status')!='COMPLETE_12_ASSOCIATION_TRAIN_FULL_CAUSAL_AUDITS':
        reasons.append('PUBLISHED_INVENTORY_INCOMPLETE')
    scenes=inventory.get('scenes',[])
    if not any(r.get('scene')==12 and r.get('role')=='validation_diagnostic' for r in scenes):
        reasons.append('FROZEN_DEVELOPMENT_SCENE12_NOT_PUBLISHED')
    if not scenes or any(r.get('features_saved') is not True for r in scenes):
        reasons.append('NEW_NATIVE_FEATURE_ASSETS_NOT_EXPORTED')
    # The independent census deliberately exports ranked IDs and scalar pair
    # evidence, rather than complete frozen candidate score matrices.
    if inventory.get('frozen_candidate_score_assets') is not True:
        reasons.append('NEW_EXACT_CANDIDATE_SCORE_ASSETS_NOT_EXPORTED')
    return dict(trainable_release=False if reasons else None,missing=reasons,
                existing_cache_join_may_partially_supply_assets=True,
                header_check_only=True,native_outcome_payload_certified=False)


def export_compact_keys(full_csv,compact_csv):
    """All rows, a small metadata projection only; never filter by outcomes."""
    fields=['scene','image_id','row_id','frame','camera','call_index','query_row',
            'candidate_count','strictly_past_sources','cross_view_past_sources']
    with Path(full_csv).open() as source,Path(compact_csv).open('w',newline='') as dest:
        writer=csv.DictWriter(dest,fieldnames=fields,lineterminator='\n');writer.writeheader()
        for row in csv.DictReader(source):writer.writerow({key:row[key] for key in fields})


def available_queries(bundle):
    """Use calls/observations only. Never inspect payload['events'] or GT."""
    body=bundle['payload'];observations=body['observations'];scene=bundle['manifest']['video_id']
    rows=[];counts=Counter()
    for ci,call in sorted(body['calls'].items()):
        scores=call['track_scores'];ids=call['track_ids']
        if tuple(scores.shape)!=(len(call['queries']),len(ids)) or not torch.isfinite(scores).all():
            raise ValueError('Original candidate score matrix invalid')
        for qi,source in enumerate(call['queries']):
            counts['cached_call_queries']+=1;key=source_key(source)
            if key not in observations:
                counts['query_feature_not_retained']+=1;continue
            query=get_observation(observations,key);columns=torch.argsort(scores[qi],descending=True,stable=True)[:8].tolist()
            if not columns:
                counts['no_native_candidate']+=1;continue
            candidate_ids=[ids[c] for c in columns]
            past_keys=[];cross=0;missing=False
            for tid in candidate_ids:
                sources=call['histories'][tid]['sources']
                owners={source_key(r['source']) for r in call['references'] if r['track']==tid}
                for src in sources:
                    ref=source_key(src)
                    if ref not in owners:raise ValueError('History owner differs from native assignment')
                    if ref not in observations:
                        missing=True;continue
                    obs=get_observation(observations,ref)
                    if obs['frame']>query['frame']:raise ValueError('Future history in cache')
                    if obs['frame']==query['frame']:continue
                    past_keys.append((tid,*ref));cross+=obs['camera']!=query['camera']
            if missing:
                counts['top8_history_feature_not_retained']+=1;continue
            # All native candidates remain. No same-time or GT-built history.
            encoded=json.dumps(dict(ids=candidate_ids,columns=columns,
                scores=scores[qi,columns].tolist(),past_sources=past_keys),sort_keys=True).encode()
            rows.append(dict(scene=scene,image_id=key[0],row_id=key[1],frame=query['frame'],
                camera=query['camera'],call_index=ci,query_row=qi,candidate_count=len(columns),
                strictly_past_sources=len(past_keys),cross_view_past_sources=cross,
                native_context_sha256=hashlib.sha256(encoded).hexdigest()))
            counts['feature_and_native_top8_available']+=1
            counts['with_any_strict_past_cross_view_history']+=cross>0
    if len({(r['scene'],r['image_id'],r['row_id']) for r in rows})!=len(rows):
        raise ValueError('Repeated physical query in availability pool')
    return rows,dict(counts)


def main():
    p=argparse.ArgumentParser();p.add_argument('--published-inventory',type=Path,required=True)
    p.add_argument('--published-commit',required=True);p.add_argument('--published-path',required=True)
    p.add_argument('--p0-audit',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();root=Path(__file__).resolve().parents[1]
    remote_bytes=subprocess.check_output(['git','show',a.published_commit+':'+a.published_path],cwd=root)
    if remote_bytes!=a.published_inventory.read_bytes():raise ValueError('Local published copy differs from pinned Git object')
    inventory=json.loads(remote_bytes);readiness=publication_readiness(inventory)
    if a.out.exists():raise FileExistsError(a.out)
    a.out.mkdir(parents=True,exist_ok=False)
    source_audit=json.loads(a.p0_audit.read_text());rows=[];sources={}
    for name,binding in source_audit['verified_sources'].items():
        folder=Path(name).parent
        if sha256(Path(name))!=binding['events_sha256']:raise ValueError('Old approved source SHA differs')
        bundle=load_trackvggt_bundle(folder,expected_manifest_sha256=binding['manifest_sha256'])
        found,counts=available_queries(bundle)
        for row in found:row['source_events_sha256']=binding['events_sha256']
        rows+=found;sources[str(folder)]=counts
    csv_path=a.out/'GT_FREE_ASSET_AVAILABILITY.csv'
    with csv_path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    compact_path=a.out/'ASSET_AVAILABILITY_QUERY_KEYS.csv';export_compact_keys(csv_path,compact_path)
    result=dict(status='READONLY_PUBLISHED_HANDOFF_COMPATIBILITY_CHECK',
        published_commit=a.published_commit,published_inventory_path=a.published_path,
        published_inventory_sha256=hashlib.sha256(remote_bytes).hexdigest(),
        published_status=inventory['status'],readiness=readiness,
        source_audit_sha256=sha256(a.p0_audit),availability_sha256=sha256(csv_path),
        compact_keys_sha256=sha256(compact_path),
        sources=sources,gt_free_available_physical_queries=len(rows),
        frozen_cohort_changed=False,new_training_queries_approved=False,new_models_trained=0,
        new_video_or_feature_collection=False,error_mining_performed=False,gpu_wall_hours=0,
        note='Availability is label-blind old-cache metadata only. New event labels and assets require a completed frozen release; no favorable scenes selected.')
    (a.out/'COMPATIBILITY.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
