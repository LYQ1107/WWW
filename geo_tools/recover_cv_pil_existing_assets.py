"""GT-free exact physical-key reuse of already frozen full1152 GMT vectors.

No encoder, images, GT/teacher arrays or model training. Every shared vector
must be bitwise identical; cached native candidates/scores stay unchanged.
"""
import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np
import torch

from cv_pil_native_handoff import sha256,source_key,load_trackvggt_bundle
from probe_cv_pil_miner_compatibility import available_queries,export_compact_keys


def detector_key(image_id,box,xyxy=False):
    x=np.asarray(box,dtype=np.float32).copy()
    if x.shape!=(4,) or not np.isfinite(x).all():raise ValueError('Invalid physical box')
    if xyxy:x[2:]-=x[:2]
    if (x[2:]<=0).any():raise ValueError('Degenerate physical box')
    return int(image_id),tuple(map(float,x))


def augment_observations(observations,detectors,metadata,features):
    if features.shape!=(len(metadata),1152) or not np.isfinite(features).all():
        raise ValueError('Recovery must use complete finite1152 native vectors')
    lookup={}
    for row in detectors:
        k=detector_key(row['source'][0],row['detection']['bbox'])
        if k in lookup:raise ValueError('Ambiguous duplicate detector box')
        lookup[k]=source_key(row['source'])
    merged=dict(observations);common=added=unmatched=0;seen=set()
    for m,x in zip(metadata,features):
        dkey=detector_key(m['image_id'],m['bbox_xyxy'],xyxy=True)
        if dkey not in lookup:unmatched+=1;continue
        key=lookup[dkey]
        if key in seen:raise ValueError('Repeated recovery physical key')
        seen.add(key);feature=torch.from_numpy(x.copy())
        if feature.float().norm()==0:raise ValueError('Zero recovered feature')
        if key in observations:
            old=observations[key]
            if old['frame']!=m['raw_frame'] or old['camera']!=m['camera'] or not torch.equal(old['feature'],feature):
                raise ValueError('Existing native feature/time/camera differs; refuse mixed provenance')
            common+=1
        else:
            merged[key]=dict(source=list(key),image_id=key[0],frame=int(m['raw_frame']),
                camera=int(m['camera']),feature=feature,box=m['bbox_xyxy'])
            added+=1
    if not common:raise ValueError('No independently overlapping native feature proof')
    return merged,dict(common_exact_features=common,added_native_vectors=added,unmatched_boxes=unmatched)


def load_augmented(folder,binding,geo_provenance):
    bundle=load_trackvggt_bundle(folder,expected_manifest_sha256=binding['manifest_sha256'])
    if bundle['artifact_sha256']!=binding['events_sha256']:raise ValueError('Original cache SHA differs')
    scene=bundle['manifest']['video_id']
    if str(scene) not in geo_provenance:return bundle,dict(added_native_vectors=0)
    source=geo_provenance[str(scene)];mp=Path(source['manifest']);geo=mp.parent
    if sha256(mp)!=source['manifest_sha256']:raise ValueError('Frozen legacy GMT source manifest differs')
    m=json.loads(mp.read_text());p=m['provenance'];old=bundle['manifest']['provenance']
    if not m['neutral_prefix40_exact'] or m['encoder_failures'] or p['seed']!=20261014:
        raise ValueError('Legacy source causal/neutrality gate differs')
    for k in ['checkpoint_sha256','config_sha256','annotation_sha256']:
        if p[k]!=old[k]:raise ValueError('Legacy native source '+k+' differs')
    for name in ['observations.json','targets.npz']:
        if sha256(geo/name)!=m['artifacts'][name]:raise ValueError('Frozen legacy feature archive differs')
    trace=folder/'detector_provenance.jsonl.gz'
    if sha256(trace)!=bundle['manifest']['artifacts'][trace.name]:raise ValueError('Original detector physical-key trace differs')
    with gzip.open(trace,'rt') as f:detectors=[json.loads(line) for line in f]
    meta=json.loads((geo/'observations.json').read_text())
    # Deliberately load only native GMT, never labels, DINO or VGGT arrays.
    with np.load(geo/'targets.npz',allow_pickle=False) as z:features=z['gmt'].copy()
    obs,counts=augment_observations(bundle['payload']['observations'],detectors,meta,features)
    body=dict(bundle['payload']);body['observations']=obs;bundle=dict(bundle,payload=body)
    counts.update(legacy_manifest_sha256=sha256(mp),legacy_target_sha256=sha256(geo/'targets.npz'),
                  physical_detector_trace_sha256=sha256(trace))
    return bundle,counts


def main():
    p=argparse.ArgumentParser();p.add_argument('--p0-audit',type=Path,required=True)
    p.add_argument('--v1-native',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    audit=json.loads(a.p0_audit.read_text());provenance=json.loads(a.v1_native.read_text())['provenance']
    rows=[];counts={}
    for name,binding in audit['verified_sources'].items():
        bundle,recovery=load_augmented(Path(name).parent,binding,provenance)
        found,availability=available_queries(bundle);rows+=found
        counts[str(bundle['manifest']['video_id'])]=dict(recovery=recovery,availability=availability)
    a.out.mkdir(parents=True,exist_ok=False);full=a.out/'FULL_GT_FREE_AVAILABILITY.csv'
    with full.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    compact=a.out/'ASSET_AVAILABILITY_QUERY_KEYS.csv';export_compact_keys(full,compact)
    result=dict(status='PASS_GT_FREE_EXISTING_NATIVE_ASSET_RECOVERY',
        p0_audit_sha256=sha256(a.p0_audit),v1_native_result_sha256=sha256(a.v1_native),
        geo_provenance=provenance,by_scene=counts,gt_free_available_physical_queries=len(rows),
        compact_keys_sha256=sha256(compact),full_availability_sha256=sha256(full),
        native_candidates_and_scores_unchanged=True,GT_and_teacher_arrays_read=False,
        new_model_training=False,new_feature_extraction=False,gpu_wall_hours=0,
        scientific_success=False,source_sha256=sha256(__file__))
    (a.out/'RECOVERY.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
