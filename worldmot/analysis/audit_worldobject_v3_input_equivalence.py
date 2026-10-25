#!/usr/bin/env python3
"""Compare actual old-support/ROI and corrected-geometry inputs, without a model."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys
import time
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--root",required=True,type=Path)
    parser.add_argument("--split",choices=["train","test"],required=True)
    parser.add_argument("--output",required=True,type=Path);args=parser.parse_args()
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from worldmot.data.visiontrack import VisionTrack
    from worldmot.data.worldobject_cache import WorldObjectCache
    from worldmot.data.worldobject_observations import replay_observations
    from worldmot.models.gmt_detection_cache import ImmutableGMTDetectionCache
    from worldmot.training.world_object_identity_targets import match_train_identities
    from worldmot.runtime.experiment_authorization import source_fingerprint
    root=args.root;split=args.split
    dataset=Path('/data/DATASETS/TRACKING/JDE/VisionTrack') if split=='train' else root/'metadata_only_dataset'
    ds=VisionTrack(dataset,split=split,load_annotations=split=='train')
    legacy=root.parent/'worldobject_fast_proof_20261025'/('cache_'+split)
    raw_path=root.parent/'gmt_detector_comparison_20261023'/('raw_'+split+'_v1')
    raw_a=ImmutableGMTDetectionCache(raw_path);raw_b=ImmutableGMTDetectionCache(raw_path)
    records=[];context={};started=time.time()
    try:
        for sid in ds.scene_ids():
            a=WorldObjectCache(legacy,sid,raw_cache=raw_a,require_features=True,require_geometry=False)
            b=WorldObjectCache(root/('cache_'+split+'_v2'),sid,raw_cache=raw_b,require_features=True,require_geometry=True)
            left,right=iter(a.frames()),iter(b.frames());count=0;exact=0;maxima=np.zeros(3);hashes=[hashlib.sha256(),hashlib.sha256()]
            missing_explicit_RGB_SHA=[0,0]
            pairs={p.frame_id:p for p in ds.synchronized_pairs(sid)}
            for timestamp,pair in pairs.items():
                context={'scene':sid,'timestamp':timestamp}
                fa,fb=next(left),next(right)
                if fa['timestamp']!=timestamp or fb['timestamp']!=timestamp:raise ValueError('TIMESTAMP_MISMATCH')
                for key in ('image_ids',):
                    va={str(k):v for k,v in fa['record'][key].items()};vb={str(k):v for k,v in fb['record'][key].items()}
                    if va!=vb:raise ValueError('RGB_OR_CAMERA_METADATA_MISMATCH')
                    if va!={str(view):record.image_id for view,record in pair.images.items()}:raise ValueError('CANONICAL_IMAGE_ID_MISMATCH')
                canonical_RGB=[raw_a.last_diagnostics['rgb_sha256'],raw_b.last_diagnostics['rgb_sha256']]
                if canonical_RGB[0]!=canonical_RGB[1]:raise ValueError('CANONICAL_APPEARANCE_RGB_SHA_MISMATCH')
                for index,frame in enumerate((fa,fb)):
                    explicit=frame['record'].get('rgb_sha256')
                    missing_explicit_RGB_SHA[index]+=int(explicit is None)
                    if explicit is not None and explicit!=canonical_RGB[index]:raise ValueError('EXPLICIT_SPARSE_RGB_SHA_MISMATCH')
                aa={(x['view'],x['detection_id']):x for x in fa['rows']};bb={(x['view'],x['detection_id']):x for x in fb['rows']}
                if len(aa)!=len(fa['rows']) or len(bb)!=len(fb['rows']) or set(aa)!=set(bb):raise ValueError('CANDIDATE_KEY_MISMATCH')
                for key in sorted(aa):
                    x,y=aa[key],bb[key]
                    delta=np.array([np.max(np.abs(x['bbox']-y['bbox'])),abs(x['score']-y['score']),
                                    np.max(np.abs(x['roi_feature']-y['roi_feature']))])
                    maxima=np.maximum(maxima,delta)
                    if not np.isfinite(delta).all() or np.any(delta>np.array([1e-5,1e-6,1e-6])):raise ValueError('BBOX_SCORE_OR_ROI_TOLERANCE_MISMATCH')
                    if x['feature_valid']!=y['feature_valid']:raise ValueError('ROI_VALIDITY_MASK_MISMATCH')
                    blobs=[]
                    for row in (x,y):
                        blobs.append(struct.pack('<4q',sid,timestamp,*key)+np.asarray(row['bbox'],dtype='<f8').tobytes()
                            +np.asarray([row['score']],dtype='<f8').tobytes()+np.asarray(row['roi_feature'],dtype='<f4').tobytes())
                    exact+=int(blobs[0]==blobs[1]);count+=1
                    for digest,blob in zip(hashes,blobs):digest.update(blob)
                if split=='train':
                    labels_a,audit_a=match_train_identities(replay_observations(fa),ds.ground_truth_pair(pair))
                    labels_b,audit_b=match_train_identities(replay_observations(fb),ds.ground_truth_pair(pair))
                    label_a={o.key:label for o,label in zip(replay_observations(fa),labels_a)}
                    label_b={o.key:label for o,label in zip(replay_observations(fb),labels_b)}
                    if label_a!=label_b or audit_a!=audit_b:raise ValueError('TRAIN_CURRENT_LABEL_OR_SUPERVISION_MASK_MISMATCH')
            if next(left,None) is not None or next(right,None) is not None:raise ValueError('EXTRA_CACHE_TIMESTAMPS')
            records.append({'scene':sid,'timestamps':len(pairs),'candidates':count,'bitwise_equal_candidates':exact,
                'max_bbox_score_ROI_abs_difference':maxima.tolist(),'old_route_candidate_SHA256':hashes[0].hexdigest(),
                'new_route_candidate_SHA256':hashes[1].hexdigest(),'old_manifest_sha256':sha(a.path/'manifest.json'),
                'new_manifest_sha256':sha(b.path/'manifest.json'),
                'frames_without_explicit_sparse_RGB_SHA_old_new':missing_explicit_RGB_SHA,
                'canonical_appearance_RGB_SHA_equal':True})
            print(json.dumps({'input_equivalence_scene_complete':sid,'split':split,'candidates':count}),flush=True)
        status='COMPLETE_INPUT_EQUIVALENCE_VERIFIED';error=None
    except Exception as exc:
        status='P0_INPUT_AUDIT_INCOMPLETE_OR_MISMATCH';error={'type':type(exc).__name__,'message':str(exc),'context':context}
    result={'status':status,'split':split,'source_sha256':source_fingerprint(),'script_sha256':sha(Path(__file__)),
        'observed_at':time.time(),'elapsed_s':time.time()-started,'scene_count':len(records),
        'timestamps':sum(x['timestamps'] for x in records),'candidates':sum(x['candidates'] for x in records),
        'records':records,'error':error,'bbox_score_ROI_abs_tolerances':[1e-5,1e-6,1e-6],
        'detector_cache_lock_sha256':raw_a.manifest()['cache_lock_sha256'],
        'annotation_or_metadata_sha256':sha(ds.annotation_path),'TRAIN_current_supervision_labels_and_masks_compared':split=='train',
        'TEST_GT_read':False,'geometry_expected_to_differ':True,
        'RGB_provenance_note':'Legacy sparse containers may omit RGB hashes. Both appearance routes are bound by exact image IDs and the same immutable GMT candidate/ROI cache RGB hashes; any explicit sparse RGB hashes must also match. Legacy 3D support is not an active input of Full No-Geometry.',
        'history_target_note':'Current clean labels/masks must match. Historical targets may differ as a consequence of each model predicted history; they are not an input-cache difference.',
        'new_training_started':False,'R2_inputs_or_assignments_modified':False}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    if error:raise RuntimeError(json.dumps(error))
    print(json.dumps({'status':status,'output':str(args.output)}),flush=True)


if __name__=='__main__':main()
