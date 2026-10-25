#!/usr/bin/env python3
"""Read-only existing TEST outputs. Never runs GMT or a V4 online TEST tracker."""
import argparse
from collections import Counter,defaultdict
from contextlib import ExitStack
import json
from pathlib import Path
import time
import numpy as np
from worldmot.data.worldobject_cache import sha256
from worldmot.data.visiontrack import VisionTrack
from worldmot.evaluation.synchronous_multiview_identity import match_boxes,events,update_counts,summarize
from worldobject_v4_audit_common import verify_immutable_annotations,EXPECTED_TEST_ANNOTATIONS_SHA

METHODS=('GMT_NATIVE','FULL_WORLD_OBJECT_V3','FULL_NO_GEOMETRY_V3','HEAD_ONLY_NO_GEOMETRY_V3')
LEGAL={1,2,3,4,6,7,8,9,10,11,15,16,18,19}


def camera_axis_angle(record):
    cameras=record.get('camera_models',{})
    if not all(str(v) in cameras for v in (1,2)):return None
    axes=[]
    for v in (1,2):
        matrix=np.asarray(cameras[str(v)].get('world_from_camera',[]),float)
        if matrix.shape!=(4,4) or not np.isfinite(matrix).all():return None
        rotation=matrix[:3,:3]
        if not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-3) or np.linalg.det(rotation)<=0:return None
        axes.append(rotation[:,2]/np.linalg.norm(rotation[:,2]))
    return float(np.degrees(np.arccos(np.clip(np.dot(*axes),-1,1))))


def flatten(row):
    if 'tracks_2d_by_camera' not in row:raise ValueError('FINAL_TRACK_OUTPUT_REQUIRED_NOT_DETECTION_CACHE')
    return [dict(x,camera_id=int(cam)) for cam,items in row['tracks_2d_by_camera'].items()
            for x in items if x.get('visible',True) and x.get('evaluation_eligible',True)]


def verify_roots(roots,evidence,r2_coverage):
    """Finish all immutability/coverage checks before opening any GT."""
    native=next(x for x in evidence['routes'] if x['method']=='gmt_native')
    expected={x['scene']:x['sha256'] for x in native['manifests']};proofs={}
    if len(expected)!=22:raise ValueError('ORIGINAL_GMT_ALL22_MANIFEST_PROOFS_REQUIRED')
    for method,root in roots.items():
        items=[]
        for sid in range(1,23):
            folder=root/f'scene_{sid:02d}';manifest=folder/'manifest.json';m=json.loads(manifest.read_text())
            if method=='GMT_NATIVE' and sha256(manifest)!=expected[sid]:
                raise ValueError('ORIGINAL_GMT_MANIFEST_CHANGED')
            if method!='GMT_NATIVE':
                entries=r2_coverage['methods'][method]['scenes']
                expected_manifest=next(x['manifest_sha256'] for x in entries if x['scene']==sid)
                if sha256(manifest)!=expected_manifest:raise ValueError('PUBLISHED_R2_MANIFEST_CHANGED')
            count=m.get('timestamps',m.get('frames'))
            track_hash=m.get('tracks_sha256',m.get('prediction_sha256'))
            if m.get('status')!='COMPLETE' or m.get('GT_in_inference') is not False or sha256(folder/'tracks.jsonl')!=track_hash:
                raise ValueError('COMPLETE_IMMUTABLE_GT_FREE_PREDICTIONS_REQUIRED')
            if method=='GMT_NATIVE' and not m.get('full_video'):raise ValueError('GMT_FULL_VIDEO_REQUIRED')
            items.append(dict(scene=sid,timestamps=count,tracks_sha256=track_hash,manifest_sha256=sha256(manifest)))
        if sum(x['timestamps'] for x in items)!=29019:raise ValueError('ALL22_29019_SCOPE_REQUIRED')
        proofs[method]=items
    return proofs


def main():
    p=argparse.ArgumentParser();p.add_argument('--r2-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--gmt-root',type=Path)
    p.add_argument('--raw-cache',type=Path);p.add_argument('--dataset',default='/data/DATASETS/TRACKING/JDE/VisionTrack')
    p.add_argument('--with-current-only',action='store_true',help='registered history-free A-only mechanism diagnostic, never V4 tracking')
    p.add_argument('--device',default='cpu')
    a=p.parse_args();started=time.perf_counter();repo=Path(__file__).resolve().parents[2]
    evidence_path=repo/'worldmot/results/gmt_vggt_official_evidence_20261024.json'
    evidence=json.loads(evidence_path.read_text());native=next(x for x in evidence['routes'] if x['method']=='gmt_native')
    roots={m:a.r2_root/'predictions'/m for m in METHODS if m!='GMT_NATIVE'}
    roots['GMT_NATIVE']=a.gmt_root or Path(native['prediction_root'])
    r2_proof_path=repo/'worldmot/results/worldobject_v3_fulltest_coverage.json'
    proofs=verify_roots(roots,evidence,json.loads(r2_proof_path.read_text()))
    from worldmot.models.gmt_detection_cache import ImmutableGMTDetectionCache
    raw=ImmutableGMTDetectionCache(a.raw_cache or a.r2_root.parent/'gmt_detector_comparison_20261023/raw_test_v1')
    current_only={};no_history_provenance={};current_counts=defaultdict(Counter);angle_coverage=Counter()
    if a.with_current_only:
        import torch
        from dataclasses import replace
        from worldmot.models.world_object_v4_checkpoint import load_frozen_r2_head
        from worldmot.runtime.world_object_tracking_core import WorldObjectTrackingCore
        from worldmot.tracking.world_object_config import WorldObjectConfig
        torch.set_num_threads(1)
        full_model,full_proof=load_frozen_r2_head(a.r2_root/'train/FULL_WORLD_OBJECT_V3/train_gate.json',raw.lock,a.device)
        ng_model,ng_proof=load_frozen_r2_head(a.r2_root/'train/FULL_NO_GEOMETRY_V3/train_gate.json',raw.lock,a.device,method='FULL_NO_GEOMETRY_V3')
        full=WorldObjectConfig(raw.feature_dim,enable_temporal_geometry=False,enable_spatial_state_update=False)
        current_only={'H0_CURRENT_FULL':WorldObjectTrackingCore(full,full_model,a.device),
            'H0_SAME_FULL_HEAD_NO_GEOMETRY_INPUT':WorldObjectTrackingCore(replace(full,enable_sync_geometry=False,enable_head_geometry=False),full_model,a.device),
            'H0_TRAINED_NO_GEOMETRY_REFERENCE':WorldObjectTrackingCore(WorldObjectConfig(raw.feature_dim,'FULL_NO_GEOMETRY_V3'),ng_model,a.device)}
        no_history_provenance={'full':full_proof,'separately_trained_no_geometry':ng_proof,
            'all_histories_empty_each_timestamp':True,'threshold_selection_on_TEST':False,'not_an_online_MOT_experiment':True}
    ds=VisionTrack(a.dataset,split='test',load_annotations=True)
    annotation_proof=verify_immutable_annotations(ds,EXPECTED_TEST_ANNOTATIONS_SHA)
    totals=defaultdict(Counter);paired=defaultdict(Counter);strata=defaultdict(Counter)
    scenes={};cases=[];case_counts=Counter();match_audit=defaultdict(Counter);ambiguity_masks=[]
    for sid in range(1,23):
        scope='legal14' if sid in LEGAL else 'blocked8_clean_subset_diagnostic'
        sc=defaultdict(Counter);sp=defaultdict(Counter);mask_counts=defaultdict(Counter)
        geometry_frames=None
        if current_only:
            from worldmot.data.worldobject_cache import WorldObjectCache
            geometry_frames=iter(WorldObjectCache(a.r2_root/'cache_test_v2',sid,raw_cache=raw,require_geometry=True).frames())
        with ExitStack() as stack:
            streams={m:stack.enter_context((roots[m]/f'scene_{sid:02d}/tracks.jsonl').open()) for m in METHODS}
            for pair in ds.synchronized_pairs(sid):
                rows={m:json.loads(next(f)) for m,f in streams.items()}
                density_count=len(flatten(rows[METHODS[1]]))
                if any(r['timestamp']!=pair.frame_id for r in rows.values()):raise ValueError('TIMESTAMP_DIFFERENCE')
                if rows['GMT_NATIVE'].get('native_GMT') is not True:raise ValueError('NATIVE_GMT_FINAL_ID_ROW_REQUIRED')
                matched={};full_decisions={(x['camera_id'],x['detection_id']):x for x in rows[METHODS[1]]['decisions']}
                raw.get(sid,pair.frame_id,image_ids={v:r.image_id for v,r in pair.images.items()})
                features=raw.last_roi_features
                angle=None;current_results={}
                if current_only:
                    from worldmot.data.worldobject_observations import replay_observations
                    geometry_frame=next(geometry_frames)
                    if geometry_frame['timestamp']!=pair.frame_id:raise ValueError('CURRENT_ONLY_CACHE_TIMESTAMP_DIFFERENCE')
                    angle=camera_axis_angle(geometry_frame['record'])
                    angle_coverage[scope+'/camera_axis_available' if angle is not None else scope+'/camera_axis_unavailable']+=1
                    observations=replay_observations(geometry_frame)
                    for key,core in current_only.items():
                        core.reset(sid);current_results[key]=core.step(observations,pair.frame_id)
                gt=ds.ground_truth_pair(pair)
                for key,row in current_results.items():
                    signature=lambda value:[(x['camera_id'],x.get('detection_id'),tuple(x['bbox_xywh']),x['score']) for x in flatten(value)]
                    if signature(row)!=signature(rows[METHODS[1]]):raise ValueError('HISTORY_FREE_CURRENT_CANDIDATE_DIFFERENCE')
                    labels,_=match_boxes(flatten(row),gt)
                    for _,same,shared in events(labels):update_counts(current_counts[f'{scope}/{key}'],same,shared)
                for m,row in rows.items():
                    matched[m],audit=match_boxes(flatten(row),gt)
                    mask_counts[m].update(audit['counts'])
                    if audit['ambiguity_mask_indices']:
                        ambiguity_masks.append({'scene':sid,'timestamp':pair.frame_id,'method':m,
                            'prediction_count':audit['counts']['detections'],
                            'masked_prediction_indices':audit['ambiguity_mask_indices']})
                    for _,same,shared in events(matched[m]):update_counts(sc[f'Protocol_A/{m}'],same,shared)
                all_common=set.intersection(*(set(v) for v in matched.values()))
                for m in METHODS[1:]:
                    common=set(matched['GMT_NATIVE'])&set(matched[m])
                    for protocol,keys in (('Protocol_B',common),('All4_intersection',all_common)):
                        for key,same,gmtshared in events(matched['GMT_NATIVE'],keys):
                            left,right=key;shared=matched[m][left]['global_id']==matched[m][right]['global_id']
                            update_counts(sc[f'{protocol}/GMT_for_{m}'],same,gmtshared)
                            update_counts(sc[f'{protocol}/{m}'],same,shared)
                            ok,other=gmtshared==same,shared==same
                            contrast='both_correct' if ok and other else 'both_wrong' if not ok and not other else 'V3_corrects_GMT' if other else 'V3_regresses_GMT'
                            kind='same_person' if same else 'different_person'
                            sp[f'{protocol}/{m}/{kind}'][contrast]+=1
                            if protocol!='Protocol_B':continue
                            full_pair=[matched[METHODS[1]].get(k) for k in key]
                            fs=[];eligible=[]
                            for x in full_pair:
                                fk=None if x is None else (x['camera_id'],x.get('detection_id'))
                                fs.append(features.get(fk));eligible.append(full_decisions.get(fk,{}).get('cross_view_fusion_eligible',False))
                            cosine=None
                            if all(x is not None for x in fs):
                                cosine=float(np.dot(fs[0],fs[1])/max(1e-12,np.linalg.norm(fs[0])*np.linalg.norm(fs[1])))
                            groups=['both_geometry_eligible' if all(eligible) else 'one_or_both_geometry_ineligible',
                                    'dense_N_ge20' if density_count>=20 else 'N_lt20']
                            if angle is not None and angle>=60:groups.append('estimated_Omega_camera_axis_angle_ge60deg')
                            if all(len(rows[METHODS[1]]['tracks_2d_by_camera'].get(str(v),[]))>=10 for v in (1,2)):
                                groups.append('both_views_N_ge10')
                            if cosine is not None:
                                if same and cosine<=.55:groups.append('same_person_low_ROI_cosine')
                                if not same and cosine>=.90:groups.append('different_person_high_ROI_cosine')
                            for group in groups:strata[f'{scope}/{m}/{kind}/{group}'][contrast]+=1
                            case_group=(m,scope,kind,contrast)
                            if contrast in ('V3_corrects_GMT','V3_regresses_GMT') and case_counts[case_group]<12:
                                case_counts[case_group]+=1
                                cases.append(dict(scene=sid,timestamp=pair.frame_id,scope=scope,method=m,event=contrast,
                                    same_person=same,gt_event_keys=key,cosine=cosine,strata=groups,
                                    actors=[dict(camera=k[0],gt_identity=k[1],GMT_ID=matched['GMT_NATIVE'][k]['global_id'],
                                        V3_ID=matched[m][k]['global_id'],bbox=matched[m][k]['bbox_xywh']) for k in key]))
            if any(f.readline() for f in streams.values()):raise ValueError('EXTRA_PREDICTION_FRAMES')
        for k,v in sc.items():totals[f'{scope}/{k}'].update(v)
        for k,v in sp.items():paired[f'{scope}/{k}'].update(v)
        for k,v in mask_counts.items():match_audit[f'{scope}/{k}'].update(v)
        scenes[str(sid)]={'scope':scope,'metrics':{k:summarize(v) for k,v in sc.items()},
                          'paired':{k:dict(v) for k,v in sp.items()},'match_audit':{k:dict(v) for k,v in mask_counts.items()}}
        print(json.dumps({'sync_audit_scene_complete':sid,'scope':scope}),flush=True)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    report={'schema':'worldmot_v4_existing_TEST_sync_identity_v1','status':'COMPLETE','official_MOT_metric':False,
        'V4_TEST_tracking_inference':False,'GT_role':'POSTHOC_ONLY','TEST_threshold_selection':False,
        'IoU':.5,'matcher':'maximum cardinality then maximum IoU; ambiguous GT masked',
        'prediction_proofs':proofs,'GMT_manifest_evidence_sha256':sha256(evidence_path),
        'published_R2_coverage_evidence_sha256':sha256(r2_proof_path),
        'original_TEST_annotation_proof':annotation_proof,
        'GT_alignment_ambiguity_masks':{'encoding':'sparse nonzero masks; omitted frame/method entries have all-false masks',
            'index_order':'flatten archived tracks_2d_by_camera in JSON camera/list order; original row hashes are in prediction_proofs',
            'records':ambiguity_masks,'history_free_masks':'same as existing Full, after exact current candidate order/boxes/scores equality checks' if current_only else None},
        'metrics':{k:summarize(v) for k,v in totals.items()},'paired':{k:dict(v) for k,v in paired.items()},
        'strata':{k:dict(v) for k,v in strata.items()},'match_audit':{k:dict(v) for k,v in match_audit.items()},
        'per_scene':scenes,'runtime_wall_s':time.perf_counter()-started,
        'camera_axis_angle':'estimated from the same-pair Omega camera rotation; not calibrated physical camera GT; low ROI cosine is not camera angle',
        'camera_axis_coverage':dict(angle_coverage),
        'no_history_experiment':{'metrics':{k:summarize(v) for k,v in current_counts.items()},'provenance':no_history_provenance,
            'scope':'history cleared each current timestamp, no future or GT in decisions; cannot substitute for online MC-MOT'}}
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    a.output.with_name('v4_crossview_hard_cases.json').write_text(json.dumps({'cases':cases,'scope':'posthoc matched GT events'},indent=2)+'\n')


if __name__=='__main__':main()
