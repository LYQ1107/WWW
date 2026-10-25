#!/usr/bin/env python3
"""Disambiguate recovered-view ID breaks from pre-existing other-view splits.

Pure posthoc TRAIN observer correction. Original traces, R2 predictions and
source stay intact. Exact counts require every recovery example to be retained.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from summarize_worldobject_v3 import read,sha,write_json


def objects(row):
    return {(int(cam),x['detection_id']):x for cam,items in row['tracks_2d_by_camera'].items() for x in items}


def classify_next_view_event(recovered_cameras,old_id,next_by_camera,new_birth_cameras,geometry_rejected_cameras):
    observed=set(recovered_cameras)&set(next_by_camera)
    changed={cam for cam in observed if next_by_camera[cam]!=old_id}
    return {'next_recovered_view_observed':bool(observed),
        'next_recovered_view_ID_change':bool(changed),
        'next_recovered_view_kept_old_ID':any(next_by_camera[cam]==old_id for cam in observed),
        'next_new_ID_on_recovered_view':bool(changed&set(new_birth_cameras)),
        'next_recovered_view_geometry_rejected':bool(changed&set(geometry_rejected_cameras)),
        'other_view_new_ID_only':bool(set(new_birth_cameras)-set(recovered_cameras)) and not bool(changed&set(new_birth_cameras))}


def refine_report(path,ds):
    from worldmot.training.world_object_identity_targets import match_train_identities
    original=read(path);counts=Counter();events=[];scene_counts={}
    for sid,scene in original['per_scene'].items():
        examples=scene['analysis']['examples'].get('CORRECT_SPATIAL_RECOVERY',[])
        expected=scene['analysis']['counts'].get('correct_selected_spatial_conflict_visual_recoveries',0)
        if len(examples)!=expected:raise ValueError('RECOVERY_EXAMPLE_TRUNCATED_CANNOT_CLAIM_EXACT_FREQUENCY')
        c=Counter()
        if examples:
            prediction=path.parent/f'scene_{int(sid):02d}/tracks.jsonl'
            if sha(prediction)!=read(prediction.with_name('manifest.json'))['tracks_sha256']:raise ValueError('ORIGINAL_TRAIN_PREDICTIONS_CHANGED')
            pairs=list(ds.synchronized_pairs(int(sid)));lookup={pair.frame_id:pair for pair in pairs}
            next_times={x.frame_id:(pairs[j+1].frame_id if j+1<len(pairs) else None) for j,x in enumerate(pairs)}
            wanted={e['timestamp'] for e in examples}|{next_times[e['timestamp']] for e in examples}
            rows={}
            for line in prediction.open():
                row=json.loads(line)
                if row['timestamp'] in wanted:rows[row['timestamp']]=row
            breaks={e['recovery_previous_frame']['timestamp']:e for e in scene['analysis']['examples'].get('RECOVERY_NEXT_FRAME_ID_BREAK',[])}
            for old in examples:
                t=old['timestamp'];before=objects(rows[t]);keys=sorted(before)
                recovered_cameras={keys[j][0] for j in old['indices']}
                c['correct_recovery_events']+=1
                c['single_view_correct_recovery_events' if len(recovered_cameras)==1 else 'two_view_correct_recovery_events']+=1
                c['stale_center_retained_missed_reset_0']+=int(old['center_unchanged'] and old['missed_after']==0)
                nt=next_times[t]
                if nt is None:c['no_following_frame']+=1;continue
                after=objects(rows[nt]);next_keys=sorted(after)
                obs=[SimpleNamespace(camera_id=k[0],key=k,bbox_xywh=np.asarray(after[k]['bbox_xywh'])) for k in next_keys]
                labels,_=match_train_identities(obs,ds.ground_truth_pair(lookup[nt]))
                current_indices=[j for j,label in enumerate(labels) if label==old['label']]
                by_camera={next_keys[j][0]:after[next_keys[j]]['global_id'] for j in current_indices}
                detail=breaks.get(t,{})
                new_cameras={next_keys[j][0] for j in detail.get('new_birth_indices',[])}
                rejected_cameras={next_keys[j][0] for j in detail.get('geometry_rejected_indices',[])}
                classified=classify_next_view_event(recovered_cameras,old['gid'],by_camera,new_cameras,rejected_cameras)
                c.update({k:int(v) for k,v in classified.items()})
                c['new_ID_on_recovered_view_with_geometry_rejection_and_stale_center']+=int(
                    bool(set(recovered_cameras)&new_cameras&rejected_cameras) and old['center_unchanged'])
                events.append({'scene':int(sid),'recovery_timestamp':t,'next_timestamp':nt,'GT_label':old['label'],
                    'recovered_ID':old['gid'],'recovered_cameras':sorted(recovered_cameras),'next_clean_ID_by_camera':by_camera,
                    'new_birth_cameras':sorted(new_cameras),'geometry_rejected_cameras':sorted(rejected_cameras),
                    'classification':classified,'original_recovery':old,'raw_following_event':detail})
        counts.update(c);scene_counts[sid]=dict(c)
    return {'method':original['method'],'selection_epoch':original['selection_epoch'],
        'checkpoint_sha256':original['checkpoint_sha256'],'timestamps':original['timestamps'],
        'scenes':original['scenes'],'counts':dict(counts),'per_scene':scene_counts,'events':events,
        'source_original_report':str(path),'source_original_report_sha256':sha(path),
        'original_diagnostics_exactly_reproduced':original['original_epoch_prediction_diagnostics_exactly_reproduced']}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    from worldmot.data.visiontrack import VisionTrack
    ds=VisionTrack('/data/DATASETS/TRACKING/JDE/VisionTrack',split='train',load_annotations=True)
    rows=[]
    for method in ('FULL_WORLD_OBJECT_V3','HEAD_ONLY_NO_GEOMETRY_V3'):
        for epoch in range(1,5):
            path=a.root/'diagnostics/retained_TRAIN_VAL'/method/f'epoch_{epoch:02d}/attribution.json'
            if path.is_file():rows.append(refine_report(path,ds))
    report={'schema':'worldobject_v3_recovered_view_frequency_refinement_v2',
        'status':'COMPLETE_ALL8_AVAILABLE_REPLAYS' if len(rows)==8 else 'PARTIAL_REAL_REPLAY_COVERAGE',
        'completed_replays':len(rows),'expected_replays':8,'replays':rows,
        'R2_source_or_predictions_modified':False,'TEST_read':False,'optimizer_used':False,
        'definition_correction':'Only follow the camera(s) whose detections were assigned to the recovered historical ID at t. A different ID already present in another camera is not a new failure of that recovery.',
        'stronger_two_view_criterion':'Single-view historical recovery is distinguished from shared-ID two-view recovery. Do not report the former as already successful global two-view identity fusion.',
        'original_any_view_counts_retained_for_audit':True,'analysis_sha256':sha(__file__)}
    write_json(a.output,report)
    for x in rows:print(json.dumps({'method':x['method'],'epoch':x['selection_epoch'],'corrected_counts':x['counts']}),flush=True)


if __name__=='__main__':main()
