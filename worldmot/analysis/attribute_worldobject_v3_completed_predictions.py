#!/usr/bin/env python3
"""Read-only identity attribution of ONE already complete immutable TEST run.

No model is executed, no prediction is rewritten, no threshold is selected.
Clean IoU=.5 joins are diagnostics, independent of native official scoring.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np

from summarize_worldobject_v3 import read, sha, write_json
from worldobject_v3_temporal_audit import TemporalErrorAudit, increment_quarter


def identity_graph(person_ids):
    labels = sorted(person_ids)
    gids = sorted({gid for values in person_ids.values() for gid in values})
    from scipy.optimize import linear_sum_assignment
    matrix = np.zeros((len(labels), len(gids)), dtype=np.int64)
    lookup = {gid: j for j, gid in enumerate(gids)}
    for i, label in enumerate(labels):
        for gid, count in person_ids[label].items():
            matrix[i, lookup[gid]] = count
    rows, cols = linear_sum_assignment(-matrix)
    total = int(matrix.sum())
    best = int(matrix[rows, cols].sum())
    return {
        "clean_GT_people": len(labels), "clean_labelled_output_global_ids": len(gids),
        "people_split_across_multiple_global_ids": sum(len(x) > 1 for x in person_ids.values()),
        "extra_global_id_fragments_per_person_summed": sum(max(0, len(x)-1) for x in person_ids.values()),
        "same_person_dominant_fragment_observations": sum(max(x.values()) for x in person_ids.values()),
        "global_one_to_one_matched_clean_observations": best,
        "global_one_to_one_unmatched_clean_observations": total-best,
        "clean_observations": total,
        "diagnostic_clean_join_one_to_one_accuracy": best/max(1, total),
        "official_CVIDF1": False,
        "per_person_fragment_count_histogram": dict(Counter(str(len(x)) for x in person_ids.values())),
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run-root', required=True, type=Path)
    p.add_argument('--method', default='FULL_NO_GEOMETRY_V3')
    p.add_argument('--dataset-root', default='/data/DATASETS/TRACKING/JDE/VisionTrack')
    p.add_argument('--output', required=True, type=Path)
    a=p.parse_args(); root=a.run_root
    from worldmot.runtime.experiment_authorization import require_experiment_authorization, source_fingerprint
    require_experiment_authorization(root/'authorization_v3.json','diagnostics','test',a.method)
    proofs=[]
    for scene in range(1,23):
        folder=root/'predictions'/a.method/f'scene_{scene:02d}'
        m=read(folder/'manifest.json')
        if m['status']!='COMPLETE' or m['method_id']!=a.method or sha(folder/'tracks.jsonl')!=m['tracks_sha256']:
            raise ValueError('ALL22_IMMUTABLE_PREDICTIONS_REQUIRED_BEFORE_GT')
        proofs.append({'scene':scene,'timestamps':m['timestamps'],'tracks_sha256':m['tracks_sha256']})
    if sum(x['timestamps'] for x in proofs)!=29019:raise ValueError('FULL_TEST_SCOPE_MISMATCH')
    # Only after all immutable outputs are verified may this observer open GT.
    from worldmot.data.visiontrack import VisionTrack
    from worldmot.training.world_object_identity_targets import match_evaluation_identities, update_history_labels, identity_key
    audit=read(root.parent/'fulltrain_20261022/audit/data_audit.json')['test']
    legal={x['scene'] for x in audit['scenes'] if x['official_gt_legal']}
    ds=VisionTrack(a.dataset_root,split='test',load_annotations=True)
    primary=Counter();blocked=Counter();scenes={};examples=defaultdict(list)
    quarter_total=[Counter() for _ in range(4)];all_graphs=Counter()
    for proof in proofs:
        sid=proof['scene'];counts=Counter();quarters=[Counter() for _ in range(4)]
        temporal=TemporalErrorAudit();active=set();history_labels={};previous={};seen=set()
        person_ids=defaultdict(Counter);gid_people=defaultdict(Counter)
        gt_visible_count=Counter();first_detected={};previous_global={}
        pairs=list(ds.synchronized_pairs(sid))
        stream=(root/'predictions'/a.method/f'scene_{sid:02d}/tracks.jsonl').open()
        for step,pair in enumerate(pairs):
            row=json.loads(next(stream));t=pair.frame_id
            if row['timestamp']!=t:raise ValueError('NONCAUSAL_TIMESTAMP_JOIN')
            before=counts.copy();gt=ds.ground_truth_pair(pair)
            for cam,anns in gt.items():
                values=Counter(identity_key(x) for x in anns)
                for label,number in values.items():
                    if label is not None and number==1:gt_visible_count[int(cam),label]+=1
            boxes=[o for camera in row['tracks_2d_by_camera'].values() for o in camera]
            obs=[SimpleNamespace(camera_id=x['camera_id'],detection_id=x['detection_id'],
                 key=(x['camera_id'],x['detection_id']),bbox_xywh=np.asarray(x['bbox_xywh'])) for x in boxes]
            labels,record=match_evaluation_identities(obs,gt)
            assignments={o.key:x['global_id'] for o,x in zip(obs,boxes)}
            decisions={(x['camera_id'],x['detection_id']):x for x in row.get('decisions',[])}
            old_ids=set(temporal.seen_ids)
            temporal_counts,events=temporal.observe(obs,labels,assignments,active,history_labels,row.get('tracks_3d'))
            counts.update(temporal_counts);counts.update({'paired_timestamps':1,'current_detections':len(obs),
                'clean_labelled_detections':sum(x is not None for x in labels),
                'unmatched_or_ambiguous_detections':sum(x is None for x in labels)})
            current=defaultdict(list)
            for j,(o,label) in enumerate(zip(obs,labels)):
                if label is None:continue
                gid=assignments[o.key];current[label].append(j)
                person_ids[label][gid]+=1;gid_people[gid][label]+=1
                first_detected.setdefault(label,step)
                key=(o.camera_id,label);prior=previous.get(key)
                if prior:
                    old,old_step,old_gt_count=prior;gap=step-old_step
                    changed=old!=gid
                    counts['per_camera_clean_temporal_comparisons']+=1
                    counts['diagnostic_per_camera_id_changes']+=int(changed)
                    if gap>1:
                        suffix='within_30' if gap<=30 else 'over_30'
                        counts['detection_gap_returns_'+suffix]+=1
                        counts['detection_gap_changed_id_'+suffix]+=int(changed)
                        gt_absent=gap-(gt_visible_count[key]-old_gt_count)
                        kind='with_GT_absent_intervals' if gt_absent>0 else 'GT_continuously_annotated'
                        counts['detection_gap_returns_'+kind]+=1
                        counts['detection_gap_changed_id_'+kind]+=int(changed)
                previous[key]=(gid,step,gt_visible_count[key])
            for label,indices in current.items():
                ids={assignments[obs[j].key] for j in indices}
                previous_entry=previous_global.get(label)
                if previous_entry:
                    old,old_step=previous_entry;gap=step-old_step
                    if gap>1:
                        counts['global_detection_gap_returns']+=1
                        counts['global_detection_gap_no_previous_id_recovered']+=int(not bool(ids&old))
                previous_global[label]=(ids,step)
            for left,o in enumerate(obs):
                if labels[left] is None:continue
                for right in range(left+1,len(obs)):
                    if labels[right] is None or o.camera_id==obs[right].camera_id:continue
                    same=labels[left]==labels[right]
                    shared=assignments[o.key]==assignments[obs[right].key]
                    counts['same_person_cross_view_pairs' if same else 'different_person_cross_view_pairs']+=1
                    kind='correct_shared_identity' if same and shared else 'same_person_split' if same else 'false_cross_view_merge' if shared else 'correctly_separate'
                    counts[kind]+=1
                    if same and first_detected[labels[left]]==step:
                        counts['first_clean_detection_pairs']+=1
                        counts['first_clean_detection_pair_splits']+=int(not shared)
                    if not same and shared:
                        decision=decisions.get(o.key,{}).get('decision','')
                        counts['false_cross_view_merge_new_birth' if assignments[o.key] not in old_ids else 'false_cross_view_merge_existing_id']+=1
                        if len(examples['FALSE_CROSS_VIEW_MERGE'])<12:
                            examples['FALSE_CROSS_VIEW_MERGE'].append({'scene':sid,'timestamp':t,'legal_GT':sid in legal,
                                'labels':[labels[left],labels[right]],'gid':assignments[o.key],'decision':decision})
            for event in events:
                if len(examples[event['error']])<20:
                    examples[event['error']].append({'scene':sid,'timestamp':t,'legal_GT':sid in legal,**event})
            update_history_labels(history_labels,assignments,obs,labels)
            active={x['global_id'] for x in row['tracks_3d']}
            seen.update(x for x in labels if x is not None)
            increment_quarter(quarters,before,counts,step,len(pairs))
        if stream.readline():raise ValueError('EXTRA_PREDICTION_TIMESTAMPS')
        stream.close()
        graph=identity_graph(person_ids)
        graph.update({'output_ids_mixed_across_clean_people':sum(len(x)>1 for x in gid_people.values()),
                      'clean_observations_in_eventually_mixed_ids':sum(sum(x.values()) for x in gid_people.values() if len(x)>1),
                      'minority_person_observations_in_mixed_ids':sum(sum(x.values())-max(x.values()) for x in gid_people.values())})
        (primary if sid in legal else blocked).update(counts)
        if sid in legal:
            for j in range(4):quarter_total[j].update(quarters[j])
            all_graphs.update({k:v for k,v in graph.items() if isinstance(v,(int,np.integer)) and not isinstance(v,bool)})
        scenes[str(sid)]={'official_GT_legal':sid in legal,'counts':dict(counts),'identity_graph':graph,
            'relative_video_quarters':[dict(x) for x in quarters]}
        write_json(a.output.with_name(a.output.stem+'.progress.json'),{'last_scene':sid,'total_scenes':22,'status':'RUNNING'})
        print(json.dumps({'attributed_scene':sid,'legal':sid in legal,'clean':counts['clean_labelled_detections'],
            'split':counts['same_person_split'],'mixed_ids':graph['output_ids_mixed_across_clean_people']}),flush=True)
    report={'schema':'worldobject_v3_completed_prediction_identity_attribution_v1','status':'COMPLETE',
        'method':a.method,'GT_role':'POSTHOC_ONLY_AFTER_ALL22_IMMUTABLE_PREDICTIONS','R2_modified':False,
        'TEST_used_for_parameter_selection':False,'official_metric':False,'diagnostic_bbox_IoU':.5,
        'source_sha256':source_fingerprint(),'analysis_sha256':sha(__file__),'prediction_proofs':proofs,
        'primary_scope':'14 original legal full TEST videos','excluded8_scope':'full inference / clean annotation subset diagnostic only',
        'legal_scene_ids':sorted(legal),'legal14_counts':dict(primary),'excluded8_diagnostic_counts':dict(blocked),
        'legal14_summed_identity_graph':dict(all_graphs),'legal14_relative_video_quarters':[dict(x) for x in quarter_total],
        'per_scene':scenes,'examples':dict(examples),
        'definitions':{
            'false_Birth_proxy':'New delivered global ID despite exactly one clean matching prior ACTIVE object; outcome attribution, not proof of Birth-head miscalibration.',
            'NULL_proxy':'Unique clean matching prior active ID serialized missed>0 although a current clean detection exists; ambiguous duplicate histories excluded.',
            'cumulative_contamination':'All clean labels ever assigned to a delivered ID; never forgets a prior error. Does not assert embedding contamination.',
            'occlusion_limit':'Detection gaps are separated by whether GT was absent in any intermediate frame. GT absence may be occlusion/out-of-view/annotation absence; it is not a visibility ground-truth sensor.',
            'IDSW_vs_CVIDF1':'Local per-camera changes and global one-to-one cross-view identity agreement measure different errors. Stable view-specific or wrongly merged IDs may have few local changes but poor global association.',
            'clean_graph':'One-to-one Hungarian overlap on clean IoU=.5 matched detections only, per-scene disjoint IDs; not TrackEval or MATLAB CVIDF1.',
        }}
    write_json(a.output,report)
    print(json.dumps({'status':'COMPLETE','output':str(a.output),'legal14':dict(primary)}),flush=True)


if __name__=='__main__':main()
