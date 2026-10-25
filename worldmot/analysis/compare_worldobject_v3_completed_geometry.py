#!/usr/bin/env python3
"""Paired Full-vs-NoGeometry posthoc attribution after BOTH all22 TEST runs.

Only completed immutable outputs enter. This report never enters R2 or R3
training, selection, scoring thresholds or prediction state.
"""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from summarize_worldobject_v3 import read,sha,write_json,official
from audit_worldobject_v3_identities import contrast

FULL='FULL_WORLD_OBJECT_V3';NG='FULL_NO_GEOMETRY_V3'


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    from worldmot.runtime.experiment_authorization import require_experiment_authorization,source_fingerprint
    for method in (FULL,NG):require_experiment_authorization(a.root/'authorization_v3.json','diagnostics','test',method)
    proofs={}
    for method in (FULL,NG):
        items=[]
        for sid in range(1,23):
            folder=a.root/'predictions'/method/f'scene_{sid:02d}';m=read(folder/'manifest.json')
            if m['status']!='COMPLETE' or m['GT_in_inference'] is not False or sha(folder/'tracks.jsonl')!=m['tracks_sha256']:
                raise ValueError('ALL22_IMMUTABLE_OUTPUTS_REQUIRED_BEFORE_GT')
            items.append({'scene':sid,'timestamps':m['timestamps'],'tracks_sha256':m['tracks_sha256']})
        if sum(x['timestamps'] for x in items)!=29019:raise ValueError('FULL22_SCOPE_REQUIRED')
        proofs[method]=items
    from worldmot.data.visiontrack import VisionTrack
    from worldmot.training.world_object_identity_targets import match_evaluation_identities
    ds=VisionTrack('/data/DATASETS/TRACKING/JDE/VisionTrack',split='test',load_annotations=True)
    legal={s['scene'] for s in read(a.root.parent/'fulltrain_20261022/audit/data_audit.json')['test']['scenes'] if s['official_gt_legal']}
    primary=Counter();blocked=Counter();scene_reports={};examples=defaultdict(list);strata=defaultdict(Counter)
    for sid in range(1,23):
        counts=Counter();last={FULL:{},NG:{}};seen=set();pairs=list(ds.synchronized_pairs(sid))
        paths={m:a.root/'predictions'/m/f'scene_{sid:02d}/tracks.jsonl' for m in (FULL,NG)}
        with paths[FULL].open() as fs,paths[NG].open() as ns:
            for step,pair in enumerate(pairs):
                outputs={FULL:json.loads(next(fs)),NG:json.loads(next(ns))};timestamp=pair.frame_id
                if any(x['timestamp']!=timestamp for x in outputs.values()):raise ValueError('TIMESTAMP_DIFFERENCE')
                objects={m:{(int(cam),x['detection_id']):x for cam,items in row['tracks_2d_by_camera'].items() for x in items} for m,row in outputs.items()}
                if set(objects[FULL])!=set(objects[NG]):raise ValueError('CANDIDATE_COVERAGE_DIFFERENCE')
                keys=sorted(objects[FULL]);obs=[]
                for key in keys:
                    if objects[FULL][key]['bbox_xywh']!=objects[NG][key]['bbox_xywh'] or objects[FULL][key]['score']!=objects[NG][key]['score']:
                        raise ValueError('CANDIDATE_BOX_SCORE_DIFFERENCE')
                    obs.append(SimpleNamespace(key=key,camera_id=key[0],detection_id=key[1],bbox_xywh=np.asarray(objects[FULL][key]['bbox_xywh'])))
                labels,_=match_evaluation_identities(obs,ds.ground_truth_pair(pair))
                decisions={(x['camera_id'],x['detection_id']):x for x in outputs[FULL]['decisions']}
                quarter=min(3,4*step//len(pairs));density='N_ge_20' if len(obs)>=20 else 'N_lt_20'
                def accumulate(kind,good):
                    contrast(counts,good[FULL],good[NG],kind)
                    if sid in legal:contrast(strata[f'quarter_{quarter}/{density}'],good[FULL],good[NG],kind)
                def sample(kind,indices):
                    key=(kind,sid in legal)
                    if len(examples[key])>=16:return
                    examples[key].append({'scene':sid,'timestamp':timestamp,'legal_GT':sid in legal,'error':kind,
                        'actors':[{'camera':keys[j][0],'detection_id':keys[j][1],'gt_identity':labels[j],
                            'Full_ID':objects[FULL][keys[j]]['global_id'],'NoGeometry_ID':objects[NG][keys[j]]['global_id'],
                            'bbox_xywh':objects[FULL][keys[j]]['bbox_xywh'],'Full_decision':decisions[keys[j]]} for j in indices]})
                for left,o in enumerate(obs):
                    label=labels[left]
                    if label is None:continue
                    temporal={};gap=None
                    for method in (FULL,NG):
                        gid=objects[method][o.key]['global_id'];prior=last[method].get((o.camera_id,label))
                        if prior:temporal[method]=prior[0]==gid;gap=step-prior[1]
                        last[method][o.camera_id,label]=(gid,step)
                    if len(temporal)==2:
                        accumulate('temporal_continuity',temporal)
                        if gap>1:accumulate('detection_gap_return_within_30' if gap<=30 else 'detection_gap_return_over_30',temporal)
                        if temporal[FULL] and not temporal[NG]:sample('Full_TEMPORAL_CORRECTION',[left])
                        elif temporal[NG] and not temporal[FULL]:sample('Full_NEW_TEMPORAL_ERROR',[left])
                    for right in range(left+1,len(obs)):
                        if labels[right] is None or o.camera_id==obs[right].camera_id:continue
                        same=label==labels[right]
                        good={m:(objects[m][o.key]['global_id']==objects[m][obs[right].key]['global_id'])==same for m in (FULL,NG)}
                        kind='same_person' if same else 'different_person';accumulate(kind,good)
                        reliable=all(decisions[keys[j]]['cross_view_fusion_eligible'] for j in (left,right))
                        if sid in legal:contrast(strata['both_current_geometry_eligible' if reliable else 'one_or_both_current_geometry_ineligible'],good[FULL],good[NG],kind)
                        if same and label not in seen:accumulate('first_clean_detection_same_person',good)
                        if good[FULL] and not good[NG]:sample('Full_'+kind.upper()+'_CORRECTION',[left,right])
                        elif good[NG] and not good[FULL]:sample('Full_'+kind.upper()+'_NEW_ERROR',[left,right])
                seen.update(x for x in labels if x is not None);counts['paired_timestamps']+=1;counts['detections']+=len(obs)
            if fs.readline() or ns.readline():raise ValueError('EXTRA_PREDICTION_FRAMES')
        for kind in ('same_person','different_person','first_clean_detection_same_person','temporal_continuity',
                     'detection_gap_return_within_30','detection_gap_return_over_30'):
            counts[kind+'_net_corrections']=counts[kind+'_corrected']-counts[kind+'_new_errors']
        (primary if sid in legal else blocked).update(counts)
        scene_reports[str(sid)]={'original_GT_legal':sid in legal,'counts':dict(counts)}
        print(json.dumps({'paired_posthoc_scene':sid,'legal':sid in legal,'temporal_net':counts['temporal_continuity_net_corrections']}),flush=True)
    scores={m:official(a.root/'evaluation'/m/'metrics.json') for m in (FULL,NG)}
    report={'schema':'worldobject_v3_completed_Full_vs_NoGeometry_paired_attribution_v1','status':'COMPLETE',
        'source_sha256':source_fingerprint(),'analysis_sha256':sha(__file__),'threshold_selection_on_TEST':False,
        'R2_or_R3_state_modified':False,'GT_role':'POSTHOC_ONLY_AFTER_BOTH_ALL22_IMMUTABLE_OUTPUTS',
        'primary_scope':'14 original legal complete TEST videos','official_metric':False,'diagnostic_bbox_IoU':.5,
        'proofs':proofs,'legal14_counts':dict(primary),'blocked8_clean_subset_diagnostic_counts':dict(blocked),
        'per_scene':scene_reports,'legal14_strata':{k:dict(v) for k,v in strata.items()},
        'examples':[e for items in examples.values() for e in items],
        'official_metrics_separate':scores,
        'interpretation_limit':'Paired outcomes measure the registered Full vs NoGeometry systems with the same current boxes/ROIs. Models have different geometry inputs, learned weights and TRAIN-selected final budgets. Current geometry eligibility strata are correlations, not isolated causal interventions.',
        'training_isolation':'This TEST report is never an input to the separately preregistered TRAIN-only R3 replay; R2 remains frozen.'}
    write_json(a.output,report)
    print(json.dumps({'status':'COMPLETE','legal14':dict(primary)}),flush=True)


if __name__=='__main__':main()
