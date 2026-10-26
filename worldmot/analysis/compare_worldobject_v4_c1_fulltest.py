#!/usr/bin/env python3
"""Read-only C1 vs archived Full V3 after BOTH all22 outputs freeze.

Reuse the V3 clean matcher, TemporalErrorAudit, history-label update and global
identity graph. Pair the same posthoc GT events; never execute or tune a model.
"""
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
from types import SimpleNamespace
import argparse
import json
import time
import numpy as np

from worldmot.runtime.world_object_v4_test_contract import (
    require_scope,completed_scene,compare_candidates,output_candidates,write_json,sha256,LEGAL_TEST)
from worldmot.training.world_object_identity_targets import match_evaluation_identities,update_history_labels,identity_key
from worldobject_v3_temporal_audit import TemporalErrorAudit,increment_quarter
from attribute_worldobject_v3_completed_predictions import identity_graph

ANNOTATIONS_SHA='7a1e735bf25026a56148deaeb432e0520c6ad5dad97d5362aa58739255389593'
METHODS=('C0_Full_V3','C1')


def paired(counter,left,right):
    a=np.asarray(left,dtype=bool).reshape(-1);b=np.asarray(right,dtype=bool).reshape(-1)
    if a.shape!=b.shape:raise ValueError('PAIRED_EVENT_SHAPE_DIFFERENCE')
    counter['events']+=int(a.size);counter['both_correct']+=int((a&b).sum())
    counter['both_wrong']+=int((~a&~b).sum());counter['corrections']+=int((~a&b).sum())
    counter['regressions']+=int((a&~b).sum())


def finish(counter):
    d=dict(counter);d['net_corrections']=d.get('corrections',0)-d.get('regressions',0);return d


def self_test():
    c=Counter();paired(c,[0,1,1,0],[1,0,1,0])
    assert dict(c)=={'events':4,'both_correct':1,'both_wrong':1,'corrections':1,'regressions':1}
    assert finish(c)['net_corrections']==0
    labels=np.array(['a','b']);left=np.array([1,2]);right=np.array([3,2])
    same=labels[:,None]==labels[None,:];good=(left[:,None]==right[None,:])==same
    slow=[]
    for i in range(2):
        for j in range(2):slow.append((left[i]==right[j])==(labels[i]==labels[j]))
    assert np.array_equal(good.reshape(-1),slow)
    assert int((~good&same).sum())==1 and int((~good&~same).sum())==0
    graph=identity_graph({'a':Counter({1:5,3:1}),'b':Counter({2:4})})
    assert graph['global_one_to_one_matched_clean_observations']==9 and graph['clean_observations']==10


def new_state():
    return {'temporal':TemporalErrorAudit(),'active':set(),'history_labels':{},'previous':{},
        'person_ids':defaultdict(Counter),'gid_people':defaultdict(Counter),'counts':Counter(),
        'quarters':[Counter() for _ in range(4)]}


def method_events(state,row,obs,labels,step,gt_visible):
    counts=state['counts'];before=counts.copy();temporal=state['temporal']
    objects=output_candidates(row);assigned={o.key:objects[o.key]['global_id'] for o in obs}
    unique=defaultdict(list)
    for gid in state['active']:
        label=state['history_labels'].get(gid)
        if label is not None:unique[label].append(gid)
    prior_sets={gid:set(temporal.cumulative_labels.get(gid,{})) for gid in set(assigned.values())}
    seen=set(temporal.seen_ids);valid=[i for i,x in enumerate(labels) if x is not None]
    events={};opportunity=set();bad_null=set()
    c,errors=temporal.observe(obs,labels,assigned,state['active'],state['history_labels'],row['tracks_3d'])
    counts.update(c)
    for event in errors:
        if event['error']=='NULL_DESPITE_UNIQUE_CLEAN_CURRENT_PERSON':
            bad_null.update(labels[i] for i in event['observation_indices'])
    for label in {x for x in labels if x is not None}:
        if len(unique[label])==1:opportunity.add(label)
    for i in valid:
        o,label=obs[i],labels[i];gid=assigned[o.key];old=prior_sets[gid]
        opportunity_key=len(unique[label])==1
        events['false_birth_proxy',o.key]=not(gid not in seen and opportunity_key)
        if opportunity_key:events['false_birth_common_unique_history',o.key]=events['false_birth_proxy',o.key]
        events['wrong_historical_ID_reuse',o.key]=not(bool(old) and label not in old)
        events['reuse_of_cumulatively_mixed_ID',o.key]=not(len(old)>1)
        events['assignment_to_cumulatively_mixed_ID',o.key]=not(len(temporal.cumulative_labels[gid])>1)
        state['person_ids'][label][gid]+=1;state['gid_people'][gid][label]+=1
        key=o.camera_id,label;previous=state['previous'].get(key)
        if previous:
            old_gid,old_step,old_gt_count=previous;gap=step-old_step;correct=old_gid==gid
            counts['per_camera_clean_temporal_comparisons']+=1;counts['diagnostic_per_camera_id_changes']+=int(not correct)
            events['temporal_continuity',key]=correct
            if gap>1:
                suffix='within_30' if gap<=30 else 'over_30'
                counts['detection_gap_returns_'+suffix]+=1;counts['detection_gap_changed_id_'+suffix]+=int(not correct)
                events['detection_gap_return_'+suffix,key]=correct
                absent=gap-(gt_visible[key]-old_gt_count)
                kind='with_GT_absent_intervals' if absent>0 else 'GT_continuously_annotated'
                counts['detection_gap_returns_'+kind]+=1;counts['detection_gap_changed_id_'+kind]+=int(not correct)
                events['detection_gap_return_'+kind,key]=correct
        state['previous'][key]=(gid,step,gt_visible[key])
    for label in opportunity:events['NULL_despite_current_detection',label]=label not in bad_null
    update_history_labels(state['history_labels'],assigned,obs,labels)
    state['active']={x['global_id'] for x in row['tracks_3d']}
    counts['paired_timestamps']+=1;counts['detections']+=len(obs);counts['clean_labelled_detections']+=len(valid)
    return events,assigned,before


def graph_and_mapping(state):
    graph=identity_graph(state['person_ids']);people=state['gid_people']
    graph.update(output_ids_mixed_across_clean_people=sum(len(x)>1 for x in people.values()),
        clean_observations_in_eventually_mixed_ids=sum(sum(x.values()) for x in people.values() if len(x)>1),
        minority_person_observations_in_mixed_ids=sum(sum(x.values())-max(x.values()) for x in people.values()))
    from scipy.optimize import linear_sum_assignment
    labels=sorted(state['person_ids']);gids=sorted(people);lookup={g:i for i,g in enumerate(gids)}
    matrix=np.zeros((len(labels),len(gids)),np.int64)
    for i,label in enumerate(labels):
        for gid,n in state['person_ids'][label].items():matrix[i,lookup[gid]]=n
    rows,cols=linear_sum_assignment(-matrix)
    assert int(matrix[rows,cols].sum())==graph['global_one_to_one_matched_clean_observations']
    return graph,{labels[i]:gids[j] for i,j in zip(rows,cols)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path);p.add_argument('--r2-root',type=Path)
    p.add_argument('--self-test',action='store_true');a=p.parse_args();self_test()
    if a.self_test:
        print(json.dumps({'posthoc_synthetic_checks':'PASSED'}));return
    started=time.perf_counter();script_sha=sha256(__file__)
    scope,approval=require_scope(a.root/'scope_v4_c1_TEST.json',a.root/'authorization_v4_c1_TEST.json','diagnostics')
    coverage=json.loads((a.root/'v4_c1_fulltest_coverage.json').read_text())
    if coverage['status']!='COMPLETE' or coverage['timestamps']!=29019 or not coverage['all_outputs_frozen_before_GT']:
        raise ValueError('ALL22_C1_MUST_FREEZE_BEFORE_GT')
    proofs={m:[] for m in METHODS}
    for sid in range(1,23):
        m=completed_scene(a.root/f'predictions/V4_C1/scene_{sid:02d}',scope,sid)
        proofs['C1'].append({'scene':sid,'timestamps':m['timestamps'],'tracks_sha256':m['tracks_sha256']})
        folder=a.r2_root/f'predictions/FULL_WORLD_OBJECT_V3/scene_{sid:02d}';old=json.loads((folder/'manifest.json').read_text())
        if old['status']!='COMPLETE' or sha256(folder/'tracks.jsonl')!=scope['scenes'][str(sid)]['original_R2_tracks_sha256']:
            raise ValueError('ORIGINAL_FULL_V3_IMMUTABLE_OUTPUT_REQUIRED')
        proofs['C0_Full_V3'].append({'scene':sid,'timestamps':old['timestamps'],'tracks_sha256':old['tracks_sha256']})
    if any(sum(x['timestamps'] for x in values)!=29019 for values in proofs.values()):raise ValueError('ALL22_REQUIRED')
    # This is the first point where original annotation contents may be opened.
    from worldmot.data.visiontrack import VisionTrack
    from worldmot.models.gmt_detection_cache import ImmutableGMTDetectionCache
    annotation=Path('/data/DATASETS/TRACKING/JDE/VisionTrack/annotations/test.json')
    if sha256(annotation)!=ANNOTATIONS_SHA:raise ValueError('ORIGINAL_TEST_GT_CHANGED')
    ds=VisionTrack(annotation.parents[1],split='test',load_annotations=True);raw=ImmutableGMTDetectionCache(scope['raw_cache'])
    primary={m:Counter() for m in METHODS};blocked={m:Counter() for m in METHODS}
    graphs={m:Counter() for m in METHODS};pairs_primary=defaultdict(Counter);pairs_blocked=defaultdict(Counter)
    strata=defaultdict(Counter);scenes={};examples=[];sample_counts=Counter()
    for sid in range(1,23):
        states={m:new_state() for m in METHODS};paired_scene=defaultdict(Counter);gt_visible=Counter();global_records=[]
        paths={'C0_Full_V3':a.r2_root/f'predictions/FULL_WORLD_OBJECT_V3/scene_{sid:02d}/tracks.jsonl',
               'C1':a.root/f'predictions/V4_C1/scene_{sid:02d}/tracks.jsonl'}
        timeline=list(ds.synchronized_pairs(sid));legal=sid in LEGAL_TEST
        with paths['C0_Full_V3'].open() as f0,paths['C1'].open() as f1:
            for step,pair in enumerate(timeline):
                rows={'C0_Full_V3':json.loads(next(f0)),'C1':json.loads(next(f1))};t=pair.frame_id
                if any(x['timestamp']!=t for x in rows.values()):raise ValueError('TIMESTAMP_JOIN_DIFFERENCE')
                compare_candidates(rows['C0_Full_V3'],rows['C1']);boxes=output_candidates(rows['C0_Full_V3']);keys=sorted(boxes)
                obs=[SimpleNamespace(key=k,camera_id=k[0],detection_id=k[1],bbox_xywh=np.asarray(boxes[k]['bbox_xywh'])) for k in keys]
                gt=ds.ground_truth_pair(pair);labels,match=match_evaluation_identities(obs,gt)
                for camera,anns in gt.items():
                    values=Counter(identity_key(x) for x in anns)
                    for label,n in values.items():
                        if label is not None and n==1:gt_visible[int(camera),label]+=1
                events={};assigned={};before={}
                for m in METHODS:
                    events[m],assigned[m],before[m]=method_events(states[m],rows[m],obs,labels,step,gt_visible)
                    states[m]['counts'].update({'GT_join/'+k:v for k,v in match.items()})
                for event in events['C0_Full_V3'].keys()&events['C1'].keys():
                    kind=event[0];left,right=events['C0_Full_V3'][event],events['C1'][event]
                    paired(paired_scene[kind],[left],[right])
                    if legal:paired(strata[f'quarter_{min(3,4*step//len(timeline))}/'+kind],[left],[right])
                    effect='correction' if right and not left else 'regression' if left and not right else None
                    sample_key=kind,effect,legal
                    if effect and sample_counts[sample_key]<8:
                        sample_counts[sample_key]+=1
                        examples.append({'scene':sid,'timestamp':t,'original_GT_legal':legal,'kind':kind,'effect':effect,
                            'event_key':str(event[1]),'scope':'posthoc frozen-output event, not a model input'})
                indices={v:[i for i,o in enumerate(obs) if o.camera_id==v and labels[i] is not None] for v in (1,2)}
                i1,i2=indices[1],indices[2]
                if i1 and i2:
                    same=np.asarray([labels[i] for i in i1],object)[:,None]==np.asarray([labels[i] for i in i2],object)[None,:]
                    good={};shared={}
                    for m in METHODS:
                        shared[m]=np.asarray([assigned[m][keys[i]] for i in i1])[:,None]==np.asarray([assigned[m][keys[i]] for i in i2])[None,:]
                        good[m]=shared[m]==same;c=states[m]['counts']
                        c['same_person_cross_view_pairs']+=int(same.sum());c['different_person_cross_view_pairs']+=int((~same).sum())
                        c['correct_shared_identity']+=int((shared[m]&same).sum());c['same_person_split']+=int((~shared[m]&same).sum())
                        c['false_cross_view_merge']+=int((shared[m]&~same).sum());c['correctly_separate']+=int((~shared[m]&~same).sum())
                    detections=raw.get(sid,t,image_ids={v:r.image_id for v,r in pair.images.items()})
                    features={(v,d.detection_id):np.asarray(d.detector_roi_feature,np.float32) for v,ds_ in detections.items() for d in ds_}
                    z1=np.asarray([features[keys[i]] for i in i1]);z2=np.asarray([features[keys[i]] for i in i2])
                    cosine=(z1/np.maximum(np.linalg.norm(z1,axis=1,keepdims=True),1e-12))@(z2/np.maximum(np.linalg.norm(z2,axis=1,keepdims=True),1e-12)).T
                    for kind,mask in [('same_person_shared_ID',same),('different_person_separation',~same)]:
                        paired(paired_scene[kind],good['C0_Full_V3'][mask],good['C1'][mask])
                        if legal:
                            density='N_ge20' if len(obs)>=20 else 'N_lt20';paired(strata[density+'/'+kind],good['C0_Full_V3'][mask],good['C1'][mask])
                            hard=mask&((cosine<=.55) if kind.startswith('same') else (cosine>=.90))
                            paired(strata[('same_person_low_ROI_cosine' if kind.startswith('same') else 'different_person_high_ROI_cosine')],good['C0_Full_V3'][hard],good['C1'][hard])
                        for effect,changed in [('correction',~good['C0_Full_V3']&good['C1']&mask),('regression',good['C0_Full_V3']&~good['C1']&mask)]:
                            group=kind,effect,legal
                            for x,y in np.argwhere(changed)[:max(0,8-sample_counts[group])]:
                                sample_counts[group]+=1;actors=[i1[x],i2[y]]
                                examples.append({'scene':sid,'timestamp':t,'original_GT_legal':legal,'kind':kind,'effect':effect,
                                    'actors':[{'camera':keys[j][0],'detection_id':keys[j][1],'GT_posthoc':labels[j],
                                        'C0_ID':assigned['C0_Full_V3'][keys[j]],'C1_ID':assigned['C1'][keys[j]],'bbox_xywh':boxes[keys[j]]['bbox_xywh']} for j in actors]})
                for i,label in enumerate(labels):
                    if label is not None:global_records.append((label,assigned['C0_Full_V3'][keys[i]],assigned['C1'][keys[i]]))
                for m in METHODS:increment_quarter(states[m]['quarters'],before[m],states[m]['counts'],step,len(timeline))
            if f0.readline() or f1.readline():raise ValueError('EXTRA_PREDICTIONS')
        scene_graphs={};maps={}
        for m in METHODS:scene_graphs[m],maps[m]=graph_and_mapping(states[m])
        good0=[maps['C0_Full_V3'].get(label)==g0 for label,g0,g1 in global_records]
        good1=[maps['C1'].get(label)==g1 for label,g0,g1 in global_records]
        paired(paired_scene['global_one_to_one_retention'],good0,good1)
        for m in METHODS:
            target=primary if legal else blocked;target[m].update(states[m]['counts'])
            if legal:graphs[m].update({k:v for k,v in scene_graphs[m].items() if isinstance(v,int) and not isinstance(v,bool)})
        target=pairs_primary if legal else pairs_blocked
        for kind,c in paired_scene.items():target[kind].update(c)
        scenes[str(sid)]={'original_GT_legal':legal,'timestamps':len(timeline),
            'counts':{m:dict(states[m]['counts']) for m in METHODS},'identity_graph':scene_graphs,
            'paired_C1_vs_C0':{k:finish(v) for k,v in paired_scene.items()},
            'relative_video_quarters':{m:[dict(q) for q in states[m]['quarters']] for m in METHODS}}
        write_json(a.root/'identity_attribution_progress.json',{'last_scene':sid,'scenes':len(scenes),'status':'RUNNING'})
        print(json.dumps({'attribution_scene':sid,'legal_GT':legal,'temporal':finish(paired_scene['temporal_continuity'])}),flush=True)
    if sha256(annotation)!=ANNOTATIONS_SHA or sha256(__file__)!=script_sha:raise ValueError('GT_OR_ANALYSIS_SOURCE_CHANGED')
    report={'schema':'worldmot_v4_c1_frozen_full_TEST_paired_identity_v1','status':'COMPLETE','approval':approval,
        'source_sha256':scope['source_sha256'],'analysis_sha256':script_sha,'prediction_proofs':proofs,
        'original_GT_sha256':ANNOTATIONS_SHA,'GT_role':'POSTHOC_ONLY_AFTER_ALL22_FROZEN_PREDICTIONS',
        'official_metric':False,'TEST_parameter_selection':False,'new_model_inference_executed':False,
        'matcher':'unchanged V3 clean greedy one-to-one IoU .5 matcher; not a replacement for native metrics',
        'legal14_scenes':list(LEGAL_TEST),'legal14_counts':{m:dict(c) for m,c in primary.items()},
        'blocked8_clean_subset_diagnostic_counts':{m:dict(c) for m,c in blocked.items()},
        'legal14_identity_graph':{m:dict(c) for m,c in graphs.items()},
        'legal14_paired_C1_vs_C0':{k:finish(c) for k,c in pairs_primary.items()},
        'blocked8_paired_diagnostic_C1_vs_C0':{k:finish(c) for k,c in pairs_blocked.items()},
        'legal14_strata':{k:finish(c) for k,c in strata.items()},'per_scene':scenes,'examples':examples,
        'definitions':{'false_birth_proxy':'New delivered ID despite exactly one clean prior ACTIVE target; histories differ, not proof of Birth-head miscalibration.',
            'NULL':'Only paired where both methods have a unique clean active target for the same currently detected GT person.',
            'mixed_ID':'Cumulative clean labels; reuse may include the current correct person, not all mixed reuse is new wrong-person contamination.',
            'gap':'Matched-detection gap; GT absent intervals may be occlusion/out-of-view/annotation absence, not measured physical visibility.',
            'global_graph':'Posthoc whole-video Hungarian one-to-one overlap on clean matched detections, not CVIDF1.',
            'appearance':'Cosine of current frozen raw GMT ROI descriptors; low cosine is not a calibrated camera angle.'},
        'created_UTC':datetime.now(timezone.utc).isoformat(),'wall_s':time.perf_counter()-started}
    write_json(a.root/'v4_c1_identity_attribution.json',report)
    print(json.dumps({'status':'COMPLETE','output':str(a.root/'v4_c1_identity_attribution.json')}),flush=True)


if __name__=='__main__':main()
