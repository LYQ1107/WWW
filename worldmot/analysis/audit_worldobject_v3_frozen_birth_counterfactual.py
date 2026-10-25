#!/usr/bin/env python3
"""TRAIN-only one-step counterfactuals on frozen R2 states/candidate graphs.

Actual assignment is returned unchanged. Hypothetical assignments are discarded
and never form the next history. No optimizer, TEST or reconstruction forward.
"""
import argparse
from collections import Counter,defaultdict
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
from summarize_worldobject_v3 import read,sha,write_json

MODES=('actual','no_new_joint_birth','no_history_competition','no_history_tiny_positive','same_logits_no_geometry_gates')
TINY_BENEFIT=1e-6*float(1/(1+np.exp(60.)))


def assignment_partition(selected,n):
    assigned={}
    for option in selected:
        group=('history',option.global_id) if option.global_id is not None else ('new_joint',tuple(option.indices))
        for index in option.indices:
            if index in assigned:raise ValueError('COUNTERFACTUAL_DUPLICATE_OBSERVATION')
            assigned[index]=group
    for index in range(n):assigned.setdefault(index,('new_singleton',index))
    return assigned


def pair_outcomes(partition,observations,labels):
    same=defaultdict(list);groups=defaultdict(list);counts=Counter()
    for j,(o,label) in enumerate(zip(observations,labels)):
        if label is not None:same[label].append(j);groups[partition[j]].append(j)
    counts['same_person_pairs']=sum(len(indices)==2 for indices in same.values())
    for group,indices in groups.items():
        if len(indices)!=2:continue
        left,right=indices
        if observations[left].camera_id==observations[right].camera_id:raise ValueError('COUNTERFACTUAL_SAME_CAMERA_MERGE')
        is_same=labels[left]==labels[right]
        counts['correct_same_person_fusions' if is_same else 'different_person_merges']+=1
        if group[0]=='new_joint':counts['selected_correct_joint_birth' if is_same else 'selected_wrong_joint_birth']+=1
        else:counts['correct_history_two_view_group' if is_same else 'wrong_history_two_view_group']+=1
    counts['same_person_splits']=counts['same_person_pairs']-counts['correct_same_person_fusions']
    return counts


class FrozenBirthCounterfactual:
    def __init__(self,core):
        self.core=core;assigner=core.bank.assigner
        original_assign=assigner.assign;original_solve=assigner._solve_component
        self.counts={mode:Counter() for mode in MODES};self.competition=Counter();self.strata=defaultdict(Counter)
        self.examples=[];self.partitions={}

        def solve(groups):
            actual,audit=original_solve(groups);self.components.extend(groups)
            self.selected['actual'].extend(actual)
            history=[[o for o in group if o.global_id is not None] for group in groups]
            births=[[o for o in group if o.global_id is None] for group in groups]
            for name,candidate in [('no_new_joint_birth',history),('no_history_competition',births)]:
                candidate=[g for g in candidate if g]
                if candidate:self.selected[name].extend(original_solve(candidate)[0])
            tiny=[[replace(o,benefit=TINY_BENEFIT if o.benefit>0 else 0.) for o in group] for group in births if group]
            if tiny:self.selected['no_history_tiny_positive'].extend(original_solve(tiny)[0])
            return actual,audit

        def assign(history,observations,learned=None):
            from worldmot.tracking.joint_world_object_assignment import JointWorldObjectAssignment
            from worldmot.tracking.world_object_config import WorldObjectConfig
            self.components=[];self.selected={mode:[] for mode in MODES}
            result=original_assign(history,observations,learned)
            if self.selected['actual']!=result.selected:raise ValueError('ACTUAL_SELECTED_OPTIONS_CHANGED')
            # Feed identical learned evidence/history/observations into a fresh
            # assigner with only geometry gates disabled. No bank update.
            cfg={**core.config.to_dict(),'method_id':'FULL_NO_GEOMETRY_V3'}
            shadow=JointWorldObjectAssignment(WorldObjectConfig(**cfg))
            self.selected['same_logits_no_geometry_gates']=shadow.assign(history,observations,learned).selected
            self.partitions={mode:assignment_partition(options,len(observations)) for mode,options in self.selected.items()}
            for mode,partition in self.partitions.items():self.counts[mode].update(pair_outcomes(partition,observations,self.labels))
            self.competition['paired_timestamps']+=1
            if learned is not None:
                probabilities=1/(1+np.exp(-np.clip(learned['birth_pair_logits'],-60,60)))
                correct_history_cover=set()
                for group in self.components:
                    for option in group:
                        if option.global_id is None:continue
                        truth=self.history_labels.get(option.global_id)
                        if truth is not None and all(self.labels[j]==truth for j in option.indices):correct_history_cover.update(option.indices)
                proposals=assigner.last_birth_candidate_edges
                for left in range(len(observations)):
                    for right in range(left+1,len(observations)):
                        a,b=observations[left],observations[right];la,lb=self.labels[left],self.labels[right]
                        if a.camera_id==b.camera_id or la is None or lb is None or la==lb or probabilities[left,right]<core.config.birth_pair_threshold:continue
                        key=(left,right);c=self.competition;c['raw_high_wrong_pairs']+=1
                        c['high_wrong_pairs_reaching_TopK_graph']+=int(key in proposals)
                        endpoints=int(left in correct_history_cover)+int(right in correct_history_cover)
                        c[f'high_wrong_pairs_with_{endpoints}_endpoints_covered_by_clean_correct_history_options']+=1
                        cosine=float(a.roi_feature@b.roi_feature)
                        quality='both_geometry_eligible' if a.cross_view_fusion_eligible and b.cross_view_fusion_eligible else 'one_or_both_geometry_ineligible'
                        appearance='cos_ge_0_9' if cosine>=.9 else 'cos_0_55_to_0_9' if cosine>=.55 else 'cos_lt_0_55'
                        density='detections_ge_20' if len(observations)>=20 else 'detections_lt_20'
                        stratum=self.strata[f'scene_{self.scene}/{density}/{quality}/{appearance}'];stratum['raw_high_wrong_pairs']+=1
                        stratum['TopK_wrong_proposals']+=int(key in proposals)
                        for mode,partition in self.partitions.items():
                            merged=partition[left]==partition[right]
                            stratum[mode+'_wrong_pair_merged']+=int(merged)
                        if key in proposals and self.partitions['actual'][left]!=self.partitions['actual'][right] and self.partitions['no_history_tiny_positive'][left]==self.partitions['no_history_tiny_positive'][right] and len(self.examples)<24:
                            self.examples.append({'scene':self.scene,'timestamp':self.timestamp,'indices':[left,right],
                                'labels':[la,lb],'actual_pair_probability':float(probabilities[left,right]),
                                'singleton_probabilities':[float(1/(1+np.exp(-np.clip(learned['birth_logits'][j],-60,60)))) for j in key],
                                'correct_history_covered_endpoints':endpoints,'actual_merged':False,'hypothetical_tiny_no_history_merged':True,
                                'geometry_quality':quality,'cosine':cosine})
            return result

        assigner._solve_component=solve;assigner.assign=assign

    def start(self,observations,labels,timestamp,history_labels,scene):
        self.labels=labels;self.timestamp=timestamp;self.history_labels=dict(history_labels);self.scene=scene;self.observations=observations

    def finish(self):
        delivered=defaultdict(set);partition=defaultdict(set)
        for j,o in enumerate(self.observations):
            delivered[self.core.bank.last_assignments[o.key]].add(j);partition[self.partitions['actual'][j]].add(j)
        if {frozenset(x) for x in delivered.values()}!={frozenset(x) for x in partition.values()}:
            raise ValueError('OBSERVED_ACTUAL_GRAPH_DIFFERS_FROM_DELIVERED_IDS')


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--method',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    import torch
    from worldmot.runtime.experiment_authorization import require_experiment_authorization,source_fingerprint
    require_experiment_authorization(a.root/'authorization_v3.json','diagnostics','train',a.method)
    from worldmot.models.world_object_update_head import WorldObjectUpdateHead
    from worldmot.models.gmt_detection_cache import ImmutableGMTDetectionCache
    from worldmot.tracking.world_object_config import WorldObjectConfig
    from worldmot.runtime.world_object_tracking_core import WorldObjectTrackingCore
    from worldmot.training.world_object_causal_training import CausalWorldObjectTraining
    from worldmot.training.world_object_identity_targets import match_train_identities
    from worldmot.data.worldobject_cache import WorldObjectCache
    from worldmot.data.worldobject_observations import replay_observations
    from worldmot.data.visiontrack import VisionTrack
    records=read(a.root/'audit_snapshots/epoch_archive_index.json')['records']
    record=next(x for x in records if x['method']==a.method and x['usable_as_selection_epoch_checkpoint'] and x['selection_epochs']==1)
    if sha(record['path'])!=record['sha256'] or not record['resume_metadata_SHA_matched_at_capture']:raise ValueError('BOUND_SELECTION1_WEIGHTS_REQUIRED')
    state=torch.load(record['path'],map_location='cpu',weights_only=False);torch.set_num_threads(2)
    raw=ImmutableGMTDetectionCache(a.root.parent/'gmt_detector_comparison_20261023/raw_train_v1');config=WorldObjectConfig(raw.feature_dim,a.method)
    model=WorldObjectUpdateHead(raw.feature_dim,128).to('cuda:0');model.load_state_dict(state['model']);model.eval()
    ds=VisionTrack('/data/DATASETS/TRACKING/JDE/VisionTrack',split='train',load_annotations=True)
    total={m:Counter() for m in MODES};competition=Counter();strata=defaultdict(Counter);scenes={};examples=[];diagnostics=Counter()
    for scene in [4,8,12,16,20,24]:
        core=WorldObjectTrackingCore(config,model,'cuda:0');runner=CausalWorldObjectTraining(core,'predicted');runner.reset(scene);observer=FrozenBirthCounterfactual(core)
        pairs={x.frame_id:x for x in ds.synchronized_pairs(scene)};cache=WorldObjectCache(a.root/'cache_train_v2',scene,raw_cache=raw,require_features=True,require_geometry=True)
        count=0
        for frame in cache.frames():
            obs=sorted(replay_observations(frame),key=lambda o:o.key);timestamp=frame['timestamp'];labels,_=match_train_identities(obs,ds.ground_truth_pair(pairs[timestamp]))
            observer.start(obs,labels,timestamp,runner.history_labels,scene);runner.step(obs,timestamp,labels,optimizer=None);observer.finish();diagnostics.update(runner.last_diagnostics);count+=1
        if count!=len(pairs):raise ValueError('INCOMPLETE_FIXED_VAL_SCENE')
        for mode,values in observer.counts.items():total[mode].update(values)
        competition.update(observer.competition)
        for key,value in observer.strata.items():strata[key].update(value)
        examples.extend(observer.examples);scenes[str(scene)]={'timestamps':count,'modes':{k:dict(v) for k,v in observer.counts.items()},'competition':dict(observer.competition)}
        print(json.dumps({'frozen_birth_cf_scene_complete':scene,'method':a.method,'actual_wrong_birth':observer.counts['actual']['selected_wrong_joint_birth']}),flush=True)
    expected=state['selection'][0]['validation']['diagnostics']
    mismatch={k:[value,diagnostics.get(k,0)] for k,value in expected.items() if not k.startswith(('teacher_prefix_','predicted_prefix_')) and diagnostics.get(k,0)!=value}
    report={'schema':'worldobject_v3_full_TRAIN_VAL_frozen_state_birth_counterfactual_v1','status':'COMPLETE','method':a.method,
        'checkpoint_sha256':record['sha256'],'selection_epoch':1,'source_sha256':source_fingerprint(),'analysis_sha256':sha(__file__),
        'scenes':[4,8,12,16,20,24],'timestamps':sum(x['timestamps'] for x in scenes.values()),
        'baseline_original_diagnostics_exact':not mismatch,'original_diagnostic_mismatches':mismatch,
        'actual_and_hypothetical_counts':{k:dict(v) for k,v in total.items()},'competition_counts':dict(competition),
        'strata':{k:dict(v) for k,v in strata.items()},'per_scene':scenes,'examples':examples,
        'hypothetical_tiny_benefit':TINY_BENEFIT,'hypothetical_margin_above_registered_pair_threshold':1e-6,'hypothetical_singleton_birth_logit':-60.,
        'R2_predictions_or_bank_state_modified':False,'hypothetical_decisions_propagated':False,'optimizer_used':False,'TEST_read':False,
        'interpretation':'One-step interventions on actual frozen TRAIN histories/evidence. Removing history competition or replacing positive Birth benefits is a hypothetical stress test, not observed R2 performance. Geometry-gates-off feeds identical learned logits into a fresh assigner; it is not a retrained NoGeometry model.'}
    write_json(a.output,report);print(json.dumps({'status':'COMPLETE','method':a.method,'baseline_exact':not mismatch,'modes':report['actual_and_hypothetical_counts']}),flush=True)


if __name__=='__main__':main()
