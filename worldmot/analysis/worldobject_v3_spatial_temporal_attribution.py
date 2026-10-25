#!/usr/bin/env python3
"""Observe frozen R2 TRAIN VAL decisions, lifecycle calibration and ID breaks.

The wrappers return the exact original values. GT is a loss/diagnostic sidecar,
never a feature, admissibility rule or replacement state. No optimizer exists.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import time
import numpy as np
from summarize_worldobject_v3 import read, sha, write_json
from worldobject_v3_temporal_audit import TemporalErrorAudit


def sigmoid(value):
    return 1/(1+np.exp(-np.clip(value,-60,60)))


class Calibration:
    def __init__(self):
        self.rows=np.zeros((10,5),dtype=np.float64)

    def add(self, probabilities, targets):
        probabilities=np.asarray(probabilities,dtype=float).reshape(-1)
        targets=np.asarray(targets,dtype=float).reshape(-1)
        if len(probabilities)!=len(targets) or not np.isfinite(probabilities).all():
            raise ValueError('INVALID_CALIBRATION_INPUT')
        for i in range(10):
            keep=np.minimum((probabilities*10).astype(int),9)==i
            p,y=probabilities[keep],targets[keep]
            self.rows[i]+=np.array([len(p),p.sum(),y.sum(),((p-y)**2).sum(),
                (((p>=.5).astype(float))==y).sum()])

    def report(self):
        total=self.rows.sum(0);n=total[0]
        return {'count':int(n),'mean_probability':float(total[1]/max(1,n)),
            'positive_target_fraction':float(total[2]/max(1,n)),
            'Brier':float(total[3]/max(1,n)),
            'ECE_10_equal_width':float(np.abs(self.rows[:,1]-self.rows[:,2]).sum()/max(1,n)),
            'threshold_0_5_accuracy':float(total[4]/max(1,n)),
            'bins':[{'range':[i/10,(i+1)/10],'count':int(r[0]),
                     'probability_sum':float(r[1]),'positive_target_count':int(r[2]),
                     'squared_error_sum':float(r[3])} for i,r in enumerate(self.rows)]}


class FrozenDecisionObserver:
    def __init__(self, core):
        self.core=core;self.config=core.config
        self.counts=Counter();self.funnel=defaultdict(Counter)
        self.calibration=defaultdict(Calibration);self.strata=defaultdict(Counter)
        self.examples=defaultdict(list);self.temporal=TemporalErrorAudit()
        self.step_index=0;self.pending=[];self.previous_fusions={};self.last_person={}
        self.seen_people=set();self.captured={};self.history_calls={};self.pair_calls={};self.groups=[]
        assigner=core.bank.assigner
        original_assign=assigner.assign
        original_history=assigner._history_evidence
        original_pair=assigner._pair_evidence
        original_solve=assigner._solve_component

        def assign(history,observations,learned=None):
            self.history_calls={};self.pair_calls={};self.groups=[]
            self.captured={} if learned is None else {k:np.asarray(v).copy() for k,v in learned.items()
                if k in ('assignment_logits','null_logits','existence_logits','birth_logits','birth_pair_logits','birth_pair_mask','update_gate')}
            result=original_assign(history,observations,learned)
            self.decision=result
            return result

        def history(q,o,app_override=None,spatial_override=None,learned_margin=None):
            value=original_history(q,o,app_override,spatial_override,learned_margin)
            app=None if q.identity_embedding is None or not o.feature_valid else (
                float(app_override) if app_override is not None else float(np.clip(q.identity_embedding@o.roi_feature,-1,1)))
            reason='ACCEPTED' if value is not None else 'OTHER_REJECTION'
            if value is None and self.config.use_appearance and app is None:reason='FEATURE_UNAVAILABLE'
            elif value is None and self.config.use_appearance and app<self.config.min_appearance_similarity and (learned_margin is None or learned_margin<self.config.learned_identity_recovery_logit):reason='APPEARANCE_REJECTION'
            elif value is None and self.config.use_geometry and q.geometry_valid and o.cross_view_fusion_eligible and spatial_override is not None and spatial_override>self.config.geometry_gate:reason='GEOMETRY_GATE_REJECTION'
            self.history_calls[q.global_id,o.key]={'accepted':value is not None,'reason':reason,
                'spatial_distance':None if spatial_override is None else float(spatial_override),
                'appearance':app,'learned_margin':learned_margin,'missed_before':q.missed,
                'accepted_reason':None if value is None else value[1]}
            return value

        def pair(left,right,birth=False,learned_probability=None):
            value=original_pair(left,right,birth=birth,learned_probability=learned_probability)
            if birth:self.pair_calls[self.observation_index[left.key],self.observation_index[right.key]]=value
            return value

        def solve(groups):
            self.groups.extend(groups)
            return original_solve(groups)

        assigner.assign=assign;assigner._history_evidence=history
        assigner._pair_evidence=pair;assigner._solve_component=solve

    def start(self,observations,labels,timestamp,history_labels):
        from worldmot.training.world_object_identity_targets import build_identity_targets
        self.obs=observations;self.labels=labels;self.timestamp=timestamp
        self.observation_index={o.key:j for j,o in enumerate(observations)}
        self.prior_labels=dict(history_labels);self.history=self.core.bank.active()
        self.pre={q.global_id:{'center':q.center.copy(),'geometry_valid':q.geometry_valid,
            'missed':q.missed,'embedding':None if q.identity_embedding is None else q.identity_embedding.copy(),
            'recent':{cam:[f.copy() for _,f in values] for cam,values in q.recent_view_features.items()}}
            for q in self.history}
        self.targets=build_identity_targets(self.history,observations,labels,history_labels,
            retention_steps=self.config.max_missed)

    def sample(self,kind,event):
        if len(self.examples[kind])<24:
            self.examples[kind].append({'scene':self.core.scene_id,'timestamp':self.timestamp,
                'frame_ordinal':self.step_index,**event})

    def finish(self,result):
        obs,labels=self.obs,self.labels;ids=self.core.bank.last_assignments
        c=self.counts;n=len(obs);m=len(self.history);raw=self.captured;t=self.targets
        current=defaultdict(list)
        for j,label in enumerate(labels):
            if label is not None:current[label].append(j)
        c['paired_timestamps']+=1;c['detections']+=n
        temporal,events=self.temporal.observe(obs,labels,ids,set(self.pre),self.prior_labels,result['tracks_3d'])
        c.update(temporal)
        for e in events:self.sample(e['error'],{k:v for k,v in e.items() if k!='error'})
        unique_history=defaultdict(list)
        for gid,label in self.prior_labels.items():
            if gid in self.pre and label is not None:unique_history[label].append(gid)
        selected_birth={tuple(x.indices) for x in self.decision.selected if x.global_id is None}
        proposals=self.core.bank.assigner.last_birth_candidate_edges
        options={tuple(x.indices) for group in self.groups for x in group if x.global_id is None}
        if raw:
            mask=raw['birth_pair_mask']|raw['birth_pair_mask'].T
            pair_probabilities=[];pair_targets=[]
            for left in range(n):
                for right in range(left+1,n):
                    if obs[left].camera_id==obs[right].camera_id:continue
                    la,lb=labels[left],labels[right]
                    kind='UNKNOWN' if la is None or lb is None else 'DIFFERENT' if la!=lb else 'NEW_SAME' if not unique_history[la] else 'EXISTING_SAME'
                    key=(left,right);f=self.funnel[kind];f['raw_cross_view_pairs']+=1
                    high=float(sigmoid(raw['birth_pair_logits'][left,right]))>=self.config.birth_pair_threshold
                    f['model_mask']+=int(mask[left,right]);f['high_score']+=int(mask[left,right] and high)
                    admissible=self.pair_calls.get(key) is not None
                    f['admissible_before_TopK']+=int(admissible)
                    f['TopK_proposal']+=int(key in proposals);f['optimizer_birth_option']+=int(key in options)
                    f['selected_joint_birth']+=int(key in selected_birth)
                    shared=ids[obs[left].key]==ids[obs[right].key]
                    f['final_shared_ID']+=int(shared)
                    if kind!='UNKNOWN' and mask[left,right]:
                        pair_probabilities.append(sigmoid(raw['birth_pair_logits'][left,right]));pair_targets.append(la==lb)
                    if kind=='DIFFERENT' and key in selected_birth:
                        self.sample('SELECTED_WRONG_JOINT_BIRTH',{'labels':[la,lb],
                            'pair_probability':float(sigmoid(raw['birth_pair_logits'][left,right])),
                            'singleton_prior_min':float(sigmoid(raw['birth_logits'][[left,right]]).min()),
                            'indices':[left,right]})
            self.calibration['cross_view_identity_pair'].add(pair_probabilities,pair_targets)
            valid=np.asarray(t['observation_valid'],dtype=bool);qvalid=np.asarray(t['history_valid'],dtype=bool)
            bp=sigmoid(raw['birth_logits'])
            self.calibration['birth_raw_sigmoid'].add(bp[valid],t['birth'][valid])
            classes=np.concatenate([raw['assignment_logits'],raw['birth_logits'][None]],axis=0)
            if n:
                probabilities=np.exp(classes-classes.max(axis=0,keepdims=True));probabilities/=probabilities.sum(axis=0,keepdims=True)
                self.calibration['birth_softmax_with_history'].add(probabilities[-1,valid],t['birth'][valid])
            current_max=raw['assignment_logits'].max(1) if n else np.zeros(m)
            self.calibration['NULL_trained_margin_sigmoid'].add(sigmoid(raw['null_logits']-current_max)[qvalid],t['null'][qvalid])
            self.calibration['existence_raw_sigmoid'].add(sigmoid(raw['existence_logits'])[qvalid],t['existence'][qvalid])
            c['raw_NULL_logit_nonnegative']+=int((raw['null_logits']>=0).sum())
            c['trained_NULL_margin_probability_ge_0_5']+=int((raw['null_logits']-current_max>=0).sum())
            for qi,q in enumerate(self.history):
                if qvalid[qi]:
                    group='missed_0' if self.pre[q.global_id]['missed']==0 else 'missed_1_5' if self.pre[q.global_id]['missed']<=5 else 'missed_6_30'
                    self.calibration['existence_'+group].add([sigmoid(raw['existence_logits'][qi])],[t['existence'][qi]])
        # An actual accepted spatial recovery at t is inspected at the next
        # processed synchronous time, including all detections of the same GT.
        for old in self.pending:
            indices=current.get(old['label'],[])
            if not indices:c['correct_spatial_recovery_next_frame_person_not_detected']+=1;continue
            c['correct_spatial_recovery_next_frame_clean_observation_opportunities']+=1
            changed=[j for j in indices if ids[obs[j].key]!=old['gid']]
            new=[j for j in changed if ids[obs[j].key] not in self.pre]
            c['correct_spatial_recovery_next_frame_any_ID_change']+=int(bool(changed))
            c['correct_spatial_recovery_next_frame_new_ID']+=int(bool(new))
            rejected=[j for j in changed if self.history_calls.get((old['gid'],obs[j].key),{}).get('reason')=='GEOMETRY_GATE_REJECTION']
            c['recovery_next_ID_change_with_correct_history_geometry_rejected']+=int(bool(rejected))
            c['recovery_next_new_ID_with_geometry_rejection_and_stale_center']+=int(bool(new and rejected and old['center_unchanged']))
            if changed:
                details={'label':old['label'],'recovered_ID':old['gid'],'new_IDs':[ids[obs[j].key] for j in changed],
                    'recovery_previous_frame':old,'geometry_rejected_indices':rejected,'new_birth_indices':new,
                    'history_evidence':[self.history_calls.get((old['gid'],obs[j].key),{}) for j in indices]}
                self.sample('RECOVERY_NEXT_FRAME_ID_BREAK',details)
                for j in changed:
                    o=obs[j];density='dense_ge_20' if n>=20 else 'sparse_lt_20'
                    reliability='eligible' if o.cross_view_fusion_eligible else 'ineligible'
                    self.strata[f'camera_{o.camera_id}/{density}/{reliability}']['recovery_next_ID_break_observations']+=1
        self.pending=[]
        for option in self.decision.selected:
            if option.reason!='VISUAL_RECOVERY_SPATIAL_CONFLICT':continue
            c['all_selected_spatial_conflict_visual_recoveries']+=1
            values={labels[j] for j in option.indices}
            if len(values)!=1 or None in values or self.prior_labels.get(option.global_id) not in values:
                c['spatial_recovery_not_clean_correctly_attributable']+=1;continue
            label=next(iter(values));q=self.core.bank.objects[option.global_id];pre=self.pre[option.global_id]
            unchanged=bool(q.geometry_valid and np.array_equal(q.center,pre['center']))
            c['correct_selected_spatial_conflict_visual_recoveries']+=1
            c['correct_spatial_recovery_center_unchanged_missed_reset_0']+=int(unchanged and q.missed==0)
            event={'label':label,'gid':option.global_id,'indices':list(option.indices),
                'timestamp':self.timestamp,'missed_before':pre['missed'],'missed_after':q.missed,
                'center_unchanged':unchanged,'old_center':pre['center'].tolist(),
                'current_centers':[obs[j].world_center.tolist() if obs[j].world_center is not None else None for j in option.indices]}
            self.pending.append(event);self.sample('CORRECT_SPATIAL_RECOVERY',event)
        for label,indices in current.items():
            for j in indices:
                o=obs[j]
                for gid in unique_history[label]:
                    evidence=self.history_calls.get((gid,o.key))
                    if evidence:
                        c['clean_correct_history_comparisons']+=1
                        c['clean_correct_history_'+evidence['reason']]+=1
                key=(o.camera_id,label);previous=self.last_person.get(key)
                if previous:
                    old,step=previous;gap=self.step_index-step
                    c['temporal_comparisons']+=1;c['temporal_ID_changes']+=int(old!=ids[o.key])
                    if gap>1:
                        suffix='within_30' if gap<=30 else 'over_30'
                        c['detection_gap_returns_'+suffix]+=1;c['detection_gap_ID_changes_'+suffix]+=int(old!=ids[o.key])
                self.last_person[key]=(ids[o.key],self.step_index)
            for left in indices:
                for right in indices:
                    if left>=right or obs[left].camera_id==obs[right].camera_id:continue
                    c['true_first_clean_detection_same_person_pairs']+=int(label not in self.seen_people)
                    c['true_first_clean_detection_splits']+=int(label not in self.seen_people and ids[obs[left].key]!=ids[obs[right].key])
            prior=self.previous_fusions.get(label)
            current_ids={ids[obs[j].key] for j in indices}
            if prior and prior[0]==self.step_index-1:
                c['previous_correct_two_view_fusion_next_frame_opportunities']+=1
                c['previous_correct_two_view_fusion_next_frame_ID_break']+=int(current_ids!={prior[1]})
            if len({obs[j].camera_id for j in indices})==2 and len(current_ids)==1:
                self.previous_fusions[label]=(self.step_index,next(iter(current_ids)))
        self.seen_people.update(current);self.step_index+=1

    def report(self):
        return {'counts':dict(self.counts),'birth_funnel':{k:dict(v) for k,v in self.funnel.items()},
            'calibration':{k:v.report() for k,v in self.calibration.items()},
            'error_strata':{k:dict(v) for k,v in self.strata.items()},'examples':dict(self.examples)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-root',required=True,type=Path)
    p.add_argument('--method',required=True);p.add_argument('--checkpoint',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);p.add_argument('--device',default='cuda:0')
    p.add_argument('--dataset-root',default='/data/DATASETS/TRACKING/JDE/VisionTrack')
    a=p.parse_args();root=a.run_root;out=a.output;out.mkdir(parents=True,exist_ok=False)
    import torch
    from worldmot.runtime.experiment_authorization import require_experiment_authorization,source_fingerprint
    require_experiment_authorization(root/'authorization_v3.json','diagnostics','train',a.method)
    from worldmot.models.world_object_update_head import WorldObjectUpdateHead
    from worldmot.tracking.world_object_config import WorldObjectConfig
    from worldmot.runtime.world_object_tracking_core import WorldObjectTrackingCore
    from worldmot.training.world_object_causal_training import CausalWorldObjectTraining
    from worldmot.training.world_object_identity_targets import match_train_identities
    from worldmot.models.gmt_detection_cache import ImmutableGMTDetectionCache
    from worldmot.data.worldobject_cache import WorldObjectCache
    from worldmot.data.worldobject_scope import load_scope_lock,validate_scene_lock
    from worldmot.data.worldobject_observations import replay_observations
    from worldmot.data.visiontrack import VisionTrack
    checkpoint_sha=sha(a.checkpoint)
    records=read(root/'audit_snapshots/epoch_archive_index.json')['records']
    record=next(x for x in records if x['method']==a.method and x['sha256']==checkpoint_sha)
    if not record['usable_as_selection_epoch_checkpoint'] or not record['resume_metadata_SHA_matched_at_capture']:
        raise ValueError('RETAINED_SOURCE_BOUND_SELECTION_WEIGHTS_REQUIRED')
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    epoch=len(state['selection']);expected=state['selection'][-1]['validation']
    raw=ImmutableGMTDetectionCache(root.parent/'gmt_detector_comparison_20261023/raw_train_v1')
    config=WorldObjectConfig(raw.feature_dim,a.method);torch.set_num_threads(2)
    model=WorldObjectUpdateHead(raw.feature_dim,128,birth_pair_chunk=config.birth_pair_chunk).to(a.device)
    model.load_state_dict(state['model'],strict=True);model.eval()
    scope=load_scope_lock(root/'scopes'/f'{a.method}.train.json','train',a.method)
    ds=VisionTrack(a.dataset_root,split='train',load_annotations=True)
    sums=Counter();diagnostics=Counter();per_scene={};observers=[];frames=0;started=time.time()
    for sid in [4,8,12,16,20,24]:
        core=WorldObjectTrackingCore(config,model,a.device);runner=CausalWorldObjectTraining(core,'predicted');runner.reset(sid)
        observer=FrozenDecisionObserver(core);observers.append(observer)
        cache=WorldObjectCache(root/'cache_train_v2',sid,raw_cache=raw,require_features=True,require_geometry=config.use_geometry)
        validate_scene_lock(scope,cache,'train');pairs={x.frame_id:x for x in ds.synchronized_pairs(sid)}
        folder=out/f'scene_{sid:02d}';folder.mkdir();times=[]
        with (folder/'tracks.jsonl').open('x') as stream:
            for frame in cache.frames():
                obs=sorted(replay_observations(frame),key=lambda x:x.key);ts=frame['timestamp']
                labels,_=match_train_identities(obs,ds.ground_truth_pair(pairs[ts]))
                observer.start(obs,labels,ts,runner.history_labels)
                result,losses=runner.step(obs,ts,labels,optimizer=None)
                observer.finish(result);sums.update(losses);diagnostics.update(runner.last_diagnostics)
                stream.write(json.dumps(result,allow_nan=False)+'\n');times.append(ts);frames+=1
        if times!=list(pairs):raise ValueError('FIXED_VAL_FULL_SCENE_COVERAGE_MISMATCH')
        report=observer.report();per_scene[str(sid)]={'timestamps':len(times),'analysis':report}
        write_json(folder/'manifest.json',{'status':'COMPLETE','method_id':a.method,'timestamps':len(times),
            'tracks_sha256':sha(folder/'tracks.jsonl'),'GT_in_model_inputs':False,'TRAIN_VAL_diagnostic_only':True})
        write_json(out/'progress.json',{'status':'RUNNING','completed_scenes':list(per_scene),'frames':frames,
            'checkpoint_sha256':checkpoint_sha,'elapsed_s':time.time()-started})
        print(json.dumps({'TRAIN_VAL_scene_complete':sid,'method':a.method,'epoch':epoch,
            'counts':{k:v for k,v in report['counts'].items() if 'recovery' in k}}),flush=True)
    mismatches={k:[v,diagnostics.get(k,0)] for k,v in expected['diagnostics'].items()
        if not k.startswith(('teacher_prefix_','predicted_prefix_')) and v!=diagnostics.get(k,0)}
    loss_differences={k:sums[k]/max(1,frames)-expected[k] for k in sums}
    counts=Counter();funnel=defaultdict(Counter);calibration=defaultdict(Calibration)
    for observer in observers:
        counts.update(observer.counts)
        for key,value in observer.funnel.items():funnel[key].update(value)
        for key,value in observer.calibration.items():calibration[key].rows+=value.rows
    report={'schema':'worldobject_v3_frozen_TRAIN_VAL_spatial_temporal_attribution_v1','status':'COMPLETE',
        'method':a.method,'selection_epoch':epoch,'checkpoint_sha256':checkpoint_sha,'checkpoint':str(a.checkpoint),
        'source_sha256':source_fingerprint(),'analysis_sha256':sha(__file__),'optimizer_used':False,'R2_modified':False,
        'TEST_read':False,'scenes':[4,8,12,16,20,24],'timestamps':frames,
        'original_epoch_prediction_diagnostics_exactly_reproduced':not mismatches,
        'original_epoch_diagnostic_mismatches':mismatches,'loss_difference_vs_original':loss_differences,
        'counts':dict(counts),'birth_funnel':{k:dict(v) for k,v in funnel.items()},
        'lifecycle_calibration':{k:v.report() for k,v in calibration.items()},'per_scene':per_scene,
        'limits':{'official_TEST_result':False,'NULL_probability':'sigmoid(null_logit-max assignment_logit), matching the trained loss; max=0 for no current detections',
            'new_same':'No matching CLEAN active predicted history; differs from physical/first clean video appearance',
            'false_Birth':'New actual ID despite a unique clean matching active history; head calibration and final decision are separately reported',
            'geometry_gate':'Rejected by the actual frozen history admissibility function after appearance checks, not inferred from spatial distance alone',
            'recovery_event':'Selected historical VISUAL_RECOVERY_SPATIAL_CONFLICT with all assigned clean detections agreeing with its prior clean identity',
            'chatter_denominator':'Immediate next processed synchronized time with clean current detections of the recovered person; no detection is counted separately'}}
    write_json(out/'attribution.json',report)
    print(json.dumps({'status':'COMPLETE','method':a.method,'epoch':epoch,'exact_original_diagnostics':not mismatches,
        'counts':dict(counts),'elapsed_s':time.time()-started}),flush=True)


if __name__=='__main__':main()
