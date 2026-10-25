#!/usr/bin/env python3
"""TRAIN-only diagnostic: normal learned decision BEFORE teacher correction.

The existing teacher-prefix counts are after current GT correction. This
observer instead clones the bank containing teacher-corrected PAST only, makes
the ordinary GT-free current assignment, and discards the clone. No optimizer.
Uses the already registered 129-time prefix per fixed VAL scene, not TEST.
"""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import numpy as np
from summarize_worldobject_v3 import read,sha,write_json


def shadow_prediction(core,observations,timestamp):
    outputs=core.forward_evidence(observations,timestamp,training=False)
    consumed={'assignment_logits','null_logits','existence_logits','birth_logits','update_gate',
        'memory_delta','birth_pair_logits','birth_pair_mask'}
    learned=None if outputs is None else {key:(value.detach() if key=='memory_delta' else value.detach().cpu().numpy())
        for key,value in outputs.items() if key in consumed}
    bank=copy.deepcopy(core.bank)
    bank.step(observations,timestamp,learned)
    return bank.last_assignments


def outcome(observations,labels,assignments,history_labels,active):
    counts=Counter();known={history_labels.get(gid) for gid in active}-{None}
    for o,label in zip(observations,labels):
        if label is None:continue
        gid=assignments[o.key];counts['clean_observations']+=1
        if gid in active:
            counts['correct_clean_historical_reuse' if history_labels.get(gid)==label else 'wrong_or_unlabelled_historical_reuse']+=1
        else:
            counts['new_ID_despite_clean_active_history' if label in known else 'new_ID_without_clean_active_history']+=1
    for a in range(len(observations)):
        for b in range(a+1,len(observations)):
            if labels[a] is None or labels[b] is None or observations[a].camera_id==observations[b].camera_id:continue
            same=labels[a]==labels[b];shared=assignments[observations[a].key]==assignments[observations[b].key]
            counts['same_person_pairs' if same else 'different_person_pairs']+=1
            counts['same_person_splits' if same and not shared else 'different_person_merges' if not same and shared else 'correct_pairs']+=1
    return counts


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:0');a=p.parse_args()
    import torch
    from worldmot.runtime.experiment_authorization import require_experiment_authorization,source_fingerprint
    from worldmot.models.world_object_update_head import WorldObjectUpdateHead
    from worldmot.runtime.world_object_tracking_core import WorldObjectTrackingCore
    from worldmot.tracking.world_object_config import WorldObjectConfig
    from worldmot.training.world_object_causal_training import CausalWorldObjectTraining
    from worldmot.training.world_object_identity_targets import match_train_identities
    from worldmot.data.visiontrack import VisionTrack
    from worldmot.data.worldobject_cache import WorldObjectCache
    from worldmot.data.worldobject_observations import replay_observations
    from worldmot.models.gmt_detection_cache import ImmutableGMTDetectionCache
    torch.set_num_threads(2)
    raw=ImmutableGMTDetectionCache(a.root.parent/'gmt_detector_comparison_20261023/raw_train_v1')
    ds=VisionTrack('/data/DATASETS/TRACKING/JDE/VisionTrack',split='train',load_annotations=True)
    records=read(a.root/'audit_snapshots/epoch_archive_index.json')['records'];rows=[]
    for method in ('FULL_WORLD_OBJECT_V3','HEAD_ONLY_NO_GEOMETRY_V3'):
        require_experiment_authorization(a.root/'authorization_v3.json','diagnostics','train',method)
        record=next(x for x in records if x['method']==method and x['usable_as_selection_epoch_checkpoint'] and x['selection_epochs']==1)
        if sha(record['path'])!=record['sha256'] or not record['resume_metadata_SHA_matched_at_capture']:raise ValueError('SOURCE_BOUND_RETAINED_WEIGHTS_REQUIRED')
        state=torch.load(record['path'],map_location='cpu',weights_only=False)
        config=WorldObjectConfig(raw.feature_dim,method);model=WorldObjectUpdateHead(raw.feature_dim,128).to(a.device)
        model.load_state_dict(state['model'],strict=True);model.eval()
        totals={mode:Counter() for mode in ('predicted_past','teacher_past_normal_current','teacher_corrected_current_ceiling')};scenes=[]
        for sid in [4,8,12,16,20,24]:
            pred=CausalWorldObjectTraining(WorldObjectTrackingCore(config,model,a.device),'predicted');pred.reset(sid)
            teacher=CausalWorldObjectTraining(WorldObjectTrackingCore(config,model,a.device),'teacher_forced');teacher.reset(sid)
            cache=WorldObjectCache(a.root/'cache_train_v2',sid,raw_cache=raw,require_features=True,require_geometry=config.use_geometry)
            pairs={p.frame_id:p for p in ds.synchronized_pairs(sid)};counts={key:Counter() for key in totals};n=0
            for frame in cache.frames():
                if n==129:break
                observations=sorted(replay_observations(frame),key=lambda o:o.key);timestamp=frame['timestamp']
                labels,_=match_train_identities(observations,ds.ground_truth_pair(pairs[timestamp]))
                pre_pred=dict(pred.history_labels);active_pred={q.global_id for q in pred.core.bank.active()}
                pred.step(observations,timestamp,labels,optimizer=None)
                counts['predicted_past'].update(outcome(observations,labels,pred.core.bank.last_assignments,pre_pred,active_pred))
                pre_teacher=dict(teacher.history_labels);active_teacher={q.global_id for q in teacher.core.bank.active()}
                shadow=shadow_prediction(teacher.core,observations,timestamp)
                counts['teacher_past_normal_current'].update(outcome(observations,labels,shadow,pre_teacher,active_teacher))
                teacher.step(observations,timestamp,labels,optimizer=None)
                counts['teacher_corrected_current_ceiling'].update(outcome(observations,labels,teacher.core.bank.last_assignments,pre_teacher,active_teacher))
                n+=1
            for mode,value in counts.items():totals[mode].update(value)
            scenes.append({'scene':sid,'prefix_timestamps':n,'counts':{k:dict(v) for k,v in counts.items()}})
            print(json.dumps({'method':method,'teacher_past_shadow_scene_complete':sid,'prefix':n}),flush=True)
        expected=state['selection'][0]['validation']['diagnostics']
        reproducibility={metric:totals['predicted_past'][key]==expected['predicted_prefix_'+metric] for key,metric in
            [('same_person_splits','same_person_split'),('different_person_merges','false_cross_view_merge')]}
        rows.append({'method':method,'selection_epoch':1,'checkpoint_sha256':record['sha256'],
            'prefix_per_scene':129,'total_prefix_timestamps':sum(x['prefix_timestamps'] for x in scenes),
            'counts':{k:dict(v) for k,v in totals.items()},'per_scene':scenes,
            'original_predicted_prefix_counts_reproduced':reproducibility})
    write_json(a.output,{'schema':'worldobject_v3_teacher_corrected_past_normal_current_shadow_v1','status':'COMPLETE',
        'source_sha256':source_fingerprint(),'analysis_sha256':sha(__file__),'replays':rows,'optimizer_used':False,'TEST_read':False,
        'R2_bank_state_modified':False,'current_GT_in_shadow_decision':False,'past_GT_role':'TRAIN-only teacher-corrected historical grouping; current model decision is ordinary frozen assignment before current teacher correction',
        'scope_limit':'129-time prefix per six fixed TRAIN VAL scenes, matching the original trainer teacher-prefix scope. This is not full-video or official TEST accuracy.',
        'missing_NoGeometry_selection1_weights':True,'parameter_selection':False})


if __name__=='__main__':main()
