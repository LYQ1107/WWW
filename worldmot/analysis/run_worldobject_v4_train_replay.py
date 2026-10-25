#!/usr/bin/env python3
"""Finite frozen-head TRAIN replay. No optimizer, TEST option or cache writes.

Runs independent closed loops from scene origin, optionally frozen-C0-past
single-step counterfactuals. Each method sees the same current detector outputs.
"""
import argparse
from collections import Counter
import copy
from dataclasses import replace
import importlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import numpy as np
from worldobject_v4_audit_common import (VAL_SCENES,LEGAL_VAL,IdentityAudit,HistoryEvidenceObserver,
    code_fingerprint,write_json,contrast_events,visual_only,visual_from_detections,timing_summary)

BASE='b448af4a7dfdf738aeab0c9dd5bc01e9a875beac'


def legacy_core(repo,workspace):
    package=workspace/'frozen_r2_package';package.mkdir(parents=True,exist_ok=False)
    data=subprocess.run(['git','archive',BASE,'worldmot/worldmot'],cwd=repo,check=True,stdout=subprocess.PIPE).stdout
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        for item in archive.getmembers():
            if item.isfile() and item.name.endswith('.py'):
                path=package/Path(item.name).relative_to('worldmot/worldmot')
                path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(archive.extractfile(item).read())
    spec=importlib.util.spec_from_file_location('frozen_r2_package',package/'__init__.py',submodule_search_locations=[str(package)])
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    core=importlib.import_module('frozen_r2_package.runtime.world_object_tracking_core').WorldObjectTrackingCore
    cfg=importlib.import_module('frozen_r2_package.tracking.world_object_config').WorldObjectConfig
    inputs=importlib.import_module('frozen_r2_package.models.world_object_model_inputs').model_inputs
    return core(cfg(1024)),inputs


def consumed_evidence(outputs):
    if outputs is None:return None
    keys={'assignment_logits','null_logits','existence_logits','birth_logits','update_gate','memory_delta',
          'birth_pair_logits','birth_pair_mask'}
    return {k:(v.detach() if k=='memory_delta' else v.detach().cpu().numpy()) for k,v in outputs.items() if k in keys}


def copy_bank(bank,config):
    from worldmot.tracking.persistent_world_object_bank import PersistentWorldObjectBank
    cloned=PersistentWorldObjectBank(config)
    cloned.objects=copy.deepcopy(bank.objects);cloned.next_id=bank.next_id;cloned.last_timestamp=bank.last_timestamp
    return cloned


def current_events(observations,labels,assigned,prior_last=None):
    out={}
    for i,a in enumerate(observations):
        if labels[i] is None:continue
        for j,b in enumerate(observations[i+1:],i+1):
            if labels[j] is None or a.camera_id==b.camera_id:continue
            same=labels[i]==labels[j]
            out['same_person' if same else 'different_person',(a.key,b.key)]=(assigned[a.key]==assigned[b.key])==same
        if prior_last and (a.camera_id,labels[i]) in prior_last:
            out['temporal',(a.camera_id,labels[i])]=assigned[a.key]==prior_last[a.camera_id,labels[i]][0]
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('--r2-root',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);p.add_argument('--scene',required=True,type=int,choices=VAL_SCENES)
    p.add_argument('--device',default='cpu');p.add_argument('--with-single-step',action='store_true')
    p.add_argument('--scope',choices=['all','state','factorial'],default='all');p.add_argument('--limit',type=int)
    p.add_argument('--retained-mechanism',action='store_true')
    a=p.parse_args();repo=Path(__file__).resolve().parents[2];fp=code_fingerprint(repo)
    a.output.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    import torch
    torch.set_num_threads(1)
    from worldmot.data.visiontrack import VisionTrack
    from worldmot.data.worldobject_cache import WorldObjectCache,sha256
    from worldmot.data.worldobject_observations import replay_observations
    from worldmot.models.gmt_detection_cache import ImmutableGMTDetectionCache
    from worldmot.models.world_object_v4_checkpoint import load_frozen_r2_head
    from worldmot.runtime.world_object_tracking_core import WorldObjectTrackingCore
    from worldmot.runtime.omega_evidence_scheduler import OmegaEvidenceScheduler,REQUEST_OMEGA
    from worldmot.tracking.world_object_v4_variants import factorial_configs,state_configs
    from worldmot.tracking.world_object_config import WorldObjectConfig
    from worldmot.training.world_object_identity_targets import match_train_identities
    from worldmot.models.world_object_model_inputs import model_inputs
    raw=ImmutableGMTDetectionCache(a.r2_root.parent/'gmt_detector_comparison_20261023/raw_train_v1')
    if a.retained_mechanism:
        if a.scene!=24 or a.limit!=705 or a.with_single_step:
            raise ValueError('EXACT_SCENE24_705_TIME_RETAINED_MECHANISM_SCOPE_REQUIRED')
        from worldmot.models.world_object_v4_checkpoint import load_retained_epoch1_mechanism_head
        model,provenance=load_retained_epoch1_mechanism_head(a.r2_root,a.device)
    else:model,provenance=load_frozen_r2_head(a.r2_root/'train/FULL_WORLD_OBJECT_V3/train_gate.json',raw.lock,a.device)
    if raw.feature_dim!=1024:raise ValueError('FROZEN_1024D_ROI_REQUIRED')
    if any(p.requires_grad for p in model.parameters()):raise ValueError('FROZEN_MODEL_REQUIRED')
    ds=VisionTrack('/data/DATASETS/TRACKING/JDE/VisionTrack',split='train',load_annotations=True)
    pairs={p.frame_id:p for p in ds.synchronized_pairs(a.scene)}
    cache=WorldObjectCache(a.r2_root/'cache_train_v2',a.scene,raw_cache=raw,require_features=True,require_geometry=True)
    states=state_configs(raw.feature_dim);factors=factorial_configs(raw.feature_dim)
    configs={'C0':states['C0']}
    if a.scope in ('all','state'):configs.update(states)
    if a.scope in ('all','factorial'):configs.update({k:v for k,v in factors.items() if k!='G1_1_G2_1_G3_1'})
    ng_model=None;ng_provenance=None
    if a.retained_mechanism:configs={key:states[key] for key in ('C0','C3','C4')}
    else:
        ng_model,ng_provenance=load_frozen_r2_head(a.r2_root/'train/FULL_NO_GEOMETRY_V3/train_gate.json',raw.lock,a.device,
                                                method='FULL_NO_GEOMETRY_V3')
        configs['R2_NO_GEOMETRY_REFERENCE']=WorldObjectConfig(raw.feature_dim,method_id='FULL_NO_GEOMETRY_V3')
    cores={k:WorldObjectTrackingCore(c,ng_model if k=='R2_NO_GEOMETRY_REFERENCE' else model,a.device) for k,c in configs.items()}
    observers={k:HistoryEvidenceObserver(c.bank.assigner) for k,c in cores.items()}
    audits={k:IdentityAudit() for k in configs};contrasts={k:Counter() for k in configs if k!='C0'}
    timings={k:[] for k in configs};search={k:Counter() for k in configs}
    scheduler=OmegaEvidenceScheduler(configs['C5']) if 'C5' in configs else None
    legacy,old_inputs=legacy_core(repo,a.output);legacy.model=model;legacy.device=a.device;legacy.reset(a.scene)
    for core in cores.values():core.reset(a.scene)
    streams={};folders={}
    for key in configs:
        folders[key]=a.output/key/f'scene_{a.scene:02d}';folders[key].mkdir(parents=True)
        streams[key]=(folders[key]/'tracks.jsonl').open('x')
    b1={key:Counter() for key in factors};b1_changes={key:Counter() for key in factors};b1_time=[]
    b1_decisions={key:Counter() for key in factors}
    factorial_edges={}
    for key in factors:
        if key=='G4_OFF':continue
        for axis in (1,2,3):
            if f'G{axis}_0' not in key:continue
            other=key.replace(f'G{axis}_0',f'G{axis}_1')
            left='C0' if key=='G1_1_G2_1_G3_1' else key
            right='C0' if other=='G1_1_G2_1_G3_1' else other
            if left in cores and right in cores:factorial_edges[left+'__TO__'+right]=Counter()
    input_proofs=[];exact_frames=0;request_records=[];examples=[];paired_examples=[];example_counts=Counter()
    try:
        frames=iter(cache.frames())
        for index,timestamp in enumerate(sorted(pairs)):
            if a.limit is not None and index>=a.limit:break
            scheduled_visual=None;request=None
            if scheduler:
                # Decide using the raw visual cache BEFORE requesting the
                # current record from the shared Ω cache iterator.
                detections=raw.get(a.scene,timestamp,image_ids={v:r.image_id for v,r in pairs[timestamp].images.items()})
                scheduled_visual=visual_from_detections(detections,timestamp)
                cores['C5'].bank.predict(timestamp)
                request=scheduler.decide(scheduled_visual,cores['C5'].bank.active(),timestamp)
            frame=next(frames)
            if frame['timestamp']!=timestamp:raise ValueError('GEOMETRY_CACHE_TIMESTAMP_DIFFERENCE')
            observations=sorted(replay_observations(frame),key=lambda o:o.key)
            input_proofs.append({'timestamp':timestamp,**frame['source']})
            before=cores['C0'].bank
            base_inputs=model_inputs(before.active(),observations,configs['C0'],timestamp,a.device)
            old=old_inputs(legacy.bank.active(),observations,legacy.config,timestamp,a.device)
            for left,right in zip(base_inputs,old):
                if not torch.equal(left,right):raise AssertionError('C0_R2_HEAD_INPUT_DIFFERENCE')
            head_started=time.perf_counter()
            with torch.inference_mode():outputs=cores['C0'].forward_evidence(observations,timestamp)
            if a.device.startswith('cuda'):torch.cuda.synchronize()
            c0_head_s=time.perf_counter()-head_started
            learned=consumed_evidence(outputs)
            prior_last=dict(audits['C0'].last)
            cf_banks={};cf_observers={}
            if a.with_single_step:
                t=time.perf_counter()
                for key,cfg in factors.items():
                    clone=copy_bank(before,cfg)
                    cf_observers[key]=HistoryEvidenceObserver(clone.assigner)
                    cf_evidence=learned
                    if key=='G4_OFF':
                        with torch.inference_mode():out=model(*model_inputs(before.active(),observations,cfg,timestamp,a.device),use_geometry=False)
                        cf_evidence=consumed_evidence(out)
                    clone.step(observations,timestamp,cf_evidence);cf_banks[key]=clone
                b1_time.append(time.perf_counter()-t)
            results={};active_obs={}
            for key,core in cores.items():
                audits[key].begin(core.bank);observers[key].calls={}
                active_obs[key]=observations
                if key in ('C4','C5'):
                    from worldmot.runtime.world_object_geometry_context import with_alignment_context
                    active_obs[key]=with_alignment_context(observations,frame['record'].get('alignment',{}))
                if key=='C5':
                    # Both streams must contain the same boxes, scores and ROIs.
                    if [o.key for o in scheduled_visual]!=[o.key for o in observations]:raise ValueError('SCHEDULED_VISUAL_CANDIDATE_DIFFERENCE')
                    for v,g in zip(scheduled_visual,observations):
                        if not np.array_equal(v.bbox_xywh,g.bbox_xywh) or v.score!=g.score or not np.array_equal(v.roi_feature,g.roi_feature):
                            raise ValueError('SCHEDULED_VISUAL_INPUT_DIFFERENCE')
                    if request.action!=REQUEST_OMEGA:active_obs[key]=scheduled_visual
                    else:scheduler.observe_alignment(timestamp,bool(frame['record'].get('alignment',{}).get('success',False)))
                    request_records.append({'timestamp':timestamp,'action':request.action,'reasons':list(request.reasons),
                        'cache_alignment_provenance':'cached world coordinates were produced by the original continuous aligner',
                        'sparse_live_reactivation_alignment_revalidated':False})
                t=time.perf_counter()
                results[key]=core.step(active_obs[key],timestamp,evidence=outputs if key=='C0' else None)
                if a.device.startswith('cuda'):torch.cuda.synchronize()
                timing={**core.last_timings,'tracking_core_wall_s':time.perf_counter()-t}
                if key=='C0':
                    timing['head_and_evidence_transfer_s']+=c0_head_s;timing['tracking_core_wall_s']+=c0_head_s
                timings[key].append(timing)
                search[key].update({k:int(results[key]['bank'].get(k,0)) for k in ('candidate_comparisons','expanded_states','births')})
                streams[key].write(json.dumps(results[key],allow_nan=False)+'\n')
            legacy_result=legacy.step(observations,timestamp)
            if results['C0']!=legacy_result:raise AssertionError('C0_R2_DECISION_DIFFERENCE')
            for gid,q in cores['C0'].bank.objects.items():
                for name in ('center','covariance','identity_embedding','support_points'):
                    if not np.array_equal(getattr(q,name),getattr(legacy.bank.objects[gid],name),equal_nan=True):
                        raise AssertionError('C0_R2_STATE_DIFFERENCE_'+name)
            exact_frames+=1
            # GT is read only after all current learned decisions and requests.
            labels,label_audit=match_train_identities(observations,ds.ground_truth_pair(pairs[timestamp]))
            for key,core in cores.items():
                audits[key].observe(core.bank,active_obs[key],labels,observers[key].calls)
            for variant in contrasts:
                def sample_change(event_key,corrected,variant=variant):
                    group=(variant,event_key[0],corrected)
                    if example_counts[group]>=8:return
                    example_counts[group]+=1
                    kind,actors=event_key
                    indices=([i for i,o in enumerate(observations) if (o.camera_id,labels[i])==actors]
                        if kind=='temporal' else [i for i,o in enumerate(observations) if o.key==actors]
                        if kind=='historical' else [i for i,o in enumerate(observations) if o.key in actors])
                    evidence=[]
                    for i in indices:
                        o=observations[i];base_id=cores['C0'].bank.last_assignments[o.key];variant_id=cores[variant].bank.last_assignments[o.key]
                        q=cores[variant].bank.objects[variant_id]
                        evidence.append({'camera':o.camera_id,'detection_id':o.detection_id,'GT_label_posthoc':labels[i],
                            'bbox_xywh':o.bbox_xywh.tolist(),'C0_ID':base_id,'variant_ID':variant_id,
                            'current_geometry_eligible':o.cross_view_fusion_eligible,
                            'current_world_center':o.world_center.tolist() if np.isfinite(o.world_center).all() else None,
                            'variant_decision':q.decision_status,'variant_spatial_status':q.spatial_state_status,
                            'variant_identity_update_status':None if q.identity_state is None else q.identity_state.identity_update_status})
                    paired_examples.append({'scene':a.scene,'timestamp':timestamp,'variant':variant,
                        'kind':kind,'effect':'correction' if corrected else 'regression','actors':evidence,
                        'scope':'posthoc TRAIN attribution; not identity labels supplied to runtime'})
                contrast_events(audits['C0'].events,audits[variant].events,contrasts[variant],sample_change)
            for key,counter in factorial_edges.items():
                left,right=key.split('__TO__');contrast_events(audits[left].events,audits[right].events,counter)
            if a.with_single_step:
                reference=current_events(observations,labels,cores['C0'].bank.last_assignments,prior_last)
                for key,clone in cf_banks.items():
                    events=current_events(observations,labels,clone.last_assignments,prior_last)
                    contrast_events(reference,events,b1[key])
                    probe=IdentityAudit();baseline_audit=audits['C0']
                    probe.active=set(baseline_audit.active);probe.prior=copy.deepcopy(baseline_audit.prior)
                    probe.history_people.update(copy.deepcopy(baseline_audit.prior));probe.last=dict(prior_last)
                    probe.old_spatial=baseline_audit.old_spatial;probe.old_identity=baseline_audit.old_identity
                    probe.step_index=baseline_audit.step_index-1
                    probe.observe(clone,observations,labels,cf_observers[key].calls)
                    b1_decisions[key].update(probe.counts)
                    if not configs['C0'].enable_motion_prediction:
                        changed=sum(not np.array_equal(q.center,cores['C0'].bank.objects[gid].center,equal_nan=True)
                            or not np.array_equal(q.covariance,cores['C0'].bank.objects[gid].covariance,equal_nan=True)
                            or q.geometry_valid!=cores['C0'].bank.objects[gid].geometry_valid
                            for gid,q in clone.objects.items() if gid in cores['C0'].bank.objects)
                        b1_changes[key]['post_update_common_ID_spatial_state_differences']+=changed
                    if key=='G1_1_G2_1_G3_0' and clone.last_assignments!=cores['C0'].bank.last_assignments:
                        raise AssertionError('G3_MUST_NOT_CHANGE_SAME_STEP_ASSIGNMENT')
            if timestamp in (704,705) and a.scene==24:
                examples.append({'timestamp':timestamp,'scope':'TRAIN_scene24_mechanism_probe_not_TEST',
                    'assignments':{k:[{'camera':o.camera_id,'detection_id':o.detection_id,'GT_label_posthoc':label,
                        'ID':cores[k].bank.last_assignments[o.key]} for o,label in zip(observations,labels)] for k in cores}})
            if (index+1)%100==0:print(json.dumps({'scene':a.scene,'completed_timestamps':index+1,'variants':list(cores)}),flush=True)
        expected=len(pairs) if a.limit is None else min(a.limit,len(pairs))
        if exact_frames!=expected:raise ValueError('INCOMPLETE_TRAIN_VAL_SCENE')
    finally:
        for stream in streams.values():stream.close()
    if code_fingerprint(repo)!=fp:raise ValueError('V4_SOURCE_CHANGED_DURING_RUN')
    for key in configs:
        folder=folders[key]
        write_json(folder/'manifest.json',{'status':'COMPLETE','scene':a.scene,'timestamps':exact_frames,
            'full_video':a.limit is None,'GT_in_inference':False,'tracks_sha256':sha256(folder/'tracks.jsonl'),
            'source_sha256':fp,'config':configs[key].to_dict(),
            'frozen_head':ng_provenance if key=='R2_NO_GEOMETRY_REFERENCE' else provenance,'variant':key,
            'cache_replay_only':True,'TEST_read':False})
    report={'schema':'worldmot_v4_frozen_TRAIN_scene_replay_v1','status':'COMPLETE' if a.limit is None else 'LIMITED_PREFLIGHT',
        'scene':a.scene,'legal_original_GT':a.scene in LEGAL_VAL,'timestamps':exact_frames,'source_sha256':fp,
        'checkpoint':provenance,'separately_trained_NoGeometry_reference_checkpoint':ng_provenance,
        'optimizer_used':False,'TEST_read':False,'C0_R2_exact_frames':exact_frames,
        'closed_loop':{k:audits[k].report() for k in audits},'corrections_and_regressions':{k:dict(v) for k,v in contrasts.items()},
        'factorial_single_factor_edges':{k:dict(v) for k,v in factorial_edges.items()},
        'B1_fixed_C0_past':{'counts':{k:dict(v) for k,v in b1.items()},'spatial_state_changes':{k:dict(v) for k,v in b1_changes.items()},
            'conditional_current_decision_diagnostics':{k:dict(v) for k,v in b1_decisions.items()},
            'counterfactual_states_propagated':False,'same_step_G3_assignment_invariance_checked':a.with_single_step,
            'official_tracking_score':None,'reason':'independent single-step decisions do not form a causal tracking video'},
        'timings':{k:timing_summary(v) for k,v in timings.items()},'search':{k:dict(v) for k,v in search.items()},
        'B1_total_wall_s':float(sum(b1_time)),'worker_total_wall_s':time.perf_counter()-started,
        'runtime_scope':'cache replay; NOT live RGB-to-ID FPS','requests':request_records,
        'C5_scope_limit':'request-count/state-consumption simulator only; original cache has continuous alignment; sparse live keyframe alignment requires separate LIVE validation',
        'mechanism_t704_t705':examples,'input_proofs':input_proofs}
    report['paired_correction_and_failure_examples']=paired_examples
    if a.retained_mechanism:
        assertions={'C0_original_158_to_162_found':False,'variants':{}}
        if len(examples)==2:
            earlier,later=examples
            candidate=[x for x in earlier['assignments']['C0'] if x['camera']==2 and x['ID']==158 and x['GT_label_posthoc'] is not None]
            for actor in candidate:
                label=actor['GT_label_posthoc']
                next_actor=next((x for x in later['assignments']['C0'] if x['camera']==2 and x['GT_label_posthoc']==label),None)
                if next_actor is None or next_actor['ID']!=162:continue
                assertions['C0_original_158_to_162_found']=True;assertions['GT_label_posthoc']=label
                for variant in ('C3','C4'):
                    a_actor=next((x for x in earlier['assignments'][variant] if x['camera']==2 and x['GT_label_posthoc']==label),None)
                    b_actor=next((x for x in later['assignments'][variant] if x['camera']==2 and x['GT_label_posthoc']==label),None)
                    same=bool(a_actor and b_actor and a_actor['ID']==b_actor['ID'])
                    wrong=bool(b_actor and any(x['ID']==b_actor['ID'] and x['GT_label_posthoc'] not in (None,label) for x in later['assignments'][variant]))
                    assertions['variants'][variant]={'recovered_camera_keeps_ID':same,'different_current_person_merged':wrong,
                        'passes_targeted_mechanism':same and not wrong,'ID_at704':a_actor['ID'] if a_actor else None,'ID_at705':b_actor['ID'] if b_actor else None}
        report['retained_original_case_assertions']=assertions
        report['status']='COMPLETE_TRAIN_ORIGINAL_CASE_MECHANISM_ONLY'
    write_json(a.output/'replay_report.json',report)
    print(json.dumps({'scene':a.scene,'status':report['status'],'wall_s':report['worker_total_wall_s']}),flush=True)


if __name__=='__main__':main()
