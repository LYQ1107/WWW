"""Finite OFF-history current-group forks; original GMT solver/Bank/ID methods unchanged.

No GT, identity labels or new foundation-model calls enter inference. This is
an isolated counterfactual study, never a deployed online fusion implementation.
"""
import collections
import copy
import gzip
import json
import random
import time
import numpy as np
import torch
from omega_common import ROOT,RUNTIME,NATIVE,sha256,save_new
from omega_modules import geometry_module
from common import load_gmt,seed_all,train_data,video_records,map_records,provenance


def rng_state():
    return random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all()


def restore_rng(s):
    random.setstate(s[0]);np.random.set_state(s[1]);torch.set_rng_state(s[2]);torch.cuda.set_rng_state_all(s[3])


def main():
    torch.set_num_threads(4)
    plan_path=ROOT/'manifests/omega512_identity_counterfactual_preregistration_v1.json'
    plan=json.loads(plan_path.read_text())
    for path,h in plan['frozen_inputs'].items():assert sha256(ROOT/path)==h,path
    source=provenance(seed=plan['seed']);assert not source['source_tree_dirty']
    out=RUNTIME/'omega512_independent_v1/solver_forks';out.mkdir(parents=True,exist_ok=False)
    cohort=json.loads((ROOT/'manifests/omega_fixed_pair_cohort_v1.json').read_text())
    pair_map={p['pair_id']:p for p in cohort['pairs']}
    # Only numeric camera outputs and background validity are read, not GT labels.
    pilots=json.loads((ROOT/'results/omega512_independent_evidence_v1.json').read_text())
    expansion=json.loads((ROOT/'results/omega512_expansion_evidence_v1.json').read_text())
    numeric=json.loads((ROOT/'results/omega_numeric_evidence_v1.json').read_text())
    models={name:{} for name in ['omega512','old_vggt']}
    for name,tag in [('omega512','omega512_initial24_v1'),('old_vggt','old_vggt_initial24_v1')]:
        models[name].update({r['pair_id']:r for r in numeric['runs'][tag]['rows']})
    for name,tag in [('omega512','omega512_remaining72_v1'),('old_vggt','old_vggt_remaining72_v1')]:
        models[name].update({r['pair_id']:r for r in expansion['models'][tag]['rows']})
    valid={name:set(pilots['manifest']['summary']['C2_sift6000_mutual'][name]['pair_ids'])|set(expansion['manifest']['summary']['C2_sift6000_mutual'][name]['pair_ids']) for name in models}
    align=geometry_module('geometry_alignment');score=geometry_module('pair_score_adapter')
    from gtr.geometry.online_runner import CausalGMT,clone_instances
    from gtr.geometry.decision_trace import DecisionTrace
    from gtr.modeling.meta_arch import gtr_rcnn as tracker_module
    from gtr.evaluation.mot_evaluation import custom_instances_to_coco_json
    def global_state():
        return copy.deepcopy((tracker_module.poss_ids.poss_ids,tracker_module.old_ids.old_ids,tracker_module.old_reids.old_reids))
    def restore_global(s):
        tracker_module.poss_ids.poss_ids,tracker_module.old_ids.old_ids,tracker_module.old_reids.old_reids=copy.deepcopy(s)
    def fork(t):
        f=copy.copy(t);f.instances=[clone_instances(x) for x in t.instances]
        f.counts=copy.deepcopy(t.counts);f.reids={k:clone_instances(x) for k,x in t.reids.items()};return f
    cfg,model=load_gmt(seed=plan['seed']);assert not model.geo_runtime.active
    data=train_data();metadata={i['id']:i for i in data['images']}
    trace=DecisionTrace(model,out);trace.install();write_original=trace.write
    context=dict(scene=None,group=None,arm='OFF_PREFIX',weight=0.)
    current_decisions=[];current_matrices=[];case_outputs=[];parity=[]
    def write(stream,record):
        if stream is trace.decisions:
            record=dict(record,experiment_context=dict(context));current_decisions.append(record)
        write_original(stream,record)
    trace.write=write
    original_association=model.get_asso
    native_boxes={};frame_indices={};current_pair=None
    def association(instances,k,**kwargs):
        result=original_association(instances,k,**kwargs)
        if kwargs.get('geo_context',True) is False:return result
        if context['arm']=='OFF_PREFIX':return result
        query=instances[k];refs=[x for j,x in enumerate(instances) if j!=k]
        qs=query.decision_trace_source.detach().cpu().tolist()
        rs=[s for inst in refs for s in inst.decision_trace_source.detach().cpu().tolist()]
        raw=result[0][0];assert tuple(raw.shape)==(len(qs),len(rs))
        arm=context['arm'];weight=context['weight'];evidence=np.zeros(raw.shape,dtype=np.float32)
        legal=np.zeros(raw.shape,dtype=bool)
        if weight and len(qs) and len(rs):
            if arm=='NativePrior_age20':
                qf=query.reid_features;rf=torch.cat([x.reid_features for x in refs])
                qvalid=(torch.isfinite(qf).all(1)&(qf.norm(dim=1)>0)).cpu().numpy()
                rvalid=(torch.isfinite(rf).all(1)&(rf.norm(dim=1)>0)).cpu().numpy()
                for i,q in enumerate(qs):
                    for j,r in enumerate(rs):
                        qm,rm=metadata[q[0]],metadata[r[0]]
                        legal[i,j]=qvalid[i] and rvalid[j] and qm['view_id']!=rm['view_id'] and abs(frame_indices[qm['frame_id']]-frame_indices[rm['frame_id']])<=20
                evidence=legal.astype(np.float32)
            elif arm!='G0_native':
                name='old_vggt' if arm=='G1_old_F' else 'omega512'
                accepted=arm=='G2_omega_F_ungated' or current_pair in valid[name]
                row=models[name][current_pair]
                for mapping in row['outputs']['AB']['mappings']:
                    assert mapping['crop_left']==0 and mapping['crop_top']==0 and mapping['crop_width']==mapping['original_width'] and mapping['crop_height']==mapping['original_height']
                Fs=[]
                for order in ['AB','BA']:
                    o=row['outputs'][order];v=o.get('F_original') if name=='omega512' else o.get('F_in_physical_BA')
                    F=np.array(v) if v is not None else None
                    if F is not None and name=='omega512' and order=='BA':F=F.T
                    Fs.append(F)
                for i,q in enumerate(qs):
                    qm=metadata[q[0]]
                    for j,r in enumerate(rs):
                        rm=metadata[r[0]]
                        permitted=qm['view_id']!=rm['view_id'] and qm['frame_id']==rm['frame_id'] and qm['frame_id']==context['group']
                        if not permitted or not accepted:continue
                        # Native public detector coordinates, not crop-normalized boxes.
                        bq=native_boxes[tuple(q)];br=native_boxes[tuple(r)]
                        cq=(bq[:2]+bq[2:])/2;cr=(br[:2]+br[2:])/2
                        if not (np.isfinite(cq).all() and np.isfinite(cr).all() and np.all(cq>=0) and np.all(cr>=0) and cq[0]<qm['width'] and cq[1]<qm['height'] and cr[0]<rm['width'] and cr[1]<rm['height']):continue
                        values=[]
                        for F in Fs:
                            physical_F=F if qm['view_id']==2 else (F.T if F is not None else None)
                            values.append(float(align.soft_box_evidence(physical_F,br[None,:],bq[None,:],physical_F is not None)[0,0]))
                        evidence[i,j]=np.mean(values);legal[i,j]=True
                if arm=='G5_shuffled_gated':
                    for i in range(len(qs)):
                        indices=np.flatnonzero(legal[i]);evidence[i,indices]=np.roll(evidence[i,indices],1)
        # Exact neutral identity for zero or unreliable geometry, including tie breaks.
        adapter=score.PairScoreAdapter(enabled=True,feature_weight=weight,max_residual=1.)
        feature=raw.new_tensor(evidence);mask=raw.new_tensor(legal,dtype=torch.bool)
        adapted=[adapter.apply(x,feature,None,mask) for x in result[0]] if legal.any() else result[0]
        if weight==0 or not legal.any():assert all(a is b for a,b in zip(adapted,result[0]))
        delta=adapted[0]-raw
        assert torch.isfinite(delta).all() and (not delta.numel() or float(delta.abs().max())<=1.)
        ids=torch.cat([x.track_ids for x in refs]) if refs else raw.new_empty((0,),dtype=torch.long)
        unique=ids.unique(sorted=True)
        def track_scores(logits):
            if not refs:return logits.new_zeros((len(query),0))
            if context['group']==min(frame_indices):
                prob=model.roi_heads._activate_asso([logits])[0]
            else:prob=torch.cat(model.roi_heads._activate_asso(logits.split([len(x) for x in refs],dim=1)),dim=1)
            return prob@(ids[:,None]==unique[None,:]).float()
        current_matrices.append(dict(queries=qs,references=[dict(source=s,track=int(t)) for s,t in zip(rs,ids.tolist())],
            raw_logits=raw.cpu().tolist(),delta=delta.cpu().tolist(),legal=legal.tolist(),
            unique_tracks=unique.cpu().tolist(),before_track_scores=track_scores(raw).cpu().tolist(),
            after_track_scores=track_scores(adapted[0]).cpu().tolist(),unchanged_object=all(a is b for a,b in zip(adapted,result[0]))))
        return (adapted,)+tuple(result[1:])
    model.get_asso=association
    variants=[('G0_native',0.)]+[(arm,w) for arm in plan['arms'][1:] for w in [.1,.3]]
    cases_by_scene=collections.defaultdict(dict)
    for case in plan['cases']:cases_by_scene[case['scene']][case['group']]=case
    stream=(out/'case_results.jsonl').open('x')
    started=time.perf_counter()
    try:
        for scene,cases in sorted(cases_by_scene.items()):
            native=NATIVE/f'scene{scene}_full_v1'
            receipt=next(r for r in cohort['native_source_receipts'] if r['path']==str(native/'manifest.json'))
            for fn in ['predictions.json','detector_provenance.jsonl.gz']:
                assert sha256(native/fn)==receipt['artifacts'][fn]
            expected=json.loads((native/'predictions.json').read_text())
            with gzip.open(native/'detector_provenance.jsonl.gz','rt') as f:
                for line in f:
                    r=json.loads(line);x,y,w,h=r['detection']['bbox'];native_boxes[tuple(r['source'])]=np.array([x,y,x+w,y+h])
            records=video_records(data,scene,max(cases));groups=collections.defaultdict(list)
            for r in records:groups[r['frame_id']].append(r)
            frame_indices={frame:i for i,frame in enumerate(sorted(groups))}
            seed_all(plan['seed']);tracker=CausalGMT(model)
            for frame,group in sorted(groups.items()):
                context.update(scene=scene,group=frame,arm='OFF_PREFIX',weight=0.)
                mapped=map_records(cfg,group)
                if frame not in cases:
                    tracker.step(mapped)
                    current_decisions.clear()
                else:
                    before=fork(tracker);rng_before=rng_state();globals_before=global_state();counts_before=copy.deepcopy(trace.counts)
                    inference_original=model.inference;cache={}
                    def capture(*args,**kwargs):
                        values=inference_original(*args,**kwargs)
                        cache[args[0][0]['image_id']]=[clone_instances(x) for x in values]
                        return values
                    model.inference=capture;current_decisions.clear();baseline=tracker.step(mapped);model.inference=inference_original
                    base_predictions=[p for r,o in zip(group,baseline) for p in custom_instances_to_coco_json(o['instances'].to('cpu'),r['image_id'])]
                    image_ids={r['image_id'] for r in group};sealed=[p for p in expected if p['image_id'] in image_ids]
                    assert base_predictions==sealed,('native case prediction parity failed',scene,frame)
                    # Check original detector coordinates for every current proposal.
                    for image,values in cache.items():
                        inst=values[0];h,w=inst.image_size;meta=metadata[image]
                        b=inst.pred_boxes.tensor.detach().cpu().numpy()*np.array([meta['width']/w,meta['height']/h]*2)
                        for s,box in zip(inst.decision_trace_source.cpu().tolist(),b):
                            np.testing.assert_allclose(box,native_boxes[tuple(s)],rtol=1e-6,atol=1e-3)
                    rng_after=rng_state();globals_after=global_state();counts_after=copy.deepcopy(trace.counts)
                    base_decisions=copy.deepcopy(current_decisions);current_pair=cases[frame]['pair_id']
                    for arm,weight in variants:
                        branch=fork(before);restore_rng(rng_before);restore_global(globals_before);trace.counts=copy.deepcopy(counts_before)
                        context.update(arm=arm,weight=weight);current_decisions.clear();current_matrices.clear()
                        def cached_inference(*args,**kwargs):return [clone_instances(x) for x in cache[args[0][0]['image_id']]]
                        model.inference=cached_inference;outputs=branch.step(mapped);model.inference=inference_original
                        predictions=[p for r,o in zip(group,outputs) for p in custom_instances_to_coco_json(o['instances'].to('cpu'),r['image_id'])]
                        if weight==0:
                            assert predictions==base_predictions,('zero fork parity failed',scene,frame)
                            a=[(d['stage'],d['queries'],d['accepted_track_ids'],d['final_track_ids']) for d in current_decisions]
                            b=[(d['stage'],d['queries'],d['accepted_track_ids'],d['final_track_ids']) for d in base_decisions]
                            assert a==b,('zero solver/Bank parity failed',scene,frame)
                            parity.append(dict(pair_id=current_pair,native_predictions_exact=True,zero_predictions_exact=True,zero_solver_and_Bank_exact=True))
                        item=dict(pair_id=current_pair,scene=scene,group=frame,arm=arm,weight=weight,
                            baseline_public_predictions=base_predictions,public_predictions=predictions,
                            baseline_decisions=base_decisions,decisions=copy.deepcopy(current_decisions),matrices=copy.deepcopy(current_matrices))
                        stream.write(json.dumps(item,allow_nan=False)+'\n');stream.flush();case_outputs.append(dict(pair_id=current_pair,arm=arm,weight=weight))
                        del branch
                    restore_rng(rng_after);restore_global(globals_after);trace.counts=counts_after
                    context.update(arm='OFF_PREFIX',weight=0.);del before
                    print(json.dumps(dict(scene=scene,group=frame,pair_id=current_pair,variants=len(variants),parity='PASS')),flush=True)
                if frame%100==0:print(json.dumps(dict(scene=scene,prefix_group=frame)),flush=True)
        stream.close();trace.close()
        save_new(out/'manifest.json',dict(status='FINITE_SOLVER_FORKS_COMPLETE',cases=len(plan['cases']),variants_per_case=len(variants),
            parity=parity,case_outputs=case_outputs,source=source,registration=dict(path=str(plan_path),sha256=sha256(plan_path)),
            output=dict(path=str(out/'case_results.jsonl'),sha256=sha256(out/'case_results.jsonl')),
            no_GT_or_identity_labels_input=True,no_foundation_model_calls=True,no_training=True,branch_states_discarded=True,
            no_production_online_integration=True,wall_after_model_setup_seconds=time.perf_counter()-started,
            CUDA_allocator_peak_bytes=torch.cuda.max_memory_allocated(),CUDA_reserved_peak_bytes=torch.cuda.max_memory_reserved()))
    finally:
        if not stream.closed:stream.close()
        model.get_asso=original_association

if __name__=='__main__':main()
