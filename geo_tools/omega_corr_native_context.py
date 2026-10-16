"""Recover original native context scores from metadata RNG schedules, never re-track TRAIN."""
import argparse
import collections
import gzip
import json
import random
import time
import numpy as np
import torch
from omega_corr_common import ROOT,RUNTIME,NATIVE,DATA,sha256,save_new
from common import load_gmt,seed_all,train_data,video_records,map_records,provenance


def lines(path):
    with gzip.open(path,'rt') as f:
        for line in f:yield json.loads(line)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true');a=parser.parse_args()
    from detectron2.structures import Instances,Boxes
    from gtr.geometry.online_runner import clone_instances
    from gtr.modeling.meta_arch.custom_rcnn import CustomRCNN
    from gtr.evaluation.mot_evaluation import custom_instances_to_coco_json
    from gtr.geometry.pair_score_adapter import PairScoreAdapter
    from scipy.optimize import linear_sum_assignment
    source=provenance(seed=20261014);assert not source['source_tree_dirty'];source['used_cache']=not a.preflight;source['cache_scope']='verified lossless preflight proposals and sealed DEV12 raw native inputs only' if not a.preflight else None;torch.set_num_threads(4)
    plan=json.loads((ROOT/'manifests/omega_corr_native_context_parity_v1.json').read_text())
    cohort=json.loads((ROOT/'manifests/omega_corr_cohort_v1.json').read_text())
    selected=[p for p in cohort['pairs'] if not a.preflight or p['pair_id'] in plan['preflight_pairs']]
    groups_selected={(p['scene'],p['group']):p['pair_id'] for p in selected}
    out=RUNTIME/('native_context_preflight_v1' if a.preflight else 'native_context_full_v1');out.mkdir(parents=True,exist_ok=False)
    inherited={}
    for row in lines(ROOT/'results/omega512_solver_numeric_v1.jsonl.gz'):
        if row['arm']=='G0_native' and row['pair_id'] in plan['preflight_pairs']:inherited[row['pair_id']]=row['matrices']
    index=json.loads((ROOT/'results/omega_corr_candidate_index_v1.json').read_text());assert sha256(index['input']['path'])==index['input']['sha256']
    native=json.loads(open(index['input']['path']).read());rank_by_query={tuple(q['query']):q['original_native_ranked_tracks'] for q in native['queries']}
    data=train_data();metadata={r['id']:r for r in data['images']};cfg,model=load_gmt(seed=20261014);assert not model.geo_runtime.active and model.test_len==40
    rows=[];image_calls=0;reused_images=0;sealed_images=0;raw_checks=0;max_raw_error=0.;max_score_error=0.;images_total=0;cache_portable={}
    if not a.preflight:
        parent=json.loads((RUNTIME/'native_context_preflight_v1/manifest.json').read_text());assert parent['status']=='MATCHED_NATIVE_CONTEXT_PARITY_PASS' and parent['max_raw_abs_error']==0
        assert parent['source']['checkpoint_sha256']==source['checkpoint_sha256'] and parent['source']['config_sha256']==source['config_sha256']
        assert sha256(parent['proposal_cache']['path'])==parent['proposal_cache']['sha256']
        cache_portable=torch.load(parent['proposal_cache']['path'],map_location='cpu',weights_only=True)
    output=out/'scores.jsonl.gz'
    with gzip.open(output,'xt') as stream:
        for scene in sorted({p['scene'] for p in selected}):
            directory=NATIVE/f'scene{scene}_full_v1';receipt=next(r for r in cohort['native_source_receipts'] if r['path']==str(directory/'manifest.json'))
            for fn in ['detector_provenance.jsonl.gz','actual_decisions.jsonl.gz']:assert sha256(directory/fn)==receipt['artifacts'][fn]
            records=video_records(data,scene);by_frame=collections.defaultdict(list)
            for r in records:by_frame[r['frame_id']].append(r)
            frames=sorted(by_frame);frame_index={f:i for i,f in enumerate(frames)}
            detectors=collections.defaultdict(list)
            for r in lines(directory/'detector_provenance.jsonl.gz'):detectors[r['source'][0]].append(r['detection'])
            calls=collections.defaultdict(list);pending=[]
            for d in lines(directory/'actual_decisions.jsonl.gz'):
                if d['stage']=='run_memory_tracker':pending.append(d);continue
                i=0 if d['stage']=='run_first_tracker_plus' else d['method_call_index']//2+1
                calls[i].append((d,pending));pending=[]
            assert not pending
            def context_ids(i,d):
                current=by_frame[frames[i]]
                if i==0:
                    counts=[len(detectors[r['image_id']]) for r in current]
                    seed=np.argsort(counts).tolist()[::-1][0]
                    return [current[seed]['image_id'],current[1-seed]['image_id']]
                view=d['method_call_index']%2
                return [r['image_id'] for f in frames[max(0,i+1-model.test_len):i] for r in by_frame[f]]+[r['image_id'] for r in current[:view+1]]
            needed=set()
            for i,entries in calls.items():
                if (scene,frames[i]) in groups_selected:
                    for d,_ in entries:needed.update(context_ids(i,d))
            images_total+=len(needed);assert images_total<=9250
            if scene==12 and not a.preflight:
                from read_dev12_native_release import NativeRelease
                release=NativeRelease('/data1/liuyeqiang/WWW_trackvggt_runtime/cv_pil_dev12_native_release_v1','ef7f65d0bccb1e665f7125391f4a3c64a1449b1fe4d204c735371522386421d4')
                assert release.labels is None
                by_primary={c['primary_index']:c for c in release.calls.values() if c['stage']!='run_memory_tracker'}
                for i,entries in sorted(calls.items()):
                    frame=frames[i];pair_id=groups_selected.get((scene,frame))
                    if pair_id is None:continue
                    for d,_ in entries:
                        ordinal=0 if d['stage']=='run_first_tracker_plus' else d['method_call_index']+1
                        c=by_primary[ordinal];assert c['queries']==d['queries'] and c['references']==d['references']
                        raw=c['raw_logits'].cuda();unique=torch.tensor(c['track_ids'],device='cuda',dtype=torch.long)
                        track_ids=torch.tensor([r['track'] for r in c['references']],device='cuda',dtype=torch.long)
                        membership=(track_ids[:,None]==unique[None,:]).float();scores=c['track_scores'].cuda()
                        rank=unique[torch.argsort(scores,dim=1,descending=True,stable=True)].cpu().tolist()
                        for q,r in zip(d['queries'],rank):assert r==rank_by_query[tuple(q)]
                        for p in d['hungarian_pairs']:assert float(scores[p['row'],list(c['track_ids']).index(p['track'])])==p['score']
                        qfeatures=torch.stack([release.vector(k) for k in c['query_feature_keys']]).cuda() if c['query_feature_keys'] else raw.new_empty((0,1152))
                        rfeatures=torch.stack([release.vector(k) for k in c['reference_feature_keys']]).cuda() if c['reference_feature_keys'] else raw.new_empty((0,1152))
                        valid=(torch.isfinite(qfeatures).all(1)&(qfeatures.norm(dim=1)>0))[:,None]&(torch.isfinite(rfeatures).all(1)&(rfeatures.norm(dim=1)>0))[None,:]
                        mask=torch.zeros_like(raw,dtype=torch.bool)
                        for qi,q in enumerate(d['queries']):
                            for rj,r in enumerate(d['references']):
                                qm,rm=metadata[q[0]],metadata[r['source'][0]]
                                mask[qi,rj]=qm['view_id']!=rm['view_id'] and abs(frame_index[qm['frame_id']]-frame_index[rm['frame_id']])<=20
                        mask&=valid;prior={}
                        for weight in [.1,.3]:
                            changed=PairScoreAdapter(enabled=True,feature_weight=weight,max_residual=1.).apply(raw,mask.float(),None,mask)
                            sc=torch.cat(model.roi_heads._activate_asso(changed.split(c['activation_block_sizes'],dim=1)),dim=1)@membership
                            prior[str(weight)]=dict(track_scores=sc.cpu().tolist(),ranked_tracks=unique[torch.argsort(sc,dim=1,descending=True,stable=True)].cpu().tolist())
                        row=dict(pair_id=pair_id,scene=scene,group=frame,queries=d['queries'],references=d['references'],raw_logits=raw.cpu().tolist(),unique_tracks=c['track_ids'],native_scores=scores.cpu().tolist(),native_ranked_tracks=rank,NativePrior=prior,original_actual_decision=d,from_sealed_DEV12=True)
                        stream.write(json.dumps(row,allow_nan=False,separators=(',',':'))+'\n');rows.append(dict(pair_id=pair_id,queries=len(d['queries']),references=len(d['references']),from_sealed=True))
                    print('pair sealed PASS',pair_id,flush=True)
                sealed_images+=len(needed);del release;continue
            seed_all(20261014);image_rng={};primary_rng={}
            def advance_association(d):
                if d['references']:random.sample(range(400),len({r['track'] for r in d['references']}))
            for i,frame in enumerate(frames):
                for r in by_frame[frame]:
                    image=r['image_id'];n=len(detectors[image])
                    if image in needed:image_rng[image]=torch.cuda.get_rng_state().clone()
                    dummy=torch.empty(n,device='cuda',dtype=torch.float32)
                    torch.rand_like(dummy);torch.randint(0,2,(n,),device='cuda');torch.randint(0,2,(n,),device='cuda')
                for d,memories in calls.get(i,[]):
                    if (scene,frame) in groups_selected:primary_rng[d['stage'],d['method_call_index']]=random.getstate()
                    advance_association(d)
                    for m in memories:advance_association(m)
            cache={}
            # No original tracking methods, no new state/history, no GT are used.
            for frame,group in sorted(by_frame.items()):
                needed_group=[r for r in group if r['image_id'] in needed]
                if not needed_group:continue
                mapped=map_records(cfg,group);mapped_by_image={r['image_id']:r for r in mapped}
                for view,r in enumerate(group):
                    image=r['image_id']
                    if image not in needed:continue
                    if image in cache_portable:
                        packed=cache_portable[image];inst=Instances(tuple(packed['image_size']))
                        for name,value in packed['fields'].items():inst.set(name,Boxes(value['box_tensor']) if isinstance(value,dict) else value)
                        assert len(inst)==len(detectors[image]);cache[image]=inst;reused_images+=1;continue
                    torch.cuda.set_rng_state(image_rng[image]);record=mapped_by_image[image]
                    with torch.inference_mode():inst=model.inference([record],2,[frames[0],frame,frame],view,do_postprocess=False)[0]
                    assert len(inst)==len(detectors[image]),('native detector count',image)
                    inst.decision_trace_source=torch.stack([torch.full((len(inst),),image,device='cuda',dtype=torch.long),torch.arange(len(inst),device='cuda')],dim=1)
                    public=CustomRCNN._postprocess([clone_instances(inst)],[record],[(0,0)],not_clamp_box=model.not_clamp_box)[0]['instances'].to('cpu')
                    actual=custom_instances_to_coco_json(public,image);assert actual==detectors[image],('native detector parity',image)
                    cache[image]=inst.to('cpu');image_calls+=1
                    packed={}
                    for k,v in cache[image].get_fields().items():
                        if isinstance(v,Boxes):packed[k]=dict(box_tensor=v.tensor)
                        elif isinstance(v,torch.Tensor):packed[k]=v
                        else:raise ValueError(('unexpected native field',k,type(v)))
                    cache_portable[image]=dict(image_size=cache[image].image_size,fields=packed)
                    if image_calls%100==0:print('images',image_calls,flush=True)
            for i,entries in sorted(calls.items()):
                frame=frames[i];pair_id=groups_selected.get((scene,frame))
                if pair_id is None:continue
                primary_counter=0
                for d,_ in entries:
                    ids=context_ids(i,d);refs_by_image=collections.defaultdict(list)
                    for r in d['references']:refs_by_image[r['source'][0]].append(r)
                    instances=[]
                    for image in ids[:-1]:
                        refs=refs_by_image[image];indices=[r['source'][1] for r in refs]
                        assert indices==list(range(len(cache[image])))
                        inst=cache[image][indices].to('cuda');inst.track_ids=torch.tensor([r['track'] for r in refs],device='cuda',dtype=torch.long);instances.append(inst)
                    qindices=[q[1] for q in d['queries']];assert all(q[0]==ids[-1] for q in d['queries'])
                    assert qindices==list(range(len(cache[ids[-1]])))
                    instances.append(cache[ids[-1]][qindices].to('cuda'));k=len(instances)-1
                    random.setstate(primary_rng[d['stage'],d['method_call_index']])
                    with torch.inference_mode():result=model.get_asso(instances,k)
                    raw=result[0][0];assert tuple(raw.shape)==(len(d['queries']),len(d['references']))
                    if pair_id in inherited:
                        old=inherited[pair_id][primary_counter];assert old['queries']==d['queries'] and old['references']==d['references']
                        expected=np.array(old['raw_logits'],dtype=np.float32).reshape(raw.shape);current=raw.cpu().numpy()
                        error=float(np.max(np.abs(current-expected))) if current.size else 0.;max_raw_error=max(max_raw_error,error)
                        np.testing.assert_allclose(current,expected,rtol=2e-5,atol=2e-5);raw_checks+=1
                    refs=[inst for j,inst in enumerate(instances) if j!=k]
                    track_ids=torch.cat([r.track_ids for r in refs]);unique=track_ids.unique(sorted=True);membership=(track_ids[:,None]==unique[None,:]).float()
                    def activate(x):
                        blocks=[x] if i==0 else x.split([len(z) for z in refs],dim=1)
                        return torch.cat(model.roi_heads._activate_asso(blocks),dim=1)@membership
                    scores=activate(raw);rank=unique[torch.argsort(scores,dim=1,descending=True,stable=True)].cpu().tolist()
                    for q,r in zip(d['queries'],rank):assert r==rank_by_query[tuple(q)],('original full GTA rank mismatch',scene,frame,q)
                    ri,ci=linear_sum_assignment(-scores.cpu().numpy());assert [(int(x),int(unique[y])) for x,y in zip(ri,ci)]==[(p['row'],p['track']) for p in d['hungarian_pairs']]
                    for x,y,p in zip(ri,ci,d['hungarian_pairs']):
                        error=abs(float(scores[x,y])-p['score']);max_score_error=max(max_score_error,error);assert error<=2e-6,('native pair score parity',scene,frame,error)
                    qfeatures=instances[k].reid_features;ref_features=torch.cat([r.reid_features for r in refs])
                    qvalid=torch.isfinite(qfeatures).all(1)&(qfeatures.norm(dim=1)>0);rvalid=torch.isfinite(ref_features).all(1)&(ref_features.norm(dim=1)>0)
                    mask=torch.zeros_like(raw,dtype=torch.bool)
                    for qi,q in enumerate(d['queries']):
                        qm=metadata[q[0]]
                        for rj,r in enumerate(d['references']):
                            rm=metadata[r['source'][0]]
                            mask[qi,rj]=qm['view_id']!=rm['view_id'] and abs(frame_index[qm['frame_id']]-frame_index[rm['frame_id']])<=20
                    mask&=qvalid[:,None]&rvalid[None,:]
                    prior={}
                    for weight in [.1,.3]:
                        changed=PairScoreAdapter(enabled=True,feature_weight=weight,max_residual=1.).apply(raw,mask.float(),None,mask)
                        sc=activate(changed);order=unique[torch.argsort(sc,dim=1,descending=True,stable=True)].cpu().tolist()
                        prior[str(weight)]=dict(track_scores=sc.cpu().tolist(),ranked_tracks=order)
                    row=dict(pair_id=pair_id,scene=scene,group=frame,queries=d['queries'],references=d['references'],raw_logits=raw.cpu().tolist(),
                        unique_tracks=unique.cpu().tolist(),native_scores=scores.cpu().tolist(),native_ranked_tracks=rank,NativePrior=prior,original_actual_decision=d)
                    stream.write(json.dumps(row,allow_nan=False,separators=(',',':'))+'\n');rows.append(dict(pair_id=pair_id,queries=len(d['queries']),references=len(d['references']),from_sealed=False));primary_counter+=1
                print('pair PASS',pair_id,flush=True)
    torch.save(cache_portable,out/'native_proposals_lossless.pt')
    save_new(out/'manifest.json',dict(status='MATCHED_NATIVE_CONTEXT_PARITY_PASS',preflight=a.preflight,source=source,pairs=len(selected),
        actual_image_forward_calls=image_calls,reused_preflight_images=reused_images,sealed_DEV12_context_images=sealed_images,unique_images_needed=images_total,primary_calls=len(rows),association_get_asso_invocations=sum(not r['from_sealed'] for r in rows),actual_association_transformer_forwards=sum(r['queries']>0 and r['references']>0 and not r['from_sealed'] for r in rows),queries=sum(r['queries'] for r in rows),
        inherited_raw_matrix_checks=raw_checks,max_raw_abs_error=max_raw_error,max_native_pair_score_error=max_score_error,
        output=dict(path=str(output),sha256=sha256(output)),proposal_cache=dict(path=str(out/'native_proposals_lossless.pt'),sha256=sha256(out/'native_proposals_lossless.pt')),
        no_tracking_replay=True,no_new_GT_matching=True,no_GT_input=True,no_new_ID_management=True,
        precision='Original GMT FP32 torch2.0 original eval jitter/RPCE restored; not Omega FP16 backend',
        CUDA_allocator_peak_bytes=torch.cuda.max_memory_allocated(),CUDA_reserved_peak_bytes=torch.cuda.max_memory_reserved()))

if __name__=='__main__':main()
