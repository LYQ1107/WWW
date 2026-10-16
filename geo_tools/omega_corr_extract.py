"""Streaming frozen local correspondence over every actual synchronous native candidate.

Dense image features are transient. All actual local matrices and coordinates
are retained in compressed scalar evidence, never a whole-video dense cache.
"""
import argparse
import collections
import gzip
import json
import time
import numpy as np
import torch
from omega_corr_common import ROOT,RUNTIME,OMEGA_RUNTIME,sha256,save_new,provenance
from omega_modules import geometry_module


def scalar(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,dict):return {k:scalar(x) for k,x in v.items()}
    return v


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--backend',choices=['omega','old_vggt','dino'],required=True);a=parser.parse_args()
    torch.set_num_threads(4);torch.manual_seed(20261016);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
    source=provenance();assert not source['source_status'].strip()
    index=json.loads((ROOT/'results/omega_corr_candidate_index_v1.json').read_text());assert sha256(index['input']['path'])==index['input']['sha256']
    native=json.loads(open(index['input']['path']).read());cohort=json.loads((ROOT/'manifests/omega_corr_cohort_v1.json').read_text())
    out=RUNTIME/'extraction'/a.backend;out.mkdir(parents=True,exist_ok=False)
    objects=geometry_module('object_patch_regions');matcher=geometry_module('crossview_point_matcher')
    grouped=collections.defaultdict(list)
    for q in native['queries']:grouped[q['pair_id']].append(q)
    proposals=collections.defaultdict(list)
    for key,p in native['proposals'].items():proposals[p['image_id']].append((key,p))
    if a.backend=='omega':
        module=geometry_module('omega_correspondence_encoder')
        encoder=module.OmegaCorrespondenceEncoder(OMEGA_RUNTIME/'official_vggt_omega',OMEGA_RUNTIME/'weights/vggt_omega_1b_512.pt',512,layers=(4,11,17,23))
    else:
        from omega_corr_controls import FrozenControl
        encoder=FrozenControl(a.backend)
    records=0;pair_receipts=[];inference_times=[]
    score_path=out/'candidate_scores.jsonl.gz'
    with gzip.open(score_path,'xt',compresslevel=5) as stream:
        for pair_index,pair in enumerate(cohort['pairs']):
            started=time.perf_counter();paths=[i['path'] for i in pair['images']];sizes=[(i['original_width'],i['original_height']) for i in pair['images']]
            assert all(sha256(i['path'])==i['sha256'] for i in pair['images'])
            layers=(4,11,17,23) if pair_index<16 else (11,17,23)
            features={};forward_times=[]
            if a.backend=='omega':
                images,maps=encoder.prepare(paths)
                joint=encoder.encode(images,[i['camera_id'] for i in pair['images']],pair['group'])
                for layer in layers:
                    v=joint['tokens'][layer]
                    features['F0',layer]=v[:,:,:,:1024];features['F1',layer]=v[:,:,:,1024:];features['F2',layer]=v
                forward_times.append(joint['CUDA_forward_and_hook_seconds'])
                independent=[];wrong=[]
                for view in range(2):
                    one=encoder.encode(images[view:view+1],[pair['images'][view]['camera_id']],pair['group'],'SINGLE_VIEW_INDEPENDENT')
                    independent.append(one['tokens']);forward_times.append(one['CUDA_forward_and_hook_seconds'])
                    two=encoder.encode(images[view:view+1].repeat(2,1,1,1),[pair['images'][view]['camera_id']]*2,pair['group'],'ARTIFICIAL_SAME_CAMERA_WRONG_PARTNER')
                    wrong.append({layer:t[:,view:view+1] for layer,t in two['tokens'].items()});forward_times.append(two['CUDA_forward_and_hook_seconds'])
                for layer in layers:
                    features['F3',layer]=torch.cat([v[layer] for v in independent],dim=1)
                    features['N1',layer]=torch.cat([v[layer] for v in wrong],dim=1)
                patch=16
            else:
                images,maps=encoder.prepare(paths,sizes);tokens,elapsed=encoder.encode(images,layers)
                features={(a.backend,layer):v for layer,v in tokens.items()};forward_times=[elapsed];patch=14
            all_regions={};views={}
            for view,(image,mapping) in enumerate(zip(pair['images'],maps)):
                for key,p in proposals[image['image_id']]:all_regions[key]=objects.object_regions(mapping,p['bbox_xyxy']);views[key]=view
            # Bounded local feature cache for only this physical pair.
            patch_cache={};roi_cache={}
            from torchvision.ops import roi_align
            for (variant,layer),full in features.items():
                for key,region in all_regions.items():
                    view=views[key];tensor=full[0,view]
                    for mode,info in region['regions'].items():
                        ids=torch.tensor(info['patch_ids'],dtype=torch.long,device=tensor.device)
                        patch_cache[variant,layer,key,mode]=tensor.index_select(0,ids).float().cpu().numpy()
                    h,w=images.shape[-2]//patch,images.shape[-1]//patch
                    grid=tensor.T.reshape(1,tensor.shape[-1],h,w).float()
                    valid_grid=objects.patch_lattice(maps[view],patch)['valid'].reshape(h,w)
                    grid=grid*grid.new_tensor(valid_grid)[None,None]
                    bbox=torch.tensor(region['network_bbox'],device=tensor.device,dtype=torch.float32)[None]
                    bbox[:,0::2].clamp_(maps[view].pad_left,maps[view].pad_left+maps[view].resize_width)
                    bbox[:,1::2].clamp_(maps[view].pad_top,maps[view].pad_top+maps[view].resize_height)
                    if bbox[0,2]<=bbox[0,0] or bbox[0,3]<=bbox[0,1]:pooled=np.zeros(tensor.shape[-1],dtype=np.float32)
                    else:pooled=roi_align(grid,[bbox],output_size=(3,3),spatial_scale=1/patch,sampling_ratio=2,aligned=True).mean((-1,-2))[0].cpu().numpy()
                    roi_cache[variant,layer,key]=pooled
            pair_records=0
            for query in grouped[pair['pair_id']]:
                qkey=':'.join(map(str,query['query']));refs=sorted(query['legal_synchronous_references'],key=lambda r:r['source'])
                for ref_index,ref in enumerate(refs):
                    rkey=':'.join(map(str,ref['source']))
                    for variant,layer in features:
                        for mode in ['M0','M1','M2']:
                            x=patch_cache[variant,layer,qkey,mode];y=patch_cache[variant,layer,rkey,mode]
                            metrics,raw=matcher.match_tokens(x,y)
                            qinfo=all_regions[qkey]['regions'][mode];rinfo=all_regions[rkey]['regions'][mode]
                            if mode=='M0':
                                u=roi_cache[variant,layer,qkey];v=roi_cache[variant,layer,rkey];den=np.linalg.norm(u)*np.linalg.norm(v)
                                metrics['roi_align_cosine']=float(u@v/den) if den>1e-12 else None
                            if metrics['status']=='LOCAL_VISUAL_MATCH_MEASURED':
                                qa=qinfo['is_object_token'][raw['mutual_indices_A']];ra=rinfo['is_object_token'][raw['mutual_indices_B']]
                                metrics['mutual_object_to_object_count']=int(np.sum(qa&ra));metrics['mutual_involving_background_count']=int(np.sum(~(qa&ra)))
                            item=dict(pair_id=pair['pair_id'],scene=pair['scene'],role=pair['role'],query=query['query'],reference=ref['source'],
                                reference_track=ref['track'],backend=a.backend,variant=variant,layer=layer,mode=mode,
                                metric=metrics,raw=raw,patch_ids_A=qinfo['patch_ids'],patch_ids_B=rinfo['patch_ids'],
                                original_centers_A=qinfo['original_centers'],original_centers_B=rinfo['original_centers'],
                                object_fraction_A=qinfo['object_token_fraction'],object_fraction_B=rinfo['object_token_fraction'])
                            stream.write(json.dumps(scalar(item),allow_nan=False,separators=(',',':'))+'\n');records+=1;pair_records+=1
                            if variant=='F2':
                                mapped=refs[(ref_index+1)%len(refs)];mapped_key=':'.join(map(str,mapped['source']))
                                mm,rr=matcher.match_tokens(x,patch_cache[variant,layer,mapped_key,mode]);mi=all_regions[mapped_key]['regions'][mode]
                                control=dict(item,variant='N2',mapped_region_source=mapped['source'],metric=mm,raw=rr,
                                    patch_ids_B=mi['patch_ids'],original_centers_B=mi['original_centers'],object_fraction_B=mi['object_token_fraction'],singleton_unchanged=len(refs)==1)
                                stream.write(json.dumps(scalar(control),allow_nan=False,separators=(',',':'))+'\n');records+=1;pair_records+=1
            pair_receipts.append(dict(pair_id=pair['pair_id'],actual_forward_calls=len(forward_times),forward_seconds=forward_times,
                scalar_records=pair_records,pair_wall_seconds=time.perf_counter()-started,mappings=[m.record() for m in maps],
                input_shape=list(images.shape),layers=list(layers),no_legal_sync_query=sum(not bool(q['legal_synchronous_references']) for q in grouped[pair['pair_id']])))
            inference_times.extend(forward_times)
            print(json.dumps(pair_receipts[-1]),flush=True)
            del features,patch_cache,roi_cache
            if a.backend=='omega':del joint,independent,wrong,one,two
            else:del tokens
            assert sum(p.stat().st_size for p in RUNTIME.rglob('*') if p.is_file() and 'deps' not in p.parts and 'sources' not in p.parts)<2147483648,'bounded research cache cap reached'
    save_new(out/'manifest.json',dict(status='FROZEN_LOCAL_CORRESPONDENCE_EXTRACTED',backend=a.backend,pairs=len(pair_receipts),
        records=records,actual_forward_calls_attempted=encoder.calls_attempted,actual_forward_calls_completed=encoder.calls_completed,
        source=source,pair_receipts=pair_receipts,precision='FP16 nativeV100 AMP; similarity FP32 CPU',
        output=dict(path=str(score_path),sha256=sha256(score_path)),
        CUDA_allocator_peak_bytes=torch.cuda.max_memory_allocated(),CUDA_reserved_peak_bytes=torch.cuda.max_memory_reserved(),
        forward_seconds_P50=float(np.quantile(inference_times,.5)),forward_seconds_P95=float(np.quantile(inference_times,.95)),
        no_GT_labels_input=True,no_new_F_or_pose_scan=True,no_training=True,no_dense_video_feature_cache=True,
        torch=torch.__version__,numpy=np.__version__,state_keys=encoder.state_keys,all_model_parameters_frozen=all(not p.requires_grad for p in encoder.model.parameters()),
        GPU=torch.cuda.get_device_name(),capability=list(torch.cuda.get_device_capability()),
        control_source_commit=getattr(encoder,'source_commit','48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6'),weight_sha256=getattr(encoder,'weight_sha256','c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934')))

if __name__=='__main__':main()
