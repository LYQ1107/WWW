"""Registered32-pair LOCAL_CROP_VISUAL scale control, independently encoded proposals."""
import argparse
import collections
import gzip
import json
import math
import time
import numpy as np
from PIL import Image
import torch
from torchvision.transforms.functional import pil_to_tensor
from torchvision.ops import roi_align
from omega_corr_common import ROOT,RUNTIME,OMEGA_RUNTIME,save_new,sha256,provenance
from omega_modules import geometry_module
from omega_corr_extract import scalar


def crop_tensor(image,box,side):
    x1,y1,x2,y2=box;cx=(x1+x2)/2;cy=(y1+y2)/2
    left=max(0,math.floor(cx-.75*(x2-x1)));top=max(0,math.floor(cy-.75*(y2-y1)))
    right=min(image.width,math.ceil(cx+.75*(x2-x1)));bottom=min(image.height,math.ceil(cy+.75*(y2-y1)))
    if right<=left or bottom<=top:return None,None
    crop=image.crop((left,top,right,bottom));cw,ch=crop.size;scale=side/max(cw,ch)
    rw=max(1,round(cw*scale));rh=max(1,round(ch*scale));pl=(side-rw)//2;pt=(side-rh)//2
    resized=crop.resize((rw,rh),Image.Resampling.BICUBIC);canvas=Image.new('RGB',(side,side),(255,255,255));canvas.paste(resized,(pl,pt))
    P=geometry_module('omega_adapter').PixelMapping
    mapping=P(image.width,image.height,left,top,cw,ch,rw,rh,pl,pt,side,side)
    return pil_to_tensor(canvas).float()/255.,mapping


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--backend',choices=['omega','dino'],required=True);a=parser.parse_args()
    source=provenance();assert not source['source_status'].strip()
    torch.set_num_threads(4);torch.manual_seed(20261016);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
    coverage=json.loads((ROOT/'results/omega_corr_coverage_audit_v1.json').read_text());assert coverage['required_local_crop_experiment']
    cohort=json.loads((ROOT/'manifests/omega_corr_cohort_v1.json').read_text());pairmap={p['pair_id']:p for p in cohort['pairs']}
    selected=cohort['local_crop_pair_ids'];assert len(selected)==32
    index=json.loads((ROOT/'results/omega_corr_candidate_index_v1.json').read_text());assert sha256(index['input']['path'])==index['input']['sha256']
    native=json.loads(open(index['input']['path']).read());by_image=collections.defaultdict(list);queries=collections.defaultdict(list)
    for key,p in native['proposals'].items():by_image[p['image_id']].append((key,p))
    for q in native['queries']:queries[q['pair_id']].append(q)
    if a.backend=='omega':
        encoder=geometry_module('omega_correspondence_encoder').OmegaCorrespondenceEncoder(OMEGA_RUNTIME/'official_vggt_omega',OMEGA_RUNTIME/'weights/vggt_omega_1b_512.pt',512,layers=(11,17,23))
        patch=16
    else:
        from omega_corr_controls import FrozenControl
        encoder=FrozenControl('dino');patch=14
    regions=geometry_module('object_patch_regions');matcher=geometry_module('crossview_point_matcher')
    out=RUNTIME/'local_crops'/a.backend;out.mkdir(parents=True,exist_ok=False);path=out/'candidate_scores.jsonl.gz'
    pair_receipts=[];records=0;times=[]
    with gzip.open(path,'xt',compresslevel=5) as stream:
        for pair_id in selected:
            pair=pairmap[pair_id];originals={}
            for i in pair['images']:
                assert sha256(i['path'])==i['sha256']
                with Image.open(i['path']) as image:originals[i['image_id']]=image.convert('RGB')
            for side in [224,336]:
                started=time.perf_counter();cache={};rois={};infos={};maps={};calls_before=encoder.calls_completed
                for i in pair['images']:
                    for key,p in by_image[i['image_id']]:
                        tensor,mapping=crop_tensor(originals[i['image_id']],p['bbox_xyxy'],side)
                        if tensor is None:continue
                        if a.backend=='omega':
                            extracted=encoder.encode(tensor[None],[i['camera_id']],pair['group'],'LOCAL_CROP_VISUAL')
                            features={layer:t[0,0] for layer,t in extracted['tokens'].items()};elapsed=extracted['CUDA_forward_and_hook_seconds']
                        else:
                            torch.cuda.synchronize();t=time.perf_counter();encoder.calls_attempted+=1
                            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
                                prediction=encoder.model.forward_features((tensor[None].cuda()-encoder.mean)/encoder.std)
                            v=prediction['x_norm_patchtokens'];assert tuple(v.shape)==(1,(side//14)**2,1024) and torch.isfinite(v).all()
                            features={23:v[0]};torch.cuda.synchronize();elapsed=time.perf_counter()-t;encoder.calls_completed+=1
                        times.append(elapsed);r=regions.object_regions(mapping,p['bbox_xyxy'],patch_size=patch);infos[key]=r;maps[key]=mapping.record()
                        valid=regions.patch_lattice(mapping,patch)['valid'].reshape(side//patch,side//patch)
                        for layer,tokens in features.items():
                            for mode,info in r['regions'].items():
                                indices=torch.tensor(info['patch_ids'],device=tokens.device,dtype=torch.long)
                                cache[layer,key,mode]=tokens.index_select(0,indices).float().cpu().numpy()
                            grid=tokens.T.reshape(1,tokens.shape[-1],side//patch,side//patch).float()*tokens.new_tensor(valid)[None,None]
                            bbox=grid.new_tensor(r['network_bbox'])[None]
                            bbox[:,0::2].clamp_(mapping.pad_left,mapping.pad_left+mapping.resize_width);bbox[:,1::2].clamp_(mapping.pad_top,mapping.pad_top+mapping.resize_height)
                            rois[layer,key]=roi_align(grid,[bbox],output_size=(3,3),spatial_scale=1/patch,sampling_ratio=2,aligned=True).mean((-1,-2))[0].cpu().numpy()
                layers=[11,17,23] if a.backend=='omega' else [23];count=0
                for q in queries[pair_id]:
                    qkey=':'.join(map(str,q['query']))
                    for ref in q['legal_synchronous_references']:
                        rkey=':'.join(map(str,ref['source']))
                        for layer in layers:
                            for mode in ['M0','M1','M2']:
                                empty=dict(patch_ids=np.empty(0,dtype=int),available_count=0,original_centers=np.empty((0,2)),is_object_token=np.empty(0,dtype=bool),object_token_fraction=None)
                                qi=infos[qkey]['regions'][mode] if qkey in infos else empty
                                ri=infos[rkey]['regions'][mode] if rkey in infos else empty
                                dim=2048 if a.backend=='omega' else 1024
                                x=cache.get((layer,qkey,mode),np.empty((0,dim),np.float32));y=cache.get((layer,rkey,mode),np.empty((0,dim),np.float32))
                                m,raw=matcher.match_tokens(x,y)
                                if qkey not in infos or rkey not in infos:m['crop_failure']='NO_NONEMPTY_CLIPPED_CROP'
                                if mode=='M0' and (layer,qkey) in rois and (layer,rkey) in rois:
                                    u,v=rois[layer,qkey],rois[layer,rkey];d=np.linalg.norm(u)*np.linalg.norm(v);m['roi_align_cosine']=float(u@v/d) if d>1e-12 else None
                                if m['status']=='LOCAL_VISUAL_MATCH_MEASURED':
                                    object_pair=qi['is_object_token'][raw['mutual_indices_A']]&ri['is_object_token'][raw['mutual_indices_B']]
                                    m['mutual_object_to_object_count']=int(object_pair.sum());m['mutual_involving_background_count']=int((~object_pair).sum())
                                item=dict(pair_id=pair_id,scene=pair['scene'],role=pair['role'],query=q['query'],reference=ref['source'],reference_track=ref['track'],
                                    route='LOCAL_CROP_VISUAL',crop_side=side,backend=a.backend,variant='LOCAL_F3' if a.backend=='omega' else 'LOCAL_DINO',
                                    layer=layer,mode=mode,metric=m,raw=raw,patch_ids_A=qi['patch_ids'],patch_ids_B=ri['patch_ids'],
                                    available_token_count_A=qi['available_count'],available_token_count_B=ri['available_count'],
                                    observed_mutual_over_all_available_A=m['mutual_count']/qi['available_count'] if qi['available_count'] else None,
                                    observed_mutual_over_all_available_B=m['mutual_count']/ri['available_count'] if ri['available_count'] else None,
                                    original_centers_A=qi['original_centers'],original_centers_B=ri['original_centers'],object_fraction_A=qi['object_token_fraction'],object_fraction_B=ri['object_token_fraction'])
                                stream.write(json.dumps(scalar(item),allow_nan=False,separators=(',',':'))+'\n');records+=1;count+=1
                pair_receipts.append(dict(pair_id=pair_id,crop_side=side,encoded_physical_proposals=len(maps),actual_forward_calls=encoder.calls_completed-calls_before,
                    records=count,unavailable_proposals=[key for i in pair['images'] for key,p in by_image[i['image_id']] if key not in infos],pair_scale_wall_seconds=time.perf_counter()-started,mappings=maps))
                print(json.dumps({k:v for k,v in pair_receipts[-1].items() if k!='mappings'}),flush=True)
                del cache,features,rois
                assert sum(p.stat().st_size for p in RUNTIME.rglob('*') if p.is_file() and 'deps' not in p.parts and 'sources' not in p.parts)<2147483648
    save_new(out/'manifest.json',dict(status='REGISTERED_LOCAL_SCALE_CONTROLS_COMPLETE',route='LOCAL_CROP_VISUAL',backend=a.backend,pairs=32,sizes=[224,336],
        records=records,actual_forward_calls_attempted=encoder.calls_attempted,actual_forward_calls_completed=encoder.calls_completed,source=source,
        pair_receipts=pair_receipts,output=dict(path=str(path),sha256=sha256(path)),precision='FP16 model, FP32 normalized similarity',
        forward_seconds_P50=float(np.quantile(times,.5)),forward_seconds_P95=float(np.quantile(times,.95)),
        CUDA_allocator_peak_bytes=torch.cuda.max_memory_allocated(),CUDA_reserved_peak_bytes=torch.cuda.max_memory_reserved(),
        no_new_GMT_or_GT_matching=True,crops_not_physical_cameras=True,no_new_original_image_detail_claim=True,independent_proposal_inputs=True))

if __name__=='__main__':main()
