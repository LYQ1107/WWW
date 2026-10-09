"""Actual one-V100 image->detector->VFCE->native association live benchmark."""
import argparse,time
import numpy as np
import torch
from jev_phase13_learning import *
from jev_phase13_runtime import *
from evaluate_jev_stage2_online import load_policy

def frontend_profile(model,image):
    from cache_jev_phase13_stage1 import extract
    from detectron2.config import get_cfg
    from centernet.config import add_centernet_config
    from gtr.config import add_gtr_config
    cfg=get_cfg();add_centernet_config(cfg);add_gtr_config(cfg);cfg.merge_from_file(str(ROOT/'configs/VISION_stage1.yaml'));total=collections.Counter();handles=[]
    def hook(m,args,out):
        if not torch.is_tensor(out):return
        if isinstance(m,torch.nn.Linear):total['linear_MAC']+=int(out.numel()*m.in_features)
        elif getattr(m,'weight',None) is not None and m.weight.ndim==4:total['convolution_MAC']+=int(out.numel()*m.weight.shape[1]*m.weight.shape[2]*m.weight.shape[3])
    for m in model.modules():
        w=getattr(m,'weight',None)
        if isinstance(m,torch.nn.Linear) or torch.is_tensor(w) and w.ndim==4:handles.append(m.register_forward_hook(hook))
    with torch.no_grad():inst=extract(model,cfg,image)
    for h in handles:h.remove()
    return {'actual_detected_ROIs':len(inst),'image_path':str(image),'executed_MAC':sum(total.values()),'matmul_FLOPs':2*sum(total.values()),'breakdown':dict(total),'scope':'actual conv/deform-conv weighted sums and linear layers on one real payload; offset convolution counted, deformable sampling/normalization/pooling/softmax/indexing excluded'}

def main(variant,seed,frames=256):
    protect();source=binding();assert not source['dirty'];torch.set_num_threads(1);torch.manual_seed(20261009);policy,trained,temp=load_policy(variant,seed,'formal');model=build_tracker(17,policy,variant=variant,temperature=(temp,1.),react_learned=False,live=True);values,total,_=cache_inputs(17);assert model.jev_perception_cache_reader is None
    with torch.no_grad():run(model,values,total,stop=7)
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();begin=time.perf_counter()
    with torch.no_grad():live,_=run(model,values,total,stop=frames-1)
    torch.cuda.synchronize();elapsed=time.perf_counter()-begin;peak=torch.cuda.max_memory_allocated()/1048576;reserved=torch.cuda.max_memory_reserved()/1048576;latency=list(model.jev_stage2_executor.latency);liveids=[i.track_ids.cpu().tolist() for i in live];out=OUT/'live_efficiency_v1'/variant/f'seed{seed}';out.mkdir(parents=True,exist_ok=True)
    # Independent cached run verifies live/cached detection geometry and IDs.
    cached=build_tracker(17,policy,variant=variant,temperature=(temp,1.),react_learned=False);captured=[None]
    def observe(**d):
        if d['context']['frame']==64 and d['context']['view']==0 and d['task']==0:captured[0]={k:v.clone() for k,v in d['batch'].items()}
    cached.jev_stage2_executor.observer=observe
    with torch.no_grad():raw,_=run(cached,values,total,stop=frames-1)
    assert len(raw)==len(live)
    counts=[{'payload':i,'cached_count':len(a),'live_count':len(b)} for i,(a,b) in enumerate(zip(raw,live)) if len(a)!=len(b)]
    assert not counts,counts[:10]
    ids_equal=[i.track_ids.cpu().tolist() for i in raw]==liveids;boxerror=max((float((a.pred_boxes.tensor-b.pred_boxes.tensor).abs().max()) for a,b in zip(raw,live) if len(a)),default=0.);featureerror=max((float((a.reid_features-b.reid_features).abs().max()) for a,b in zip(raw,live) if len(a)),default=0.);assert boxerror<=1e-3 and featureerror<=1e-3
    samples=[r['total_stage2_ms'] for r in latency[10:]];quantiles=dict(zip(['p50','p95','max'],map(float,np.quantile(samples,[.5,.95,1]))));capacity=profile(policy,captured[0]) if policy is not None else {'registered_parameters':0,'executed_MAC':None,'matmul_FLOPs':None,'scope':'parameter-free cosine plus native Hungarian; not approximated as NN MAC'}
    capacity['frontend_live_profile']=frontend_profile(model,values[0]['phase13_image_path'])
    r={'status':'COMPLETE','binding':source,'variant':variant,'seed':seed,'trained':trained,'video':17,'first_scene_frames':frames,'camera_payloads':2*frames,'live_real_images':True,'perception_cache_hits':0,'GTA_RPCE_throw_mock':True,'warmup_frames':8,'actual_wall_seconds':elapsed,'full_scene_FPS':frames/elapsed,'full_camera_payload_FPS':2*frames/elapsed,'stage2_ms':quantiles,'Stage2_latency_scope':'real memory input construction, independent NN or cosine scores, private-terminal Hungarian, native bank/birth/Gallery/metadata commit; no journals/GT/evaluator during timed run','live_cached_same_ID_stream':ids_equal,'live_cached_max_box_error':boxerror,'live_cached_max_VFCE_error':featureerror,'peak_full_VRAM_allocated_MiB':peak,'peak_full_VRAM_reserved_MiB':reserved,'stage2_capacity':capacity,'loaded_native_frontend_parameters_including_dormant_GTA':sum(p.numel() for p in model.parameters()),'dormant_GTA_scope':'original module still loaded for strict Stage1 checkpoint compatibility; forbidden, never scored; dormant weights are included in full VRAM/loaded parameter count','hardware':torch.cuda.get_device_name(),'budget':{'stage2_p95_ms':10.,'full_two_camera_scene_FPS_min':25.},'budget_PASS':quantiles['p95']<=10 and frames/elapsed>=25.,'heldout':'SEALED','Full24':False,'official_TEST':False};save(out/'RESULT.json',r);save(out/'LATENCIES.json',latency);print('PHASE13_REAL_LIVE_EFFICIENCY',variant,r['full_scene_FPS'],quantiles,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--variant',required=True);p.add_argument('--seed',type=int,default=20261009);p.add_argument('--frames',type=int,default=256);a=p.parse_args();main(a.variant,a.seed,a.frames)
