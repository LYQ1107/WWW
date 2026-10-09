"""Isolated real-image FPS, exact dormant-weight parity and nested component timing."""
import collections
import gc
import time
import numpy as np
import torch
from jev_phase14_common import *
from jev_phase13_runtime import build_tracker,cache_inputs,run
from jev_phase14_online import load_policy
from gtr.modeling.jev_phase14.memory import CachedIdentityMemory
from gtr.modeling.jev_stage2.memory import IdentityMemory
from gtr.modeling.jev_native_state import fingerprint


def quantile(values):
    return dict(zip(['p50','p95','max'],map(float,np.quantile(values,[.5,.95,1])))) if values else None

def committed(raw,frames):return raw[:2*frames]

def ids(raw,frames):return [i.track_ids.cpu().tolist() if i.has('track_ids') else [] for i in committed(raw,frames)]

def prune(model):
    count={};torch.cuda.synchronize();before=torch.cuda.memory_allocated()
    for name in ['transformer','asso_predictor','s_t_head']:
        module=getattr(model.roi_heads,name);count[name]=sum(p.numel() for p in module.parameters());setattr(model.roi_heads,name,None)
    gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
    return dict(removed_parameters=count,removed_total=sum(count.values()),actual_allocated_bytes_released=before-torch.cuda.memory_allocated(),scope='after strict original checkpoint load; detector, VFCE pooler and VFCE head remain')

def mirror_extract(model,cfg,image,timed):
    from PIL import Image
    from detectron2.structures import Instances
    from gtr.data.transforms.custom_augmentation_impl import EfficientDetResizeCrop
    start=time.perf_counter();rgb=np.array(Image.open(image).convert('RGB'));rgb=EfficientDetResizeCrop(cfg.INPUT.TEST_SIZE,(1.,1.)).get_transform(rgb).apply_image(rgb)
    tensor=torch.as_tensor(np.ascontiguousarray(rgb.transpose(2,0,1)));timed['image_load_resize_CPU_ms'].append((time.perf_counter()-start)*1000)
    def duration(name,fn):
        torch.cuda.synchronize();start=time.perf_counter();out=fn();sync=time.perf_counter();torch.cuda.synchronize();end=time.perf_counter()
        timed[name].append((end-start)*1000);timed['CUDA_blocking_sync_subset_ms'].append((end-sync)*1000);return out
    images=duration('image_preprocess_transfer_ms',lambda:model.preprocess_image([{'image':tensor}]))
    features=duration('backbone_ms',lambda:model.backbone(images.tensor))
    proposals,_=duration('detector_ms',lambda:model.proposal_generator(images,features,None))
    proposals=[p[p.objectness_logits>model.roi_heads.asso_thresh_test] for p in proposals]
    pooled=duration('VFCE_ROI_pooling_ms',lambda:model.roi_heads.asso_pooler([features[f] for f in model.roi_heads.asso_in_features],[p.proposal_boxes for p in proposals]))
    vfce=duration('VFCE_projection_ms',lambda:model.roi_heads.asso_head(pooled));p=proposals[0]
    inst=Instances(p.image_size);inst.pred_boxes=p.proposal_boxes;inst.scores=p.objectness_logits;inst.pred_classes=torch.zeros(len(p),dtype=torch.long,device=model.device);inst.reid_features=vfce;return inst

def install_attribution(model,policy,timed):
    from detectron2.config import get_cfg
    from centernet.config import add_centernet_config
    from gtr.config import add_gtr_config
    import gtr.modeling.jev_stage2.native as native
    cfg=get_cfg();add_centernet_config(cfg);add_gtr_config(cfg);cfg.merge_from_file(str(ROOT/'configs/VISION_stage1.yaml'));cfg.freeze()
    def wrapper(name,fn):
        def call(*a,**k):
            torch.cuda.synchronize();start=time.perf_counter();value=fn(*a,**k);sync=time.perf_counter();torch.cuda.synchronize();end=time.perf_counter()
            timed[name].append((end-start)*1000);timed['CUDA_blocking_sync_subset_ms'].append((end-sync)*1000);return value
        return call
    def inference(payloads,*a,**k):return [mirror_extract(model,cfg,payloads[0]['phase13_image_path'],timed)]
    model.inference=inference
    ex=model.jev_stage2_executor
    class TimedMemory(CachedIdentityMemory):
        def build(self,*a,**k):return wrapper('history_build_ms',super().build)(*a,**k)
    ex.memory_factory=TimedMemory;original_scores=ex.scores
    def score(batch,refs,context,task):return wrapper('policy_MATCH_ms' if task==0 else 'REACT_fallback_ms',original_scores)(batch,refs,context,task)
    ex.scores=score;ex.commit=wrapper('total_native_Stage2_ms',ex.commit);ex.promote_bank=wrapper('native_bank_ms',ex.promote_bank)
    previous=native.lawful_choice;native.lawful_choice=wrapper('global_assignment_ms',previous)
    if policy is not None:
        for name,method in [('StateEncoder_ms','encode_state'),('QuestionReader_ms','encode_questions'),('OptionReader_ms','score_questions')]:setattr(policy.core,method,wrapper(name,getattr(policy.core,method)))
    return previous,cfg

def main():
    protect();source=binding();protocol=read(REPORTS/'EFFICIENCY_PROTOCOL.json');assert os.environ.get('CUDA_VISIBLE_DEVICES')==str(protocol['GPU'])
    assert torch.cuda.device_count()==1;torch.set_num_threads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    out=OUT/'latency_v2';out.mkdir(parents=True,exist_ok=True);frames=protocol['scene_frames'];values,total,reader=cache_inputs(17)
    cases=[]
    for variant in protocol['primary_models']:
        case=out/variant;case.mkdir(parents=True,exist_ok=True)
        if (case/'RESULT.json').exists():cases.append(read(case/'RESULT.json'));continue
        torch.manual_seed(20261009);policy,trained,_=load_policy(variant,20261009,'formal');model=build_tracker(17,policy=policy,react_learned=False,live=True);model.jev_stage2_executor.memory_factory=CachedIdentityMemory
        with torch.no_grad():run(model,values,total,stop=7)
        torch.cuda.synchronize();start=time.perf_counter()
        with torch.no_grad():baseline,_=run(model,values,total,stop=frames-1)
        torch.cuda.synchronize();baseline_seconds=time.perf_counter()-start;baseline_ids=ids(baseline,frames);baseline_tensor_SHA=fingerprint(committed(baseline,frames))
        del baseline;released=prune(model);trials=[];pruned_sha=None
        for repeat in range(protocol['repeats']):
            with torch.no_grad():run(model,values,total,stop=7)
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
            with torch.no_grad():raw,_=run(model,values,total,stop=frames-1)
            torch.cuda.synchronize();seconds=time.perf_counter()-start
            assert ids(raw,frames)==baseline_ids,'dormant-weight removal changed native committed IDs'
            pruned_sha=fingerprint(committed(raw,frames));assert pruned_sha==baseline_tensor_SHA,'dormant-weight removal changed actual tensors'
            sample=model.jev_stage2_executor.latency[16:]
            trials.append(dict(repeat=repeat,seconds=seconds,full_sceneFPS=frames/seconds,camera_payloadFPS=2*frames/seconds,
                total_Stage2_ms=quantile([s['total_stage2_ms'] for s in sample]),history_build_ms=quantile([s['build_ms'] for s in sample]),
                peak_allocated_MiB=torch.cuda.max_memory_allocated()/1048576,peak_reserved_MiB=torch.cuda.max_memory_reserved()/1048576))
            del raw
            save(case/'PROGRESS.json',dict(status='LIVE_FPS_TRIAL_COMPLETE',repeat=repeat,values=trials[-1]));print('PHASE14_REAL_IMAGE_TRIAL',variant,repeat,trials[-1],flush=True)
        # Real varying native inputs for a separate policy-only timing sample.
        # This capture and microbenchmark never enters measured image FPS.
        captured=[];executor=model.jev_stage2_executor
        def capture(**d):
            c=d['context']
            if d['task']==0 and c['frame']>=8 and c['frame']%8==0:
                captured.append(dict(batch={k:v.clone() for k,v in d['batch'].items()},refs=list(d['refs']),context=dict(c)))
        model.jev_perception_cache_reader=reader;executor.observer=capture
        with torch.no_grad():cached_raw,_=run(model,values,total,stop=frames-1)
        cached_live_ids_equal=ids(cached_raw,frames)==baseline_ids;executor.observer=None;policy_times=[]
        for repeat in range(3):
            for item in captured:
                torch.cuda.synchronize();start=time.perf_counter()
                with torch.no_grad():executor.scores(item['batch'],item['refs'],item['context'],0)
                torch.cuda.synchronize();policy_times.append((time.perf_counter()-start)*1000)
        policy_only=quantile(policy_times[4:]);policy_input_shapes=[{k:list(t.shape) for k,t in item['batch'].items()} for item in captured]
        del cached_raw,captured;model.jev_perception_cache_reader=None
        # Attribution is deliberately separate; added synchronizations never enter
        # primary FPS or establish an optimization speed benefit.
        timed=collections.defaultdict(list);previous,cfg=install_attribution(model,policy,timed)
        from cache_jev_phase13_stage1 import extract
        with torch.no_grad():
            for payload in [values[0],values[total],values[16],values[total+16]]:
                original=extract(model,cfg,payload['phase13_image_path']);mirror=mirror_extract(model,cfg,payload['phase13_image_path'],timed)
                assert fingerprint(original)==fingerprint(mirror),'profiled preprocessing changed perception'
            timed.clear();observed,_=run(model,values,total,stop=63)
        import gtr.modeling.jev_stage2.native as native
        native.lawful_choice=previous;attribution={name:quantile(times[16:]) for name,times in timed.items()};save(case/'ATTRIBUTION_SAMPLES.json',dict(timed))
        r=dict(status='COMPLETE',binding=source,variant=variant,seed=20261009,trained=trained,hardware=torch.cuda.get_device_name(),source_GPU=9,
            real_image_trials=trials,unpruned_single_reference_seconds=baseline_seconds,
            policy_only_ms=policy_only,policy_only_scope='synchronized same-policy MATCH scoring on varying actual cached native prefixes every8frames,3repeats; no history build or solver; outside primary live FPS',
            policy_input_shapes=policy_input_shapes,cached_live_same_ID_stream=cached_live_ids_equal,
            dormant_pruning=released,exact_pruned_native_ID_and_all_Instance_field_SHA=True,original_native_tensor_SHA256=baseline_tensor_SHA,
            profile_mirror_bitwise_verified_four_images=True,instrumented_attribution=attribution,
            attribution_scope='separate synchronized wall-timing run; nested State/Question/Option and CUDA blocking are overlapping subsets, not additive; total policy includes typed aux heads; native residual includes ID/Gallery writes and Python overhead',
            full_FPS_scope='real image IO+detector+VFCE+native tracks;512 actual camera payloads;8warmup scene frames;no GT/journals/evaluator in timed path;3restarts',
            no_speed_claim_from_single_unpruned_reference=True,perception_cache_hits=0,
            gate=dict(Stage2_p95_ms_max=10.,full_sceneFPS_min=25.),historical_Phase13_G8='FAIL unchanged')
        save(case/'RESULT.json',r);cases.append(r);del model,policy,observed;gc.collect();torch.cuda.empty_cache()
    save(REPORTS/'LATENCY_ATTRIBUTION.json',dict(status='COMPLETE',binding=source,cases=cases,protocol_SHA256=sha(REPORTS/'EFFICIENCY_PROTOCOL.json'),heldout='SEALED',Full24=False,official_TEST=False))
    print('PHASE14_ISOLATED_LATENCY_COMPLETE',flush=True)

if __name__=='__main__':main()
