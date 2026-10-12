"""Actual model loads and parameter reads on real TRAIN camera payloads.

Forward-active is observed tensor storage read by PyTorch operators, including
functional/direct parameter accesses missed by module hooks. It is prefix scoped.
MAC counts cover dense matrix, convolution and attention operators, not all custom
deformable/ROI kernels or pointwise FLOPs. Full-video latency is a separate task.
"""
import collections
import gc
import traceback
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten
from jev_oneweek_common import *


class ParameterReads(TorchDispatchMode):
    def __init__(self, modules):
        super().__init__()
        self.parameters={}
        for prefix, module in modules.items():
            if module is None:continue
            for name,p in module.named_parameters():
                self.parameters.setdefault(id(p),(prefix+'.'+name,p))
        self.by_storage=collections.defaultdict(set)
        for key,(name,p) in self.parameters.items():
            self.by_storage[p.untyped_storage().data_ptr()].add(key)
        self.active=set();self.macs=collections.Counter();self.operator_counts=collections.Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs=kwargs or {}
        for value in tree_flatten((args,kwargs))[0]:
            if isinstance(value,torch.Tensor) and value.layout==torch.strided:
                self.active.update(self.by_storage.get(value.untyped_storage().data_ptr(),()))
        result=func(*args,**kwargs)
        name=str(func);self.operator_counts[name]+=1
        if name in ('aten.mm.default','aten.bmm.default'):
            a,b=args[:2];self.macs['matrix']+=a.numel()*b.shape[-1]
        elif name=='aten.addmm.default':
            a,b=args[1:3];self.macs['matrix']+=a.numel()*b.shape[-1]
        elif name=='aten.convolution.default':
            weight=args[1];groups=args[8]
            self.macs['convolution']+=result.numel()*weight[0].numel()
        elif '_scaled_dot_product' in name and 'backward' not in name:
            q,k=args[:2]
            self.macs['fused_attention']+=2*q.numel()*k.shape[-2]
        return result

    def report(self):
        total=sum(p.numel() for _,p in self.parameters.values())
        active=sum(self.parameters[k][1].numel() for k in self.active)
        return dict(total_loaded_parameters=total,total_registered_models_union=total,
                    trainable_at_measurement=sum(p.numel() for _,p in self.parameters.values() if p.requires_grad),
                    actual_forward_active_parameters=active,dormant_observed=total-active,
                    active_parameter_names=[self.parameters[k][0] for k in sorted(self.active)],
                    registered_parameters_by_module={name:sum(p.numel() for p in module.parameters())
                                                      for name,module in self.modules.items()} if hasattr(self,'modules') else None,
                    counted_MACs=dict(self.macs), counted_dense_FLOPs=2*sum(self.macs.values()),
                    FLOP_completeness='PARTIAL_NATIVE_MHA_CUSTOM_DEFORMABLE_ROI_POINTWISE_NOT_COUNTED',
                    operator_counts=dict(self.operator_counts),
                    native_MHA_operator_calls_not_in_MAC_total=sum(n for name,n in self.operator_counts.items() if '_native_multi_head_attention' in name),
                    observed_scope='TRAIN12 first2 actual scene frames, real image to native tracks; bank-specific branches may be unobserved')


def original_model():
    from detectron2.config import get_cfg
    from detectron2.modeling import build_model
    from centernet.config import add_centernet_config
    from gtr.config import add_gtr_config
    from jev_phase13_common import STAGE2
    from gtr.modeling.jev_runtime import JEVRuntimePolicy
    cfg=get_cfg();add_centernet_config(cfg);add_gtr_config(cfg)
    cfg.merge_from_file(str(ROOT/'configs/VISION_test.yaml'));cfg.defrost()
    cfg.MODEL.JEV.ENABLED=False;cfg.MODEL.WEIGHTS=str(STAGE2);cfg.freeze()
    model=build_model(cfg).cuda().eval()
    model.load_state_dict(torch.load(STAGE2,map_location='cpu')['model'],strict=True)
    model.jev_enabled=True;model.jev_mode='off';model.jev_policy=JEVRuntimePolicy('off',None)
    model.jev_candidate_policy=None;model.visual_jev_enabled=False
    model.jev_perception_cache_reader=None;model.jev_stage2_executor=None
    for p in model.parameters():p.requires_grad_(False)
    return model,cfg


def original_live_payloads(model, values, cfg):
    """Decode each current image just before the unchanged original inference."""
    import numpy as np
    from PIL import Image
    from gtr.data.transforms.custom_augmentation_impl import EfficientDetResizeCrop
    inference=model.inference
    def live(payloads,*args,**kwargs):
        ready=[]
        for payload in payloads:
            rgb=np.array(Image.open(payload['phase13_image_path']).convert('RGB'))
            rgb=EfficientDetResizeCrop(cfg.INPUT.TEST_SIZE,(1.,1.)).get_transform(rgb).apply_image(rgb)
            item=dict(payload,image=torch.as_tensor(np.ascontiguousarray(rgb.transpose(2,0,1))))
            ready.append(item)
        return inference(ready,*args,**kwargs)
    model.inference=live
    return model


def main():
    protect();seed_all()
    from jev_phase13_runtime import build_tracker,cache_inputs,run
    from jev_phase15_common import load_old_policy
    from jev_phase15_evaluate import load_policy
    from jev_phase14_latency import prune
    from gtr.modeling.jev_phase15.native_commit_adapter import attach
    values,frames,reader=cache_inputs(12)
    results=[]
    cases=['original_GMT','cosine','fixed_question','multi_question','set_transformer','XV_V3']
    for case in cases:
        seed_all()
        checkpoint=None;policy=None;removed=0
        if case=='original_GMT':
            model,cfg=original_model();original_live_payloads(model,values,cfg)
        else:
            if case=='XV_V3':policy,checkpoint,_=load_policy('F_full',20261009,3,'pilot')
            elif case!='cosine':policy,checkpoint,_=load_old_policy(case,20261009)
            if policy is not None:
                for p in policy.parameters():p.requires_grad_(False)
            model=build_tracker(12,policy=policy,react_learned=False,live=True)
            if case=='XV_V3':attach(model)
            loaded_before=sum(p.numel() for p in model.parameters())
            removed=prune(model)['removed_total']
        modules=dict(tracker=model,policy=policy)
        tracker=ParameterReads(modules)
        with torch.no_grad(),tracker:
            raw,_=run(model,values,frames,stop=1)
        torch.cuda.synchronize()
        report=tracker.report()
        report['registered_parameters_by_module']={key:sum(p.numel() for p in value.parameters())
                                                   for key,value in modules.items() if value is not None}
        report.update(case=case,checkpoint=checkpoint,removed_GTA_parameters=removed,
                      policy_parameters=sum(p.numel() for p in policy.parameters()) if policy is not None else 0,
                      expected_XIV_registered=4231173 if case=='multi_question' else None,
                      expected_XV_registered=5923209 if case=='XV_V3' else None,
                      no_GT_actor_inputs=True)
        results.append(report)
        save(OUT/'parameter_count/PROGRESS.json',dict(status='RUNNING',cases=results))
        print('ACTUAL_PARAMETER_COUNT',case,report['total_loaded_parameters'],report['actual_forward_active_parameters'],flush=True)
        del raw,model,policy,modules,tracker
        gc.collect();torch.cuda.empty_cache()
    base=results[0]['total_loaded_parameters']
    for item in results:item['net_total_parameters_vs_actual_original_GMT']=item['total_loaded_parameters']-base
    result=dict(status='COMPLETE_EXISTING_MODEL_COUNTS',binding=binding('Actual PyTorch loads + real TRAIN12 prefix parameter reads'),
                cases=results,new_JEVTrack=dict(status='NOT_RUN',reason='New architecture not implemented yet',parameters=None),
                FPS=dict(status='NOT_RUN',reason='Complete-video three-restart same-GPU task still pending',value=None),
                hardware=torch.cuda.get_device_name(),torch=str(torch.__version__),CUDA=torch.version.cuda,
                partial_FLOPs_not_claimed_complete=True,
                counting_note='Plain native executor policies are not child nn.Modules of GTRRCNN; loaded/registered model union explicitly includes policy. Forward read counts include tensor aliases/direct weight accesses.')
    save(REPORTS/'PARAMETERS_AND_FPS.json',result)
    registry=read(REPORTS/'EXPERIMENT_REGISTRY.json');registry['jobs']['parameter_count'].update(status='COMPLETE_EXISTING_MODELS_NEW_PENDING',results=ref(REPORTS/'PARAMETERS_AND_FPS.json'));save(REPORTS/'EXPERIMENT_REGISTRY.json',registry)


if __name__=='__main__':
    try:main()
    except Exception:
        failure('parameter_count',traceback.format_exc());raise
