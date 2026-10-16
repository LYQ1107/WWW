"""Engineering-only CPU/GPU checks on actual native risk evidence and logits.

External observers use plain-function hooks so native state snapshots never
deepcopy an observer/model/file graph. No timing here is formal method FPS.
"""
import torch
from gtr.modeling.identity_revision import controller as controller_module
from gtr.modeling.identity_revision.model import load_native_risk


class RiskTrace:
    def __init__(self,controller,path,digest):
        self.controller=controller;self.cpu,self.threshold=load_native_risk(path,digest)
        self.original=None;self.hook=None
        self.stats=dict(evidence_calls=0,model_forward_calls=0,accepted_model_rows=0,
            evidence_CPU_GPU_max_abs_error=0.,logit_CPU_GPU_max_abs_error=0.,threshold_mask_disagreements=0)

    def install(self):
        if self.original is not None:raise ValueError('Risk trace already installed')
        self.original=controller_module.native_conflicts
        def evidence(*args,**kwargs):
            result=self.original(*args,**kwargs)
            cpu_args=[a.detach().cpu() if isinstance(a,torch.Tensor) else a for a in args]
            cpu_kwargs={k:v.detach().cpu() if isinstance(v,torch.Tensor) else v for k,v in kwargs.items()}
            expected=self.original(*cpu_args,**cpu_kwargs)[1];actual=result[1].detach().cpu()
            error=(actual-expected).abs();tolerance=8*torch.finfo(torch.float32).eps*(expected.abs()+3)
            if (error>tolerance).any():raise RuntimeError('Live native risk features differ from CPU formula')
            self.stats['evidence_calls']+=1
            self.stats['evidence_CPU_GPU_max_abs_error']=max(self.stats['evidence_CPU_GPU_max_abs_error'],float(error.max()) if error.numel() else 0.)
            return result
        controller_module.native_conflicts=evidence
        # A bound-method hook would deepcopy its observer on StateTransaction.
        self.hook=self.controller.risk_head.register_forward_hook(lambda module,args,output:self.check_logits(args[0],output))

    def check_logits(self,features,output):
        with torch.inference_mode():expected=self.cpu(features.detach().cpu())
        actual=output.detach().cpu();error=(actual-expected).abs()
        if (error>1e-5*(1+expected.abs())).any():raise RuntimeError('Live frozen risk logits differ from CPU checkpoint')
        self.stats['model_forward_calls']+=1;self.stats['accepted_model_rows']+=len(features)
        self.stats['logit_CPU_GPU_max_abs_error']=max(self.stats['logit_CPU_GPU_max_abs_error'],float(error.max()) if error.numel() else 0.)
        self.stats['threshold_mask_disagreements']+=int(((actual>self.threshold)!=(expected>self.threshold)).sum())

    def close(self):
        if self.original is not None:controller_module.native_conflicts=self.original;self.original=None
        if self.hook is not None:
            # Restore can replace the risk module with its snapshot clone;
            # remove this observer from that clone as well as the old handle.
            self.controller.risk_head._forward_hooks.pop(self.hook.id,None)
            self.hook.remove();self.hook=None
