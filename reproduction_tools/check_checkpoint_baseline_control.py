from pathlib import Path
import os, sys, json, copy
r=Path(__file__).resolve().parents[1];repo=r/'code/GMT';os.chdir(repo)
sys.path[:0]=[str(repo),str(repo/'third_party/CenterNet2')]
os.environ['CUDA_VISIBLE_DEVICES']='0'
for key in ['HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy']:os.environ.pop(key,None)
import torch,gtr
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from centernet.config import add_centernet_config
from gtr.config import add_gtr_config
from runtime_checkpoint import install
cfg=get_cfg();add_centernet_config(cfg);add_gtr_config(cfg);cfg.merge_from_file(str(repo/'configs/VISION_stage1.yaml'))
torch.manual_seed(20260929)
a=build_model(cfg);b=copy.deepcopy(a);targets=[];a.train();b.train()
checks=[]
for step in range(2):
    x=torch.randn(2,3,256,256,device='cuda');left=x.clone().requires_grad_();right=x.clone().requires_grad_()
    a.zero_grad(set_to_none=True);b.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats();ya=a.backbone(left);sum(v.square().mean() for v in ya.values()).backward();peak_a=torch.cuda.max_memory_allocated()
    torch.cuda.reset_peak_memory_stats();yb=b.backbone(right);sum(v.square().mean() for v in yb.values()).backward();peak_b=torch.cuda.max_memory_allocated()
    for k in ya:torch.testing.assert_close(ya[k],yb[k],rtol=1e-5,atol=1e-6)
    torch.testing.assert_close(left.grad,right.grad,rtol=1e-5,atol=1e-6)
    max_grad=0.;compared=0
    for (ka,pa),(kb,pb) in zip(a.named_parameters(),b.named_parameters()):
        assert ka==kb
        assert (pa.grad is None)==(pb.grad is None)
        if pa.grad is not None:
            torch.testing.assert_close(pa.grad,pb.grad,rtol=1e-5,atol=1e-6)
            max_grad=max(max_grad,float((pa.grad-pb.grad).abs().max()));compared+=1
    for (ka,ba),(kb,bb) in zip(a.named_buffers(),b.named_buffers()):
        assert ka==kb;torch.testing.assert_close(ba,bb,rtol=0,atol=0)
    checks.append({'step':step,'parameter_gradients_compared':compared,'max_gradient_absolute_difference':max_grad,'batchnorm_buffers':'bitwise equal','baseline_peak_bytes':peak_a,'checkpoint_peak_bytes':peak_b})
    with torch.no_grad():
        for pa,pb in zip(a.parameters(),b.parameters()):
            if pa.grad is not None:pa.add_(pa.grad,alpha=-1e-4);pb.add_(pb.grad,alpha=-1e-4)
report={'status':'PASS','scope':'Two sequential synthetic backbone forward/backward steps, outputs/input and parameter gradients/BN buffers; actual ten-rank full-model smoke still required','targets':targets,'checks':checks}
(r/'manifests/checkpoint_baseline_control.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
