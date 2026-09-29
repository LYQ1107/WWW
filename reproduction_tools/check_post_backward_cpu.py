"""Native DDP versus post-backward synchronization with dynamic unused parameters."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
from pathlib import Path
import copy, datetime, json, tempfile
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from runtime_post_backward import synchronize_model, install_optimizer_hook
ROOT = Path(__file__).resolve().parents[1]
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = torch.nn.Linear(11, 17)
        self.branches = torch.nn.ModuleList([torch.nn.Linear(17, 3) for _ in range(3)])
        self.never_used = torch.nn.Linear(4, 4)
        self.register_buffer('counter', torch.tensor(5))
    def forward(self, x, branch):
        return self.branches[branch](torch.relu(self.base(x)))
class ClippedAdamW(torch.optim.AdamW):
    def step(self, closure=None):
        torch.nn.utils.clip_grad_norm_([p for g in self.param_groups for p in g['params']], .1)
        return super().step(closure)
def worker(rank, world, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+rendezvous,rank=rank,world_size=world,timeout=datetime.timedelta(seconds=120))
    torch.manual_seed(1400+rank)
    a=Model();a.counter.fill_(5+rank);b=copy.deepcopy(a)
    da=torch.nn.parallel.DistributedDataParallel(a,broadcast_buffers=False,find_unused_parameters=True,bucket_cap_mb=.001)
    synchronize_model(b)
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
    oa=ClippedAdamW(a.parameters(),lr=5e-5);ob=ClippedAdamW(b.parameters(),lr=5e-5)
    hook=install_optimizer_hook(ob,b);checks=[]
    for step in range(6):
        torch.manual_seed(2600+17*step+rank);x=torch.randn(7,11);y=torch.randn(7,3)
        branch=(rank+step)%3 if step%2==0 else step%3
        oa.zero_grad(set_to_none=True);ob.zero_grad(set_to_none=True)
        ((da(x,branch)-y)**2).mean().backward();((b(x,branch)-y)**2).mean().backward()
        oa.step();ob.step()
        error=0.;grad_error=0.;unused=0
        for pa,pb in zip(a.parameters(),b.parameters()):
            assert (pa.grad is None)==(pb.grad is None)
            if pa.grad is None:unused+=1
            else:
                torch.testing.assert_close(pa.grad,pb.grad,rtol=2e-5,atol=1e-7)
                grad_error=max(grad_error,float((pa.grad-pb.grad).abs().max()))
            torch.testing.assert_close(pa,pb,rtol=2e-5,atol=1e-7)
            error=max(error,float((pa-pb).abs().max()))
        checks.append({'step':step,'globally_unused_parameter_tensors':unused,'max_parameter_error':error,'max_clipped_gradient_error':grad_error})
    dist.barrier()
    if rank==0:
        result={'status':'PASS','device':'CPU','world_size':world,'steps':checks,'scope':'Initial state, per-rank and changing globally unused parameters, gradient clipping before AdamW, six optimizer steps; GPU/full GMT stability not established'}
        (ROOT/'manifests/post_backward_cpu_equivalence.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    hook.remove();dist.destroy_process_group()
if __name__=='__main__':
    with tempfile.TemporaryDirectory(prefix='gmt_post_backward_') as temp:
        mp.spawn(worker,args=(3,str(Path(temp)/'rendezvous')),nprocs=3,join=True)
