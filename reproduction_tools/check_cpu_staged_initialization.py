"""Compare the CPU-staged averaging hook with native DDP averaging; CPU only."""
from pathlib import Path
import copy,datetime,json,os,tempfile
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1'
import torch,torch.distributed as dist,torch.multiprocessing as mp
from runtime_distributed import install_cpu_bucket_hook, install_cpu_ddp_initialization
R=Path(__file__).resolve().parents[1]
class Model(torch.nn.Module):
 def __init__(self):
  super().__init__();self.layers=torch.nn.Sequential(torch.nn.Linear(128,128),torch.nn.ReLU(),torch.nn.Linear(128,128),torch.nn.ReLU(),torch.nn.Linear(128,13));self.unused=torch.nn.Linear(7,7);self.register_buffer("counter",torch.tensor(7,dtype=torch.int64));self.register_buffer("scale",torch.tensor(1.25))
 def forward(self,x):return self.layers(x)
def worker(rank,world,path):
 torch.set_num_threads(1);dist.init_process_group('gloo',init_method='file://'+path,rank=rank,world_size=world,timeout=datetime.timedelta(seconds=90))
 torch.manual_seed(1024+rank);a=Model();a.counter.fill_(rank+7);a.scale.fill_(rank+1.25);b=copy.deepcopy(a)
 da=torch.nn.parallel.DistributedDataParallel(a,find_unused_parameters=True,bucket_cap_mb=.02)
 install_cpu_ddp_initialization()
 db=torch.nn.parallel.DistributedDataParallel(b,find_unused_parameters=True,bucket_cap_mb=.02)
 for key,value in a.state_dict().items():
  assert torch.equal(value,b.state_dict()[key]),key
 assert b.counter.item()==7 and b.scale.item()==1.25
 install_cpu_bucket_hook(db);checks=[]
 for step in range(3):
  torch.manual_seed(600+rank+step*17);x=torch.randn(5,128)+rank*.13;y=torch.randn(5,13)
  da.zero_grad(set_to_none=True);db.zero_grad(set_to_none=True)
  ((da(x)-y)**2).mean().backward();((db(x)-y)**2).mean().backward()
  error=0.;count=0
  for pa,pb in zip(a.parameters(),b.parameters()):
   assert (pa.grad is None)==(pb.grad is None)
   if pa.grad is not None:
    torch.testing.assert_close(pa.grad,pb.grad,rtol=1e-6,atol=1e-8);error=max(error,float((pa.grad-pb.grad).abs().max()));count+=1
    with torch.no_grad():pa.add_(pa.grad,alpha=-.01);pb.add_(pb.grad,alpha=-.01)
  checks.append({'step':step,'compared_gradient_groups':count,'max_absolute_error':error})
 dist.barrier()
 if rank==0:
  record={'status':'PASS','backend':'gloo','device':'CPU','world_size':world,'steps':checks,'initial_parameters_and_mixed_dtype_buffers':'BITWISE_EQUAL_TO_NATIVE_DDP', 'scope':'Three distinct rank initial states synchronized identically; gradient averaging, unused parameters and multiple buckets. CPU only, no GPU viability claim.'}
  (R/'manifests/cpu_staged_initialization_equivalence.json').write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)
 dist.destroy_process_group()
if __name__=='__main__':
 with tempfile.TemporaryDirectory(prefix='gmt_bucket_check_') as path:mp.spawn(worker,args=(3,str(Path(path)/'rendezvous')),nprocs=3,join=True)
