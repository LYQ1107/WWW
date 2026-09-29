"""Exercise full-sized communication independently of GMT/autograd checkpointing."""
from pathlib import Path
import os,datetime,json,time
WORLD_SIZE=int(os.environ.get('GMT_WORLD_SIZE','4'))
import torch
from runtime_distributed import install_launch_backend
install_launch_backend()
from detectron2.engine import launch
from detectron2.utils import comm

def worker():
 rank=comm.get_rank();device=comm.get_local_rank();records=[]
 for size in [1,262144,6553600,16777216]:
  x=torch.full((size,),float(rank+1),device=f'cuda:{device}');torch.cuda.synchronize();t=time.monotonic()
  print('BEFORE_ALLREDUCE',rank,size,flush=True);torch.distributed.all_reduce(x);torch.cuda.synchronize()
  assert bool((x==WORLD_SIZE*(WORLD_SIZE+1)/2).all());records.append({'floats':size,'seconds':time.monotonic()-t});print('AFTER_ALLREDUCE',rank,size,flush=True);del x
 model=torch.nn.Linear(4096,4096,bias=False).cuda(device)
 ddp=torch.nn.parallel.DistributedDataParallel(model,device_ids=[device],find_unused_parameters=True,broadcast_buffers=False)
 for step in range(2):
  ddp.zero_grad();ddp(torch.full((1,4096),float(rank+1),device=f'cuda:{device}')).sum().backward();torch.cuda.synchronize();assert torch.allclose(model.weight.grad,torch.full_like(model.weight.grad,(WORLD_SIZE+1)/2));print('DDP_BACKWARD_PASS',rank,step,flush=True)
 comm.synchronize()
 if rank==0:
  out=Path(os.environ['GMT_COMM_RESULT']);out.write_text(json.dumps({'world_size':WORLD_SIZE,'status':'PASS','allreduce':records,'ddp_steps':2,'backend':torch.distributed.get_backend(),'CUDA_LAUNCH_BLOCKING':os.environ.get('CUDA_LAUNCH_BLOCKING'),'NCCL_P2P_DISABLE':os.environ.get('NCCL_P2P_DISABLE'),'NCCL_SHM_DISABLE':os.environ.get('NCCL_SHM_DISABLE')},indent=2))
if __name__=='__main__':launch(worker,WORLD_SIZE,dist_url='auto',timeout=datetime.timedelta(seconds=90))
