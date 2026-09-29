"""Tiny ten-GPU NCCL/DDP runtime test; no GMT model training or dataset usage."""
from pathlib import Path
import os,json,datetime
os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3,4,5,6,7,8,9'
import torch
from torch.nn.parallel import DistributedDataParallel
from detectron2.engine import launch
from detectron2.utils import comm

def worker():
 rank=comm.get_rank();local=comm.get_local_rank();world=comm.get_world_size()
 x=torch.tensor(float(rank+1),device=f'cuda:{local}')
 torch.distributed.all_reduce(x)
 assert x.item()==55 and world==10
 model=torch.nn.Linear(2,1,bias=False).cuda(local)
 ddp=DistributedDataParallel(model,device_ids=[local],broadcast_buffers=False,find_unused_parameters=True)
 inputs=torch.full((1,2),float(rank+1),device=f'cuda:{local}')
 ddp(inputs).sum().backward()
 assert torch.allclose(model.weight.grad,torch.full_like(model.weight.grad,5.5))
 torch.cuda.synchronize();comm.synchronize()
 print(f'rank={rank} NCCL sum55 and DDP average gradient5.5 PASS',flush=True)
 if rank==0:
  Path('/data3/liuyeqiang/GMT_VisionTrack_repro/manifests/distributed_smoke_pass.json').write_text(json.dumps({'world_size':world,'nccl_sum':55,'ddp_average_gradient':5.5,'timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat(),'GMT_training':False},indent=2))
if __name__=='__main__':
 launch(worker,10,num_machines=1,machine_rank=0,dist_url='auto',args=(),timeout=datetime.timedelta(seconds=45))
