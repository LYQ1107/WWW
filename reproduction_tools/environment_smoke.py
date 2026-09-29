"""Verify runtime and official evaluator installation; does not start training."""
from pathlib import Path
import os,sys,json,hashlib,inspect,shutil
r=Path('/data3/liuyeqiang/GMT_VisionTrack_repro');repo=r/'code/GMT'
os.chdir(repo)
import torch,torchvision,torchaudio,numpy,cv2,detectron2
installed=Path(inspect.getfile(detectron2)).parent
original=installed/'evaluation/evaluator.py'
backup=original.with_name('evaluator.py.original')
if not backup.exists():shutil.copy2(original,backup)
source=repo/'evaluator.py'
shutil.copy2(source,original)
rows=[]
for p in [backup,source,original]:rows.append(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p))
(r/'manifests/evaluator_replacement.sha256').write_text('\n'.join(rows)+'\n')
assert hashlib.sha256(source.read_bytes()).digest()==hashlib.sha256(original.read_bytes()).digest()
print('Installed Detectron2:',installed)
print('Exact official evaluator copied; original backup:',backup)
print('torch',torch.__version__,'torchvision',torchvision.__version__,'torchaudio',torchaudio.__version__,'cuda',torch.version.cuda,'numpy',numpy.__version__,'opencv',cv2.__version__)
assert torch.__version__.startswith('2.0.0') and torch.version.cuda=='11.8'
assert torch.cuda.is_available()
print('gpu_count',torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
 with torch.cuda.device(i):
  x=torch.ones(16,device=f'cuda:{i}',requires_grad=True);(x*x).sum().backward();torch.cuda.synchronize()
 print(i,torch.cuda.get_device_name(i),'allocation and backward PASS')
sys.path.insert(0,str(repo));sys.path.insert(0,str(repo/'third_party/CenterNet2'))
import train_net,test_net,gtr
from detectron2.layers import DeformConv
conv=DeformConv(3,4,kernel_size=3,padding=1).cuda(0)
x=torch.randn(1,3,8,8,device='cuda:0',requires_grad=True)
off=torch.zeros(1,18,8,8,device='cuda:0',requires_grad=True)
conv(x,off).sum().backward();torch.cuda.synchronize()
print('Detectron2 CUDA deformable convolution forward/backward PASS')
from detectron2.config import get_cfg
from centernet.config import add_centernet_config
from gtr.config import add_gtr_config
for stage in [1,2]:
 cfg=get_cfg();add_centernet_config(cfg);add_gtr_config(cfg);cfg.merge_from_file(str(repo/f'configs/VISION_stage{stage}.yaml'))
 cfg.merge_from_list(['SOLVER.IMS_PER_BATCH','10','MODEL.WEIGHTS',str(r/('checkpoints/backbone/CH_FPN_1x.pth' if stage==1 else 'outputs/stage1/model_20000.pth')),'OUTPUT_DIR',str(r/f'outputs/stage{stage}')])
 assert cfg.SOLVER.MAX_ITER==20000 and cfg.SOLVER.OPTIMIZER=='ADAMW'
 assert cfg.SOLVER.IMS_PER_BATCH//10==1
 (r/f'reports/config_snapshots/resolved_stage{stage}.yaml').write_text(cfg.dump())
 print('stage',stage,'MAX_ITER',cfg.SOLVER.MAX_ITER,'LR',cfg.SOLVER.BASE_LR,'SEED',cfg.SEED,'TRAIN_ITER',cfg.SOLVER.TRAIN_ITER)
print('ENVIRONMENT_SMOKE_PASS (data loader and full model smoke remain pending dataset)')
