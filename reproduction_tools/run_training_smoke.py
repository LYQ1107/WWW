"""20-iteration official training smoke, isolated outputs; never formal checkpoint."""
from pathlib import Path
import os,sys,json,re
r=Path('/data3/liuyeqiang/GMT_VisionTrack_repro');repo=r/'code/GMT'
os.chdir(repo);sys.path[:0]=[str(repo),str(repo/'third_party/CenterNet2')]
for key in ['HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy']:os.environ.pop(key,None)
os.environ.setdefault('CUDA_VISIBLE_DEVICES','0,1,2,3')
WORLD_SIZE=int(os.environ.get('GMT_WORLD_SIZE','4')); BATCH_SIZE=int(os.environ.get('GMT_BATCH_SIZE','4'))
assert WORLD_SIZE==4 and BATCH_SIZE==4 and os.environ['CUDA_VISIBLE_DEVICES']=='0,1,2,3'
import torch
import train_net
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import launch,default_argument_parser
from detectron2.utils import comm

class SmokeCheckpointer(DetectionCheckpointer):
    latest=None
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);SmokeCheckpointer.latest=self
    def _load_model(self,checkpoint):
        incompatible=super()._load_model(checkpoint)
        if os.environ['GMT_SMOKE_STAGE']=='2' and (incompatible.missing_keys or incompatible.unexpected_keys or getattr(incompatible,'incorrect_shapes',[])):
            raise RuntimeError('Stage2 trained Stage1 checkpoint incompatibility: '+str(incompatible))
        return incompatible

def worker(args):
    train_net.DetectionCheckpointer=SmokeCheckpointer
    train_net.main(args)
    comm.synchronize()
    if comm.is_main_process():
        # Extra checkpoint only in isolated smoke outputs: official loop saves every500.
        SmokeCheckpointer.latest.save('smoke_final',iteration=19)
    comm.synchronize()

if __name__=='__main__':
    stage=int(os.environ.get('GMT_SMOKE_STAGE','1'));assert stage in (1,2)
    os.environ['GMT_SMOKE_STAGE']=str(stage)
    data_state=json.loads((r/'manifests/data_preparation_state.json').read_text())
    assert data_state['status']=='complete_pending_loader_smoke'
    variant=os.environ.get('GMT_SMOKE_VARIANT','')
    assert not variant or re.fullmatch(r'[a-z0-9_]+',variant)
    prefix=variant+'_' if variant else ''
    suffix='_'+variant if variant else ''
    out=r/f'outputs/smoke_{prefix}stage{stage}'
    if out.exists() and any(out.iterdir()):raise RuntimeError('Smoke output already exists: inspect prior attempt before reuse')
    weight=r/('checkpoints/backbone/CH_FPN_1x_key_adapted.pth' if stage==1 else 'outputs/stage1/model_20000.pth')
    assert weight.is_file()
    args=default_argument_parser().parse_args(['--num-gpus',str(WORLD_SIZE),'--config-file',str(repo/f'configs/VISION_stage{stage}.yaml'),'SOLVER.IMS_PER_BATCH',str(BATCH_SIZE),'SOLVER.TRAIN_ITER','20','MODEL.WEIGHTS',str(weight),'OUTPUT_DIR',str(out)])
    args.eval_only=False;args.dist_url='tcp://127.0.0.1:{}'.format(torch.randint(11111,60000,(1,))[0].item())
    print('ISOLATED SMOKE; 20 iterations; official MAX_ITER/scheduler retained; not a formal Stage checkpoint.',flush=True)
    print(args,flush=True)
    launch(worker,args.num_gpus,num_machines=args.num_machines,machine_rank=args.machine_rank,dist_url=args.dist_url,args=(args,))
    checkpoint=torch.load(out/'smoke_final.pth',map_location='cpu')
    assert checkpoint['iteration']==19 and all(k in checkpoint for k in ['model','optimizer','scheduler'])
    assert all(torch.isfinite(v).all() for v in checkpoint['model'].values() if torch.is_tensor(v) and (v.is_floating_point() or v.is_complex()))
    (r/f'manifests/stage{stage}_smoke{suffix}_pass.json').write_text(json.dumps({'world_size':WORLD_SIZE,'batch_size':BATCH_SIZE,'stage':stage,'iterations':20,'checkpoint_reload':'PASS','model_finiteness':'PASS','output':str(out),'formal_training':False,'runtime':{'backend':os.environ.get('GMT_DISTRIBUTED_BACKEND'),'CUDA_LAUNCH_BLOCKING':os.environ.get('CUDA_LAUNCH_BLOCKING'),'GMT_SYNC_DDP_BUCKETS':os.environ.get('GMT_SYNC_DDP_BUCKETS','0'),'GMT_CPU_COLLECTIVES':os.environ.get('GMT_CPU_COLLECTIVES','0'),'GMT_CPU_DDP_INIT':os.environ.get('GMT_CPU_DDP_INIT','0'),'GMT_POST_BACKWARD_CPU':os.environ.get('GMT_POST_BACKWARD_CPU','0'),'CUDA_MODULE_LOADING':os.environ.get('CUDA_MODULE_LOADING'),'CUDA_MODULE_DATA_LOADING':os.environ.get('CUDA_MODULE_DATA_LOADING')} },indent=2))
    print('TRAINING_SMOKE_PASS',stage,flush=True)
