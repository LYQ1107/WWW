"""One-iteration diagnostic using official ten-rank loader and model, no loss edits."""
from pathlib import Path
import os,sys,json
r=Path(__file__).resolve().parents[1];repo=r/'code/GMT';os.chdir(repo);sys.path[:0]=[str(repo),str(repo/'third_party/CenterNet2')]
import torch,train_net
from detectron2.engine import launch,default_argument_parser
from detectron2.utils import comm

def worker(args):
    import faulthandler
    stack_log=(r/f"logs/distributed_blocking_stack_rank{comm.get_rank()}.log").open("w")
    faulthandler.dump_traceback_later(45,repeat=True,file=stack_log)
    original=train_net.build_model
    def build(cfg):
        model=original(cfg);rank=comm.get_rank();record={}
        def save():
            (r/f'manifests/distributed_blocking_rank{rank}.json').write_text(json.dumps(record,indent=2))
        def inputs(module, args):
            record['frames']=[{k:x[k] for k in ['file_name','frame_id','video_id'] if k in x} for clip in args[0] for x in (clip if isinstance(clip,list) else [clip])];save()
        def proposal(module,args,result):
            proposals,loss=result
            record['proposals']=[{'count':len(x),'above_asso_threshold':int((x.objectness_logits>cfg.MODEL.ASSO_HEAD.ASSO_THRESH).sum()),'max_score':float(x.objectness_logits.max()) if len(x) else None} for x in proposals];save()
        def classify(module,args):
            x=args[0];record['reid_features']={'shape':list(x.shape),'finite':bool(torch.isfinite(x).all())};save()
            if x.numel()==0:raise RuntimeError('DIAGNOSTIC: Empty ReID features; see rank record')
        grad_log=(r/f"logs/distributed_blocking_grad_rank{rank}.log").open("w")
        def grad_hook(name,grad):
            grad_log.write(name+"\n");grad_log.flush()
            return grad
        for name,param in model.named_parameters():
            if param.requires_grad:param.register_hook(lambda grad,name=name:grad_hook(name,grad))
        model.register_forward_pre_hook(inputs)
        model.proposal_generator.register_forward_hook(proposal)
        model.roi_heads.classify_head.register_forward_pre_hook(classify)
        return model
    train_net.build_model=build
    train_net.main(args)

if __name__=='__main__':
    args=default_argument_parser().parse_args(['--num-gpus','10','--config-file',str(repo/'configs/VISION_stage1.yaml'),'SOLVER.IMS_PER_BATCH','10','SOLVER.TRAIN_ITER','1','SEED','32995656','MODEL.WEIGHTS',str(r/'checkpoints/backbone/CH_FPN_1x_key_adapted.pth'),'OUTPUT_DIR',str(r/'outputs/diagnose_blocking')])
    args.dist_url='tcp://127.0.0.1:45938'
    launch(worker,10,args=(args,),dist_url=args.dist_url)
