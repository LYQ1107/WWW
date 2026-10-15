"""Five full shared epochs of independent C2/C3/C4, model-wise vectorized.

Query batch size stays1 per model. Each slice has independent AdamW state and
per-arm norm clipping. A budget signal checkpoints after the current completed
step; the same joint state can resume on CPU, without a new epoch or DEV tuning.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import time
import torch
from pirga_indexed_store import sha,epoch_order,coverage
from pirga_training_cache import NativeTrainingCache
from pirga_fulltrain_dataset import FullTrainDataset,load_supervision
from pirga_grouped_heads import GroupedHeads,grouped_losses
from cv_pil_native_handoff import safe_torch_load


def save(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def state_digest(state):
    h=hashlib.sha256()
    for n,x in state.items():h.update(n.encode());h.update(x.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def execute(protocol_path,protocol_sha,out,device,resume=None):
    if sha(protocol_path)!=protocol_sha:raise ValueError('Frozen training protocol changed')
    p=json.loads(Path(protocol_path).read_text());root=Path(__file__).resolve().parents[1]
    for n,d in p['source_code_sha256'].items():
        if sha(root/n)!=d:raise ValueError('Training source changed '+n)
    if p['status']!='PREREGISTERED_BEFORE_FULLTRAIN_UPDATE' or p['seed']!=20261015 or p['epochs']!=[1,3,5] or device not in p['allowed_devices']:raise ValueError('Unregistered schedule/device')
    torch.set_num_threads(p.get('CPU_resume_threads',8) if device=='cpu' else 1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    out.mkdir(parents=True,exist_ok=resume is not None);started=time.monotonic();stop=[False]
    for s in (signal.SIGINT,signal.SIGTERM):signal.signal(s,lambda *_:stop.__setitem__(0,True))
    status=dict(status='PREPARING',device=device,protocol_sha256=protocol_sha,DEV_read=False,source_commit_at_start=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip())
    save(out/'STATUS.json',status)
    for field,expected in [('input_executor_parity','PASS_REAL_ALL11_FAST_INPUT_BITWISE_PARITY'),('grouped_math_shadow','PASS_REAL_TRAIN_GROUPED_INDEPENDENT_HEAD_SHADOW')]:
        artifact=p[field]
        if sha(artifact['path'])!=artifact['sha256'] or json.loads(Path(artifact['path']).read_text())['status']!=expected:raise ValueError('Unverified training engine '+field)
    if sha(p['admission']['path'])!=p['admission']['sha256']:raise ValueError('Admission changed')
    ad=json.loads(Path(p['admission']['path']).read_text());dataset=FullTrainDataset(ad['stores']);labels=load_supervision(ad['supervision']['path'],ad['supervision']['sha256'],dataset)
    cache=NativeTrainingCache(p['cache']['path'],p['cache']['sha256'],device=device)
    if len(cache.locations)!=154911:raise ValueError('Complete input query cache coverage changed')
    if sha(p['common_training_domain']['path'])!=p['common_training_domain']['sha256']:raise ValueError('Common supervision domain changed')
    keys=[tuple(k) for k in json.loads(Path(p['common_training_domain']['path']).read_text())]
    if len(keys)!=113415 or len(set(keys))!=113415:raise ValueError('Common full epoch changed')
    by={}
    for key in keys:by.setdefault(key[0],[]).append(key)
    group=GroupedHeads(seed=p['seed']).to(device);initials=[state_digest(group.arm_state(i)) for i in range(3)]
    if set(initials)!={p['expected_initial_parameter_sha256']}:raise ValueError('Shared original initialization differs')
    optimizer=torch.optim.AdamW(group.parameters(),lr=.001,weight_decay=.0001)
    step=0;completed_epochs=[];checkpoints=[];start_epoch=1;skip=0;epoch_rank=epoch_pred=0;loss_sums=[0.,0.,0.];zero=[0,0,0];nonzero=[0,0,0]
    if resume:
        state=safe_torch_load(resume)
        if state['protocol_sha256']!=protocol_sha:raise ValueError('Resume belongs to another protocol')
        group.load_state_dict(state['group_state']);optimizer.load_state_dict(state['optimizer_state']);step=state['step'];completed_epochs=state['completed_epochs'];checkpoints=state['checkpoints']
        start_epoch=state['epoch'];skip=state['epoch_visit'];epoch_rank=state['epoch_rank'];epoch_pred=state['epoch_pred'];loss_sums=state['loss_sums'];zero=state['zero'];nonzero=state['nonzero']
    def snapshot(epoch,visit):
        if device=='cuda':torch.cuda.synchronize()
        path=out/'RESUME.pth';temp=path.with_suffix('.tmp')
        torch.save(dict(group_state=group.state_dict(),optimizer_state=optimizer.state_dict(),protocol_sha256=protocol_sha,step=step,epoch=epoch,epoch_visit=visit,
            completed_epochs=completed_epochs,checkpoints=checkpoints,epoch_rank=epoch_rank,epoch_pred=epoch_pred,loss_sums=loss_sums,zero=zero,nonzero=nonzero),temp);temp.replace(path)
    def save_heads(kind,epoch):
        for i,arm in enumerate(('C2','C3','C4')):
            path=out/f'{arm}_{kind}.pth';weights=group.arm_state(i)
            torch.save(dict(state_dict=weights,arm=arm,seed=p['seed'],step=step,epoch=epoch,protocol_sha256=protocol_sha,admission_sha256=p['admission']['sha256']),path)
            checkpoints.append(dict(arm=arm,kind=kind,epoch=epoch,step=step,path=str(path.resolve()),sha256=sha(path),parameter_sha256=state_digest(weights)))
        save(out/'CHECKPOINTS.json',dict(checkpoints=checkpoints))
    with open(out/'TRAINING_CURVES.csv','a' if resume else 'x') as f:
        fields=['arm','epoch','step','epoch_visit','rank_loss','prediction_loss','total_loss','gradient_norm_before_clip'];writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n')
        if not resume:writer.writeheader()
        try:
            for epoch in range(start_epoch,6):
                order=list(epoch_order(by,p['seed'],epoch));digest=hashlib.sha256(json.dumps(order,separators=(',',':')).encode()).hexdigest()
                if digest!=p['epoch_order_sha256'][str(epoch)]:raise ValueError('Preregistered permutation changed')
                visited=order[:skip];epoch_start=time.monotonic()
                if not skip:epoch_rank=epoch_pred=0;loss_sums=[0.,0.,0.]
                for visit,key in enumerate(order[skip:],skip+1):
                    batch=cache.batch(key);label=labels[key];scores,h=group(batch);loss,rank,pred,count=grouped_losses(scores,h,batch,label)
                    if not torch.isfinite(loss).all():raise ValueError('Nonfinite objective')
                    optimizer.zero_grad(set_to_none=True);loss.sum().backward();grad=group.clip_per_arm()
                    if not torch.isfinite(grad).all():raise ValueError('Nonfinite gradient')
                    optimizer.step();step+=1;visited.append(key)
                    values=torch.stack((loss.detach(),rank.detach(),pred.detach(),grad.detach())).cpu().tolist()
                    for i in range(3):loss_sums[i]+=values[0][i];zero[i]+=values[3][i]==0;nonzero[i]+=values[3][i]>0
                    pos=set(label['certified_target_indices']);epoch_rank+=any(v and i not in pos for i,v in enumerate(label['candidate_known_mask']));epoch_pred+=count>0
                    if step==300:save_heads('step300_diagnostic',None)
                    if visit==1 or visit%1000==0 or visit==len(order):
                        for i,arm in enumerate(('C2','C3','C4')):writer.writerow(dict(arm=arm,epoch=epoch,step=step,epoch_visit=visit,rank_loss=values[1][i],prediction_loss=values[2][i],total_loss=values[0][i],gradient_norm_before_clip=values[3][i]))
                        f.flush();status.update(status='TRAINING',epoch=epoch,step=step,epoch_visit=visit,wall_seconds=time.monotonic()-started,last_loss=values[0]);save(out/'STATUS.json',status)
                        print(json.dumps(dict(epoch=epoch,step=step,epoch_visit=visit,loss=values[0],seconds=time.monotonic()-started)),flush=True)
                    if step%10000==0:snapshot(epoch,visit)
                    if stop[0]:
                        snapshot(epoch,visit);status.update(status='RESOURCE_CHECKPOINT_READY_FOR_CPU_CONTINUATION',epoch=epoch,step=step,epoch_visit=visit,resume_sha256=sha(out/'RESUME.pth'));save(out/'STATUS.json',status);return
                cov=coverage(keys,visited)
                if not cov['full_epoch_pass'] or epoch_rank!=113197 or epoch_pred!=73715:raise ValueError('Actual epoch/objective coverage incomplete')
                completed_epochs.append(dict(epoch=epoch,device=device,coverage=cov,order_sha256=digest,rank_queries=epoch_rank,prediction_queries=epoch_pred,
                                             per_arm_mean_total_loss=[v/len(order) for v in loss_sums],epoch_wall_seconds=time.monotonic()-epoch_start))
                save(out/'TRAINING_EPOCH_COVERAGE.json',dict(status='IN_PROGRESS',epochs=completed_epochs))
                if epoch in (1,3,5):save_heads(f'epoch{epoch}',epoch)
                skip=0;snapshot(epoch+1,0)
            result=dict(status='FULLTRAIN_TRAINED_FIXED5EPOCHS_NOT_SCIENTIFIC_SUCCESS',protocol_sha256=protocol_sha,parameters_per_arm=38080,per_arm_query_batch_size=1,
                shared_initial_parameter_sha256=initials,per_arm_optimizer_steps=step,epochs=completed_epochs,checkpoints=checkpoints,zero_gradient_updates=zero,nonzero_gradient_updates=nonzero,
                final_parameter_sha256=[state_digest(group.arm_state(i)) for i in range(3)],models_independent=True,per_arm_clipping=True,DEV_read=False,scientific_success=False,wall_seconds=time.monotonic()-started)
            save(out/'RESULTS.json',result);status.update(status='COMPLETE',step=step,epoch=5);save(out/'STATUS.json',status)
        except BaseException as e:
            status.update(status='FAILED_RETAIN_EVIDENCE',step=step,reason=repr(e));save(out/'STATUS.json',status);raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--protocol-sha',required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--device',choices=['cpu','cuda'],required=True);p.add_argument('--resume',type=Path)
    a=p.parse_args();execute(a.protocol,a.protocol_sha,a.out,a.device,a.resume)
