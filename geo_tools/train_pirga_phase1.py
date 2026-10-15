"""Preregistered full-data C2/C3/C4, identical full epochs, no DEV/model selection."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import time
import torch
from pirga_indexed_store import sha,epoch_order,coverage
from pirga_training_cache import NativeTrainingCache
from pirga_fulltrain_dataset import FullTrainDataset,load_supervision
from cv_pil_v2_model import CandidatePredictor,supervised_losses,parameter_digest


def save(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def execute(protocol_path,protocol_sha,arm,out):
    if sha(protocol_path)!=protocol_sha:raise ValueError('Unpinned training protocol')
    p=json.loads(Path(protocol_path).read_text());root=Path(__file__).resolve().parents[1]
    for name,digest in p['source_code_sha256'].items():
        if sha(root/name)!=digest:raise ValueError('Frozen training implementation changed '+name)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    if p['status']!='PREREGISTERED_BEFORE_FULLTRAIN_UPDATE' or p['seed']!=20261015 or p['epochs']!=[1,3,5] or arm not in ['C2','C3','C4'] or p['device']!='cpu':raise ValueError('Unregistered model/schedule/device')
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);started=time.monotonic();status=dict(status='PREPARING',arm=arm,protocol_sha256=protocol_sha,GPU_wall_hours=0,DEV_loaded=False)
    save(out/'STATUS.json',status)
    ad=p['admission']
    if sha(ad['path'])!=ad['sha256']:raise ValueError('Full input admission changed')
    admission=json.loads(Path(ad['path']).read_text())
    if admission['status']!='FULLTRAIN_DATA_CONTRACT_PASS':raise ValueError('Cannot train incomplete data')
    parity=p['input_executor_parity']
    if sha(parity['path'])!=parity['sha256'] or json.loads(Path(parity['path']).read_text())['status']!='PASS_REAL_ALL11_FAST_INPUT_BITWISE_PARITY':raise ValueError('New native input executor must pass real parity')
    cache=NativeTrainingCache(p['cache']['path'],p['cache']['sha256']);dataset=FullTrainDataset(admission['stores']);supervision=load_supervision(admission['supervision']['path'],admission['supervision']['sha256'],dataset)
    domain=p['common_training_domain']
    if sha(domain['path'])!=domain['sha256']:raise ValueError('Common sample domain changed')
    keys=[tuple(k) for k in json.loads(Path(domain['path']).read_text())]
    if len(keys)!=113415 or len(set(keys))!=len(keys) or set(keys)-set(cache.locations):raise ValueError('Incomplete/duplicate common domain')
    by_scene={}
    for key in keys:by_scene.setdefault(key[0],[]).append(key)
    torch.manual_seed(p['seed']);model=CandidatePredictor(use_camera=arm!='C4');initial=parameter_digest(model)
    if initial!=p['expected_initial_parameter_sha256']:raise ValueError('Old shared initialization changed')
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);step=0;epochs=[];checkpoints=[];nonzero=0;zero=0;rank_visits=pred_visits=0
    with open(out/'TRAINING_CURVES.csv','x') as f:
        fields=['arm','epoch','step','epoch_visit','rank_loss','prediction_loss','total_loss','gradient_norm_before_clip'];w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader()
        try:
            for epoch in range(1,6):
                visited=[];epoch_rank=epoch_pred=0;loss_sum=0.;epoch_start=time.monotonic();order=list(epoch_order(by_scene,p['seed'],epoch))
                order_sha=__import__('hashlib').sha256(json.dumps(order,separators=(',',':')).encode()).hexdigest()
                if order_sha!=p['epoch_order_sha256'][str(epoch)]:raise ValueError('Shared epoch permutation differs')
                for visit,key in enumerate(order,1):
                    b=cache.batch(key);label=supervision[key];scores,h=model(b);rank,pred,count=supervised_losses(scores,h,b,label)
                    has_neg=any(known and i not in label['certified_target_indices'] for i,known in enumerate(label['candidate_known_mask']))
                    epoch_rank+=bool(has_neg);epoch_pred+=count>0
                    loss=rank+(0. if arm=='C2' else .5)*pred
                    if not torch.isfinite(loss):raise ValueError('Nonfinite objective')
                    optimizer.zero_grad(set_to_none=True);loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                    if not torch.isfinite(grad):raise ValueError('Nonfinite gradient')
                    optimizer.step();step+=1;visited.append(key);loss_sum+=float(loss.detach());nonzero+=float(grad)>0;zero+=float(grad)==0
                    if step==300 or visit==1 or visit%1000==0 or visit==len(order):
                        row=dict(arm=arm,epoch=epoch,step=step,epoch_visit=visit,rank_loss=float(rank.detach()),prediction_loss=float(pred.detach()),total_loss=float(loss.detach()),gradient_norm_before_clip=float(grad))
                        w.writerow(row);f.flush();status.update(status='TRAINING',epoch=epoch,step=step,epoch_visit=visit,wall_seconds=time.monotonic()-started,last_loss=row);save(out/'STATUS.json',status)
                        print(json.dumps(row),flush=True)
                    if step==300:
                        q=out/'step300_diagnostic.pth';torch.save(dict(state_dict=model.state_dict(),arm=arm,seed=p['seed'],step=step,epoch=None,protocol_sha256=protocol_sha),q)
                        checkpoints.append(dict(kind='first300_visits_diagnostic_not_full_epoch',path=str(q.resolve()),sha256=sha(q),step=step))
                cov=coverage(keys,visited)
                if not cov['full_epoch_pass'] or epoch_rank!=113197 or epoch_pred!=73715:raise ValueError('Actual full supervision coverage changed')
                rank_visits+=epoch_rank;pred_visits+=epoch_pred
                epochs.append(dict(epoch=epoch,coverage=cov,order_sha256=order_sha,rank_objective_queries=epoch_rank,valid_prediction_target_queries=epoch_pred,
                                   C2_zero_rank_prediction_only_queries=218,mean_total_loss=loss_sum/len(order),wall_seconds=time.monotonic()-epoch_start))
                save(out/'TRAINING_EPOCH_COVERAGE.json',dict(status='IN_PROGRESS',epochs=epochs))
                if epoch in (1,3,5):
                    q=out/f'epoch{epoch}.pth';torch.save(dict(state_dict=model.state_dict(),arm=arm,seed=p['seed'],step=step,epoch=epoch,protocol_sha256=protocol_sha,
                        admission_sha256=ad['sha256']),q);checkpoints.append(dict(kind='fixed_epoch',epoch=epoch,step=step,path=str(q.resolve()),sha256=sha(q)))
                    save(out/'CHECKPOINTS.json',dict(arm=arm,checkpoints=checkpoints))
            result=dict(status='FULLTRAIN_TRAINED_FIXED5EPOCHS_NOT_SCIENTIFIC_SUCCESS',arm=arm,parameters=sum(x.numel() for x in model.parameters()),initial_parameter_sha256=initial,
                final_parameter_sha256=parameter_digest(model),protocol_sha256=protocol_sha,epochs=epochs,checkpoints=checkpoints,total_optimizer_steps=step,
                rank_objective_visits=rank_visits,valid_prediction_target_visits=pred_visits,prediction_coefficient=0. if arm=='C2' else .5,
                zero_gradient_updates=zero,nonzero_gradient_updates=nonzero,wall_seconds=time.monotonic()-started,GPU_wall_hours=0,DEV_loaded=False,scientific_success=False)
            save(out/'RESULTS.json',result);status.update(status='COMPLETE',step=step);save(out/'STATUS.json',status)
        except BaseException as e:
            status.update(status='FAILED_RETAIN_EVIDENCE',step=step,reason=repr(e));save(out/'STATUS.json',status);raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--protocol-sha',required=True);p.add_argument('--arm',required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();execute(a.protocol,a.protocol_sha,a.arm,a.out)
