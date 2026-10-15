"""Paired current-mask300 and full legacy-input1/3/5epoch coverage controls."""
import argparse,csv,hashlib,json,os,signal,subprocess,time,shutil
from pathlib import Path
import torch
from cv_pil_v2_model import CandidatePredictor,parameter_digest,supervised_losses
from pirga_legacy_preprocessing import LegacyTrainingCache
from pirga_fulltrain_dataset import FullTrainDataset,load_supervision
from pirga_grouped_heads import GroupedHeads,grouped_losses
from pirga_indexed_store import sha,epoch_order,coverage


def atomic(path,value):
    q=path.with_suffix('.tmp');q.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');q.replace(path)


def execute(protocol_path,protocol_sha,out):
    if sha(protocol_path)!=protocol_sha:raise ValueError('Frozen legacy control protocol changed')
    p=json.loads(protocol_path.read_text());repo=Path(__file__).resolve().parents[1]
    if p['status']!='PREREGISTERED_LEGACY_CANONICAL_COVERAGE_CONTROL' or p['epochs']!=[1,3,5] or p['seed']!=20261015 or p['GPU_wall_hours']!=0:raise ValueError('Unregistered control')
    for name,digest in p['source_code_sha256'].items():
        if sha(repo/name)!=digest:raise ValueError('Frozen source changed '+name)
    for name in ['paired_admission','input_parity','full_admission','old300_protocol']:
        if sha(p[name]['path'])!=p[name]['sha256']:raise ValueError('Source changed '+name)
    parity=json.loads(Path(p['input_parity']['path']).read_text())
    if parity['status']!='PASS_OLD_INPUTS_FOR_CANONICAL_PAIRED_COVERAGE_CONTROL' or not parity['native_candidate_IDs_equal'] or not parity['batch_fields_bitwise_equal']:raise ValueError('No full old-input parity')
    out.mkdir(parents=True,exist_ok=False);start=time.monotonic();torch.set_num_threads(1);torch.use_deterministic_algorithms(True);stop=[False]
    for sig in [signal.SIGINT,signal.SIGTERM]:signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    status=dict(status='PREPARING_CPU',pid=os.getpid(),protocol_sha256=protocol_sha,GPU_wall_hours=0,DEV_read=False,scientific_success=False,source_commit_at_start=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip());atomic(out/'STATUS.json',status)
    ad=json.loads(Path(p['full_admission']['path']).read_text());ds=FullTrainDataset(ad['stores']);labels=load_supervision(ad['supervision']['path'],ad['supervision']['sha256'],ds)
    cache=LegacyTrainingCache(p['input_manifest']['path'],p['input_manifest']['sha256'])
    if sha(p['eligibility']['path'])!=p['eligibility']['sha256']:raise ValueError('Legacy supervision domain changed')
    keys=[tuple(k) for k in json.loads(Path(p['eligibility']['path']).read_text())];n=p['queries_per_epoch']
    if len(keys)!=n or n!=106800 or len(set(keys))!=n:raise ValueError('Complete old-interface domain changed')
    old=json.loads(Path(p['old300_protocol']['path']).read_text());schedule=[tuple(k) for k in old['schedule_query_keys']]
    if len(schedule)!=300 or not set(schedule)<=set(keys):raise ValueError('Original300 schedule or eligibility changed')
    small=out/'canonical300';small.mkdir();small_heads=[]
    for arm in ['C2','C3','C4']:
        torch.manual_seed(p['seed']);model=CandidatePredictor(use_camera=arm!='C4');initial=parameter_digest(model)
        if initial!=p['initial_parameter_sha256']:raise ValueError('Original initialization differs')
        opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);logs=[]
        for step,key in enumerate(schedule,1):
            b=cache.batch(key);label=cache.label(key,labels[key]);scores,h=model(b);rank,predict,count=supervised_losses(scores,h,b,label);loss=rank+(0 if arm=='C2' else .5)*predict
            if not torch.isfinite(loss):raise ValueError('Nonfinite canonical300 loss')
            opt.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(norm):raise ValueError('Nonfinite canonical300 gradient')
            opt.step()
            if step in [1,100,200,300]:logs.append(dict(arm=arm,step=step,loss=float(loss),rank_loss=float(rank),prediction_loss=float(predict),gradient_norm=float(norm)))
        path=small/f'{arm}_canonical300.pth';torch.save(dict(state_dict=model.state_dict(),arm=arm,seed=p['seed'],step=300,epoch=None,mode='CANONICAL_MASKED300',protocol_sha256=protocol_sha,admission_sha256=p['full_admission']['sha256']),path)
        small_heads.append(dict(arm=arm,path=str(path),sha256=sha(path),initial_parameter_sha256=initial,final_parameter_sha256=parameter_digest(model),logs=logs))
        atomic(small/'CHECKPOINTS.json',dict(checkpoints=small_heads))
    atomic(small/'RESULTS.json',dict(status='CANONICAL_MASKED300_CONTROL_TRAINED_NOT_SCIENTIFIC_SUCCESS',heads=small_heads,steps_per_arm=300,unique_queries=len(set(schedule)),GT_mask_policy='same conservative canonical as full control; no upgrade ofUNKNOWN',DEV_read=False,GPU_wall_hours=0,protocol_sha256=protocol_sha))
    status.update(status='CANONICAL300_COMPLETE_PREPARING_FULL',step=0);atomic(out/'STATUS.json',status)
    group=GroupedHeads(seed=p['seed']);optimizer=torch.optim.AdamW(group.parameters(),lr=.001,weight_decay=.0001);by={}
    for key in keys:by.setdefault(key[0],[]).append(key)
    full=out/'full';full.mkdir();step=0;epochs=[];checkpoints=[];zeros=[0,0,0];nonzeros=[0,0,0]
    try:
        with (full/'TRAINING_CURVES.csv').open('x') as f:
            fields=['arm','epoch','step','epoch_visit','rank_loss','prediction_loss','total_loss','gradient_norm'];w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader()
            for epoch in range(1,6):
                order=list(epoch_order(by,p['seed'],epoch));digest=hashlib.sha256(json.dumps(order,separators=(',',':')).encode()).hexdigest()
                if digest!=p['epoch_order_sha256'][str(epoch)]:raise ValueError('Frozen permutation changed')
                visited=[];loss_sums=[0.,0.,0.];prediction=0;epoch_start=time.monotonic()
                for visit,key in enumerate(order,1):
                    b=cache.batch(key);label=cache.label(key,labels[key]);scores,h=group(b);loss,rank,pred,count=grouped_losses(scores,h,b,label)
                    if not torch.isfinite(loss).all():raise ValueError('Nonfinite full legacy loss')
                    optimizer.zero_grad(set_to_none=True);loss.sum().backward();norm=group.clip_per_arm()
                    if not torch.isfinite(norm).all():raise ValueError('Nonfinite full legacy gradient')
                    optimizer.step();step+=1;visited.append(key);prediction+=count>0
                    values=torch.stack([loss.detach(),rank.detach(),pred.detach(),norm.detach()]).tolist()
                    for i in range(3):loss_sums[i]+=values[0][i];zeros[i]+=values[3][i]==0;nonzeros[i]+=values[3][i]>0
                    if visit==1 or visit%1000==0 or visit==n:
                        for i,arm in enumerate(['C2','C3','C4']):w.writerow(dict(arm=arm,epoch=epoch,step=step,epoch_visit=visit,total_loss=values[0][i],rank_loss=values[1][i],prediction_loss=values[2][i],gradient_norm=values[3][i]))
                        f.flush();status.update(status='FULL_LEGACY_TRAINING_CPU',epoch=epoch,step=step,epoch_visit=visit,CPU_wall_seconds=time.monotonic()-start);atomic(out/'STATUS.json',status);print(json.dumps(status),flush=True)
                    resource_stop=stop[0] or time.monotonic()-start>p['CPU_wall_hours_cap']*3600 or shutil.disk_usage(out).free<30*2**30
                    if step%10000==0 or resource_stop:
                        tmp=full/'RESUME.tmp';torch.save(dict(group=group.state_dict(),optimizer=optimizer.state_dict(),protocol_sha256=protocol_sha,step=step,epoch=epoch,visit=visit,completed_epochs=epochs,checkpoints=checkpoints,epoch_loss_sums=loss_sums,epoch_prediction_queries=prediction,zero_gradient_updates=zeros,nonzero_gradient_updates=nonzeros),tmp);tmp.replace(full/'RESUME.pth')
                    if resource_stop:
                        status.update(status='RESOURCE_STOP_NOT_COMPLETE',step=step);atomic(out/'STATUS.json',status);return
                cov=coverage(keys,visited)
                if not cov['full_epoch_pass'] or prediction!=p['prediction_queries_per_epoch']:raise ValueError('Actual old-interface supervision coverage failed')
                epochs.append(dict(epoch=epoch,coverage=cov,ranking_queries=n,prediction_queries=prediction,mean_total_loss_per_arm=[v/n for v in loss_sums],epoch_wall_seconds=time.monotonic()-epoch_start,order_sha256=digest));atomic(full/'TRAINING_EPOCH_COVERAGE.json',dict(status='IN_PROGRESS',epochs=epochs))
                if epoch in [1,3,5]:
                    for i,arm in enumerate(['C2','C3','C4']):
                        path=full/f'{arm}_epoch{epoch}.pth';torch.save(dict(state_dict=group.arm_state(i),arm=arm,seed=p['seed'],step=step,epoch=epoch,mode='LEGACY_FULL_CANONICAL',protocol_sha256=protocol_sha,admission_sha256=p['full_admission']['sha256']),path);checkpoints.append(dict(arm=arm,epoch=epoch,step=step,path=str(path),sha256=sha(path)))
                    atomic(full/'CHECKPOINTS.json',dict(checkpoints=checkpoints))
        result=dict(status='LEGACY_CANONICAL_FIXED5EPOCHS_TRAINED_NOT_SCIENTIFIC_SUCCESS',epochs=epochs,checkpoints=checkpoints,per_arm_optimizer_steps=step,parameters_per_arm=38080,zero_gradient_updates=zeros,nonzero_gradient_updates=nonzeros,canonical300_control=str(small/'RESULTS.json'),CPU_wall_seconds=time.monotonic()-start,GPU_wall_hours=0,DEV_read=False,scientific_success=False,protocol_sha256=protocol_sha)
        atomic(full/'RESULTS.json',result);status.update(status='COMPLETE_CANONICAL300_AND_FULL5EPOCHS',step=step);atomic(out/'STATUS.json',status)
    except BaseException as e:
        status.update(status='FAILED_RETAIN_EVIDENCE',step=step,reason=repr(e));atomic(out/'STATUS.json',status);raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--protocol-sha',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();execute(a.protocol,a.protocol_sha,a.out)
