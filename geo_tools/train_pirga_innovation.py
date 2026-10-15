"""Fixed full-data Phase2 CPU experiment, all seven matched controls together."""
import argparse,csv,hashlib,json,os,resource,shutil,signal,subprocess,time
from collections import Counter
from pathlib import Path
import torch
from pirga_indexed_store import sha,epoch_order,coverage
from pirga_training_cache import NativeTrainingCache
from pirga_fulltrain_dataset import FullTrainDataset,load_supervision
from pirga_candidate_state import CandidateStates
from pirga_innovation import ARMS
from pirga_grouped_innovation import GroupedInnovations,grouped_innovation_losses


def atomic(path,value):
    q=path.with_suffix('.tmp');q.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');q.replace(path)


def execute(protocol_path,protocol_sha,out):
    if sha(protocol_path)!=protocol_sha:raise ValueError('Unpinned innovation protocol')
    p=json.loads(protocol_path.read_text());repo=Path(__file__).resolve().parents[1]
    if p['status']!='PREREGISTERED_PHASE2_BEFORE_OPTIMIZATION' or p['epochs']!=[1,3,5] or p['GPU_wall_hours']!=0:raise ValueError('Unregistered schedule/device')
    for name,digest in p['source_code_sha256'].items():
        if sha(repo/name)!=digest:raise ValueError('Training source changed '+name)
    for field in ('admission','horizons','shadow'):
        if sha(p[field]['path'])!=p[field]['sha256']:raise ValueError('Source changed '+field)
    if json.loads(Path(p['shadow']['path']).read_text())['status']!='PASS_REAL_TRAIN_INDEPENDENT_INNOVATION_SHADOW':raise ValueError('Shadow not passed')
    out.mkdir(parents=True,exist_ok=False);start=time.monotonic();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    stop=[False]
    for s in (signal.SIGINT,signal.SIGTERM):signal.signal(s,lambda *_:stop.__setitem__(0,True))
    status=dict(status='PREPARING_CPU',pid=os.getpid(),protocol_sha256=protocol_sha,GPU_wall_hours=0,DEV_read=False,scientific_success=False,
        source_commit_at_start=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip())
    atomic(out/'STATUS.json',status)
    ad=json.loads(Path(p['admission']['path']).read_text());ds=FullTrainDataset(ad['stores']);labels=load_supervision(ad['supervision']['path'],ad['supervision']['sha256'],ds)
    cache=NativeTrainingCache(p['cache']['path'],p['cache']['sha256']);states=CandidateStates(cache,p['horizon_boundaries'],30000)
    domain=p['common_training_domain']
    if sha(domain['path'])!=domain['sha256']:raise ValueError('Common domain changed')
    keys=[tuple(k) for k in json.loads(Path(domain['path']).read_text())]
    if len(keys)!=113415 or len(set(keys))!=len(keys):raise ValueError('Full eligible domain changed')
    by={}
    for k in keys:by.setdefault(k[0],[]).append(k)
    model=GroupedInnovations(p['seed']);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    checkpoints=[];epochs=[];step=0;zero=[0]*len(ARMS);nonzero=[0]*len(ARMS)
    def save_resume(epoch,visit):
        temp=out/'RESUME.tmp';torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,visit=visit,step=step,
            protocol_sha256=protocol_sha,checkpoints=checkpoints,completed_epochs=epochs),temp);temp.replace(out/'RESUME.pth')
    try:
        with (out/'TRAINING_CURVES.csv').open('x') as f:
            fields=['arm','epoch','step','epoch_visit','rank_loss','prediction_loss','total_loss','gradient_norm_before_clip'];writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader()
            for epoch in range(1,6):
                order=list(epoch_order(by,p['seed'],epoch));digest=hashlib.sha256(json.dumps(order,separators=(',',':')).encode()).hexdigest()
                if digest!=p['epoch_order_sha256'][str(epoch)]:raise ValueError('Shared order changed')
                visited=[];loss_sums=[0.]*len(ARMS);prediction_covered=[0]*len(ARMS);top1=[0]*len(ARMS);difficulty=Counter();groups=set();total_candidates=0;epoch_start=time.monotonic()
                for visit,key in enumerate(order,1):
                    st,q,native,ci,row=states.query(key);label=labels[key]
                    scores,mu,lv=model(st,q,native);values,counts=grouped_innovation_losses(scores,mu,lv,st,q,label)
                    loss,rank,pred=values
                    if not torch.isfinite(loss).all():raise ValueError('Nonfinite loss')
                    optimizer.zero_grad(set_to_none=True);loss.sum().backward();grad=model.clip_per_arm()
                    if not torch.isfinite(grad).all():raise ValueError('Nonfinite gradient')
                    optimizer.step();step+=1;visited.append(key);total_candidates+=len(native);difficulty[label['actual_primary_state']]+=1;groups.add((key[0],label['GT_person']))
                    vals=torch.stack([loss.detach(),rank.detach(),pred.detach(),grad.detach()]).tolist()
                    chosen=scores.detach().argmax(1).tolist()
                    for i in range(len(ARMS)):
                        loss_sums[i]+=vals[0][i];prediction_covered[i]+=counts[i]>0;top1[i]+=chosen[i] in label['certified_target_indices'];zero[i]+=vals[3][i]==0;nonzero[i]+=vals[3][i]>0
                    if visit==1 or visit%1000==0 or visit==len(order):
                        for i,arm in enumerate(ARMS):writer.writerow(dict(arm=arm,epoch=epoch,step=step,epoch_visit=visit,total_loss=vals[0][i],rank_loss=vals[1][i],prediction_loss=vals[2][i],gradient_norm_before_clip=vals[3][i]))
                        f.flush();status.update(status='TRAINING_CPU',epoch=epoch,epoch_visit=visit,step=step,CPU_wall_seconds=time.monotonic()-start,last_loss=vals[0],peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024);atomic(out/'STATUS.json',status);print(json.dumps(status),flush=True)
                        if shutil.disk_usage(out).free<30*2**30:raise RuntimeError('Disk reserve breached')
                    if step%10000==0:save_resume(epoch,visit)
                    if stop[0] or time.monotonic()-start>p['CPU_wall_hours_cap']*3600:
                        save_resume(epoch,visit);status.update(status='RESOURCE_STOP_WITH_EXACT_CHECKPOINT_NOT_COMPLETE',step=step);atomic(out/'STATUS.json',status);return
                cov=coverage(keys,visited)
                if not cov['full_epoch_pass']:raise ValueError('Actual full coverage failed')
                epochs.append(dict(epoch=epoch,coverage=cov,order_sha256=digest,person_scene_groups=len(groups),difficulty_counts=dict(difficulty),
                    average_native_candidate_count=total_candidates/len(order),per_arm_prediction_query_coverage=prediction_covered,
                    per_arm_mean_total_loss=[v/len(order) for v in loss_sums],train_preupdate_top1=[v/len(order) for v in top1],
                    train_metrics_qualification='Online-in-optimization mixed checkpoints, never heldout estimates',epoch_wall_seconds=time.monotonic()-epoch_start))
                atomic(out/'TRAINING_EPOCH_COVERAGE.json',dict(status='IN_PROGRESS',epochs=epochs))
                if epoch in (1,3,5):
                    for i,arm in enumerate(ARMS):
                        path=out/f'{arm}_epoch{epoch}.pth';state=model.arm_state(i)
                        torch.save(dict(state_dict=state,arm=arm,seed=p['seed'],step=step,epoch=epoch,protocol_sha256=protocol_sha,
                            admission_sha256=p['admission']['sha256'],horizon_boundaries=p['horizon_boundaries']),path)
                        checkpoints.append(dict(arm=arm,epoch=epoch,step=step,path=str(path),sha256=sha(path)))
                    atomic(out/'CHECKPOINTS.json',dict(checkpoints=checkpoints))
                save_resume(epoch+1,0)
        result=dict(status='PHASE2_FIXED5EPOCHS_TRAINED_NOT_SCIENTIFIC_SUCCESS',arms=list(ARMS),checkpoints=checkpoints,epochs=epochs,
            zero_gradient_updates=zero,nonzero_gradient_updates=nonzero,per_arm_optimizer_steps=step,protocol_sha256=protocol_sha,
            GPU_wall_hours=0,CPU_wall_seconds=time.monotonic()-start,DEV_read=False,scientific_success=False)
        atomic(out/'RESULTS.json',result);status.update(status='COMPLETE_FIXED5EPOCHS',step=step);atomic(out/'STATUS.json',status)
    except BaseException as e:
        status.update(status='FAILED_RETAIN_EVIDENCE',step=step,reason=repr(e));atomic(out/'STATUS.json',status);raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--protocol-sha',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();execute(a.protocol,a.protocol_sha,a.out)
