"""Real TRAIN shadow comparison of independent versus grouped optimizer states."""
import argparse,json,time
from pathlib import Path
import torch
from pirga_training_cache import NativeTrainingCache
from pirga_fulltrain_dataset import FullTrainDataset,load_supervision
from pirga_grouped_heads import GroupedHeads,grouped_losses,NAMES
from pirga_device_losses import supervised_losses
from cv_pil_v2_model import CandidatePredictor
from cv_pil_native_handoff import safe_torch_load
from pirga_indexed_store import sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--cache-sha',required=True);p.add_argument('--admission',type=Path,required=True);p.add_argument('--source-probe',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;started=time.monotonic()
    ad=json.loads(a.admission.read_text());ds=FullTrainDataset(ad['stores']);labels=load_supervision(ad['supervision']['path'],ad['supervision']['sha256'],ds)
    source=json.loads(a.source_probe.read_text());measure={tuple(v['key']):v for v in source['measurements']};keys=[]
    for scene in sorted({k[0] for k in measure}):keys.append(max((k for k in measure if k[0]==scene),key=lambda k:measure[k]['histories']))
    keys+= [k for k in sorted(measure,key=lambda k:-measure[k]['histories']) if k not in keys][:20-len(keys)]
    cache=NativeTrainingCache(a.cache,a.cache_sha,device='cuda');states=[]
    cps=['134e6ff366e14f51c0c6648a4d4996d87d6e6dce3f4d1840d588d4719aa77c47','76e66718cde4ebd73d55b06fbecd3bff21c0205335e6dc4c45ef650ac920208c','13b27d180135538834092ff67dde3877ce69e2b6796f834782a8ebab2474574c']
    for arm,digest in zip(['C2','C3','C4'],cps):
        cp=Path('/data1/liuyeqiang/WWW_cvpil_v3_runtime_20261015/p3_matched_step300_seed20261015_v1')/(arm+'_step300.pth')
        if sha(cp)!=digest:raise ValueError('Frozen checkpoint changed')
        states.append(safe_torch_load(cp)['state_dict'])
    independent=[]
    for i,state in enumerate(states):
        m=CandidatePredictor(use_camera=i!=2).cuda();m.load_state_dict(state);independent.append(m)
    group=GroupedHeads(states).cuda();opts=[torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001) for m in independent];gopt=torch.optim.AdamW(group.parameters(),lr=.001,weight_decay=.0001)
    maximum_score=maximum_loss=maximum_parameter=0.;logs=[]
    for step,key in enumerate(keys,1):
        b=cache.batch(key);label=labels[key];separate_losses=[]
        with torch.no_grad():gs,_=group(b)
        for i,(m,opt) in enumerate(zip(independent,opts)):
            s,h=m(b);maximum_score=max(maximum_score,float((s-gs[i]).abs().max()));r,pred,_=supervised_losses(s,h,b,label);loss=r+(0 if i==0 else .5)*pred
            separate_losses.append(loss.detach());opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
        s,h=group(b);loss,r,pred,count=grouped_losses(s,h,b,label);maximum_loss=max(maximum_loss,float((loss-torch.stack(separate_losses)).abs().max()))
        gopt.zero_grad(set_to_none=True);loss.sum().backward();group.clip_per_arm();gopt.step()
        for i,m in enumerate(independent):
            for name,value in group.arm_state(i).items():maximum_parameter=max(maximum_parameter,float((value-m.state_dict()[name].detach().cpu()).abs().max()))
        if maximum_score>5e-4 or maximum_loss>5e-4 or maximum_parameter>5e-4:raise ValueError('Grouped math exceeds preregistered FP32 shadow tolerance '+str((maximum_score,maximum_loss,maximum_parameter)))
    # Independent gradients: changing C3 loss weight cannot affect C2 slice.
    b=cache.batch(keys[-1]);s,h=group(b);loss,_,_,_=grouped_losses(s,h,b,labels[keys[-1]]);gopt.zero_grad(set_to_none=True);loss.sum().backward();before=[p.grad[0].clone() for p in group.parameters()]
    s,h=group(b);loss,_,_,_=grouped_losses(s,h,b,labels[keys[-1]]);gopt.zero_grad(set_to_none=True);(loss[0]+0*loss[1]+loss[2]).backward()
    if not all(torch.equal(old,p.grad[0]) for old,p in zip(before,group.parameters())):raise ValueError('Cross-arm gradient leakage')
    times=[]
    for key in keys:
        b=cache.batch(key);torch.cuda.synchronize();t=time.monotonic();s,h=group(b);loss,_,_,_=grouped_losses(s,h,b,labels[key]);gopt.zero_grad(set_to_none=True);loss.sum().backward();group.clip_per_arm();gopt.step();torch.cuda.synchronize();times.append(dict(key=list(key),histories=len(b.histories),seconds=time.monotonic()-t))
    result=dict(status='PASS_REAL_TRAIN_GROUPED_INDEPENDENT_HEAD_SHADOW',shadow_comparison_steps_per_arm=len(keys),shadow_timing_steps_per_arm=len(keys),total_discarded_grouped_steps_per_arm=2*len(keys),discarded_shadow_weights=True,formal_research_training=False,DEV_read=False,
        max_score_absolute_difference=maximum_score,max_loss_absolute_difference=maximum_loss,max_parameter_absolute_difference=maximum_parameter,FP32_preregistered_shadow_tolerance=.0005,bitwise_equivalence_claim=False,
        C2_gradients_unchanged_when_C3_loss_removed=True,per_arm_gradient_clipping=True,per_arm_query_batch_size=1,models_independent=True,
        grouped_optimizer_step_seconds=times,wall_seconds=time.monotonic()-started,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated())
    if a.out.exists():raise FileExistsError(a.out)
    a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='grouped_optimizer_step_seconds'},indent=2),flush=True)

if __name__=='__main__':main()
