"""TRAIN-only CPU correctness/throughput probe; all probe weights discarded."""
import json,time,statistics,os,resource
from pathlib import Path
import torch
from pirga_indexed_store import sha
from pirga_training_cache import NativeTrainingCache
from pirga_fulltrain_dataset import FullTrainDataset,load_supervision
from pirga_candidate_state import CandidateStates
from pirga_innovation import InnovationHead,ARMS,innovation_losses
from pirga_grouped_innovation import GroupedInnovations,grouped_innovation_losses


def run(protocol_path,output):
    if output.exists():raise FileExistsError(output)
    p=json.loads(protocol_path.read_text());torch.set_num_threads(1);start=time.monotonic()
    cache=NativeTrainingCache(p['cache']['path'],p['cache']['sha256']);states=CandidateStates(cache,p['horizon_boundaries'],30000)
    a=json.loads(Path(p['admission']['path']).read_text());ds=FullTrainDataset(a['stores']);labels=load_supervision(a['supervision']['path'],a['supervision']['sha256'],ds)
    keys=[tuple(k) for k in json.loads(Path(p['common_training_domain']['path']).read_text())]
    selected=[]
    for scene in p['fit_scenes']:
        local=[k for k in keys if k[0]==scene];selected.append(local[len(local)//2]);selected.append(max(local,key=lambda k:len(cache.calls[cache.locations[k][0]]['histories'])))
    group=GroupedInnovations(p['seed']);independent=[InnovationHead(arm) for arm in ARMS]
    for i,m in enumerate(independent):m.load_state_dict(group.arm_state(i))
    optimizer=torch.optim.AdamW(group.parameters(),lr=.001,weight_decay=.0001)
    individual=[torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001) for m in independent]
    errors=dict(score=0.,loss=0.,parameter=0.,C1_baseline=0.);timings=[];keys_done=[]
    for key in selected:
        st,q,native,ci,row=states.query(key);label=labels[key]
        from cv_pil_v2_model import fused_scores
        original=cache.batch(key);errors['C1_baseline']=max(errors['C1_baseline'],float(abs(fused_scores(original,original.histories)-(native+2*(st.mean@q))).max()))
        begin=time.monotonic()
        scores,mu,lv=group(st,q,native);values,counts=grouped_innovation_losses(scores,mu,lv,st,q,label)
        optimizer.zero_grad(set_to_none=True);values[0].sum().backward();group.clip_per_arm();optimizer.step();timings.append(time.monotonic()-begin)
        for i,(m,opt,arm) in enumerate(zip(independent,individual,ARMS)):
            sc,mm,ll=m(st,q,native);loss,_,_,_=innovation_losses(sc,mm,ll,st,q,label,arm)
            errors['score']=max(errors['score'],float(abs(sc-scores[i].detach()).max()))
            errors['loss']=max(errors['loss'],float(abs(loss-values[0][i].detach())))
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
            errors['parameter']=max(errors['parameter'],max(float(abs(v-group.arm_state(i)[name]).max()) for name,v in m.state_dict().items()))
        keys_done.append(list(key))
    result=dict(status='PASS_REAL_TRAIN_INDEPENDENT_INNOVATION_SHADOW' if max(errors.values())<.0005 else 'FAILED_PARITY',
        tolerance=.0005,max_absolute_errors=errors,steps_per_arm=len(selected),parameters_per_arm=sum(v.numel() for v in independent[0].parameters()),
        query_batch_size=1,arms=list(ARMS),selected_real_keys=keys_done,selection='per TRAIN scene midpoint common query and maximum native-history count; no GT error selection or DEV',
        CPU_joint_step_seconds_median=statistics.median(timings),CPU_joint_step_seconds_max=max(timings),batch_assembly_outside_step_timer=True,
        source_sha256=sha(__file__),CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,all_weights_discarded=True,scientific_training_updates=0,GPU_wall_hours=0,wall_seconds=time.monotonic()-start,protocol_path=str(protocol_path),protocol_sha256=sha(protocol_path))
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    if result['status'].startswith('FAILED'):raise ValueError('Grouped independent shadow failed')

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.protocol,a.out)
