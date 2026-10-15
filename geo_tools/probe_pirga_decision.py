"""Discarded real TRAIN shadow of two independent ordinary decision heads."""
import argparse,json,statistics,time,os
from pathlib import Path
import torch
from pirga_indexed_store import sha
from pirga_training_cache import NativeTrainingCache
from pirga_candidate_state import CandidateStates
from pirga_fulltrain_dataset import FullTrainDataset,load_supervision
from pirga_decision_head import ARMS,DecisionMLP,GroupedDecisions,decision_input,decision_losses
from cv_pil_native_handoff import safe_torch_load


def run(protocol_path,out):
    if out.exists():raise FileExistsError(out)
    start=time.monotonic();torch.set_num_threads(1);p=json.loads(protocol_path.read_text())
    cache=NativeTrainingCache(p['cache']['path'],p['cache']['sha256']);states=CandidateStates(cache,p['horizon_boundaries'],30000)
    ad=json.loads(Path(p['admission']['path']).read_text());ds=FullTrainDataset(ad['stores']);labels=load_supervision(ad['supervision']['path'],ad['supervision']['sha256'],ds)
    if sha(p['context']['path'])!=p['context']['sha256']:raise ValueError('Native context changed')
    cm=json.loads(Path(p['context']['path']).read_text())
    if sha(cm['data_path'])!=cm['data_sha256']:raise ValueError('Full native context changed')
    contexts=safe_torch_load(cm['data_path']);keys=[tuple(k) for k in json.loads(Path(p['common_training_domain']['path']).read_text())];selected=[]
    for scene in p['fit_scenes']:
        local=[k for k in keys if k[0]==scene];selected.extend([local[len(local)//2],max(local,key=lambda k:len(cache.calls[cache.locations[k][0]]['histories']))])
    group=GroupedDecisions(p['seed']);heads=[DecisionMLP(),DecisionMLP()]
    for i,m in enumerate(heads):m.load_state_dict(group.arm_state(i))
    opt=torch.optim.AdamW(group.parameters(),lr=.001,weight_decay=.0001);opts=[torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001) for m in heads]
    errors=dict(score=0.,loss=0.,parameter=0.);timings=[]
    for key in selected:
        st,q,native,ci,row=states.query(key);ctx=contexts[ci];label=labels[key];threshold=ctx['thresholds'];opp=ctx['opportunity'][row];accepted=int(ctx['original_accepted_columns'][row]);x=decision_input(st,q,native,threshold,opp,accepted)
        begin=time.monotonic();sc,de=group(x,st,q,native);loss=torch.stack([decision_losses(sc[i],de[i],label,threshold,opp,accepted,bool(i))[0] for i in range(2)])
        opt.zero_grad(set_to_none=True);loss.sum().backward();group.clip_per_arm();opt.step();timings.append(time.monotonic()-begin)
        for i,m in enumerate(heads):
            s,d=m(x,st,q,native);l=decision_losses(s,d,label,threshold,opp,accepted,bool(i))[0];errors['score']=max(errors['score'],float(abs(s-sc[i].detach()).max()));errors['loss']=max(errors['loss'],float(abs(l-loss[i].detach())))
            opts[i].zero_grad(set_to_none=True);l.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opts[i].step()
            errors['parameter']=max(errors['parameter'],max(float(abs(v-group.arm_state(i)[name]).max()) for name,v in m.state_dict().items()))
    result=dict(status='PASS_REAL_TRAIN_ORDINARY_DECISION_SHADOW' if max(errors.values())<.0005 else 'FAILED_PARITY',max_absolute_errors=errors,tolerance=.0005,
        arms=list(ARMS),parameters_per_arm=sum(v.numel() for v in heads[0].parameters()),query_batch_size=1,steps_per_arm=len(selected),selected_real_keys=[list(k) for k in selected],
        CPU_joint_step_seconds_median=statistics.median(timings),CPU_joint_step_seconds_max=max(timings),batch_assembly_outside_timer=True,
        all_weights_discarded=True,scientific_training_updates=0,DEV_read=False,GPU_wall_hours=0,CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),
        wall_seconds=time.monotonic()-start,protocol_sha256=sha(protocol_path),source_sha256=sha(__file__))
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    if result['status']=='FAILED_PARITY':raise ValueError('Independent optimizer shadow failed')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.protocol,a.out)
