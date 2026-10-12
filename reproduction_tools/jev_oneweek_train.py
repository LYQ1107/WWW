"""Bounded actual equal-target learning; no DEV gradient or future-value head."""
import argparse
import collections
import random
import traceback
import torch
from jev_oneweek_common import *
from jev_oneweek_learning import load_records,batch,initialize,loss,comparisons,assess
from gtr.modeling.jev_oneweek.model import ARMS


def tiny_records(train,model):
    # One shared immutable Tiny fixture, selected only from authorized TRAIN.
    path=REPORTS/'TINY_FIXTURE.json'
    by_key={tuple(r['key']):r for r in train}
    if path.exists():return [by_key[tuple(k)] for k in read(path)['keys']]
    chosen=[]
    for video in [12,13,14]:
        candidates=[r for r in train if r['key'][0]==video and r['key'][1]>=8
                    and (r['oneweek_labels']['cross']>=0).any()
                    and (r['oneweek_labels']['stability']>=0).any()]
        for r in candidates:
            x,y=batch([r])
            with torch.no_grad():details=model.details(x)
            if comparisons(details,y,[r]):chosen.append(r);break
    assert len(chosen)>=2,'No certified changed-row plan ranking fixture across two videos'
    if len(chosen)<4:chosen.append(chosen[-1])
    save(path,dict(status='FROZEN_TRAIN_ONLY_TINY_DIAGNOSTIC',keys=[r['key'] for r in chosen],
                   source='First chronological pair-rank-certifiable natural payload per TRAIN12/13/14; explicit repeated Tiny overfit fixture, not natural full sampling',
                   inference_GT_selection=False,training_only=True))
    return chosen


def main(arm,phase,seed):
    protect();assert arm in ARMS and seed in SEEDS
    contract=read(REPORTS/'MODEL_AND_NATIVE_CONTRACT_TESTS.json');assert contract['status']=='PASS'
    if phase!='tiny':
        assert read(REPORTS/'TRAIN_jev_tiny_SEED20261009.json')['engineering_training_pass']
    budget=read(REPORTS/'PREREGISTRATION.json')['training']
    requested={'tiny':budget['tiny_updates'],'pilot':budget['pilot_updates'],'formal':budget['formal_total_updates']}[phase]
    folder=OUT/'training_v1'/arm/f'seed{seed}'/phase
    folder.mkdir(parents=True,exist_ok=False)
    train,validation,sources=load_records();model,initializer=initialize(arm,seed);rng=random.Random(seed)
    source=binding('Equal-supervision current association training, not future safe action learning')
    groups=collections.defaultdict(list)
    for r in train:
        assert r['key'][0] in [12,13,14] and not r['audit_block'] and r['oneweek_labels']['train_gradient']
        groups[r['key'][0],r['key'][1]//64].append(r)
    blocks={v:[k for k in groups if k[0]==v] for v in [12,13,14]}
    weights={}
    for name in ['cross','stability']:
        negative=sum(int((r['oneweek_labels'][name]==0).sum()) for r in train)
        positive=sum(int((r['oneweek_labels'][name]==1).sum()) for r in train)
        assert positive and negative
        weights[name]=negative/positive
    positive=sum(int(r['positive'][:,:-1].sum()) for r in train)
    negative=sum(int((r['known_options'][:,:-1]&~r['positive'][:,:-1]).sum()) for r in train)
    weights['compatibility']=negative/max(positive,1)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=budget['learning_rate'],weight_decay=budget['weight_decay'])
    start_step=0
    if phase=='formal':
        pilot=read(OUT/'training_v1'/arm/f'seed{seed}/pilot/RESULT.json')
        assert pilot['status']=='COMPLETE' and pilot['engineering_training_pass']
        checkpoint=pilot['checkpoint'];assert sha(checkpoint['path'])==checkpoint['SHA256']
        saved=torch.load(checkpoint['path'],map_location='cuda:0')
        model.load_state_dict(saved['new_modules'],strict=False);optimizer.load_state_dict(saved['optimizer'])
        start_step=saved['updates'];assert start_step==1000
    fixture=tiny_records(train,model) if phase=='tiny' else None
    before=assess(model,validation,128);begin=time.monotonic();history=[];gradients=collections.Counter();rank_batches=0
    initial_batch=fixture or [groups[sorted(groups)[0]][0]]
    x,y=batch(initial_batch);details=model.details(x)
    initial_loss,_,_=loss(details,x,y,initial_batch,weights)
    save(folder/'START.json',dict(binding=source,arm=arm,phase=phase,seed=seed,
         start_updates=start_step,total_target_updates=requested,initializer=initializer,
         training_records=len(train),controller_audit_records=len(validation),sources=sources,
         pos_weights_TRAIN_only=weights,new_trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
         gradients_videos=[12,13,14],old_reserved_temporal_blocks_excluded=True,DEV_gradients=False,
         synthetic_samples=False,initial_loss=float(initial_loss.detach())))
    for step in range(start_step+1,requested+1):
        if fixture is not None:part=fixture
        else:
            part=[]
            for _ in range(budget['batch_payloads']):
                video=rng.choice([12,13,14]);key=rng.choice(blocks[video]);part.append(rng.choice(groups[key]))
        x,y=batch(part);model.train();optimizer.zero_grad(set_to_none=True)
        details=model.details(x);value,components,pairs=loss(details,x,y,part,weights)
        assert torch.isfinite(value)
        value.backward();norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
        assert torch.isfinite(norm)
        for name,p in model.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all();gradients[name]=max(gradients[name],float(p.grad.abs().max()))
        optimizer.step();rank_batches+=bool(pairs)
        if step==start_step+1 or step%32==0 or step==requested:
            record=dict(step=step,total_steps=requested,loss=float(value.detach()),components=components,
                        certified_rank_pairs=len(pairs),grad_norm=float(norm),elapsed=time.monotonic()-begin)
            history.append(record);save(folder/'PROGRESS.json',dict(status='TRAINING',**record))
            print('ONEWEEK_TRAIN_STEP',arm,phase,seed,record,flush=True)
        if step%128==0:protect()
    model.eval()
    with torch.no_grad():
        x,y=batch(initial_batch);last_loss=loss(model.details(x),x,y,initial_batch,weights)[0]
    final=assess(model,validation,128)
    target=folder/'NEW_MODULES.pth'
    torch.save(dict(new_modules={k:v.cpu() for k,v in model.state_dict().items() if not k.startswith('identity.')},
                    initializer=initializer,updates=requested,optimizer=optimizer.state_dict(),arm=arm,seed=seed,
                    source_binding=source),target)
    result=dict(status='COMPLETE',binding=source,arm=arm,phase=phase,seed=seed,
                total_updates=requested,actual_new_updates=requested-start_step,
                checkpoint=ref(target),initializer=initializer,
                before=before,final=final,history=history,
                fixture_loss_before=float(initial_loss.detach()),fixture_loss_after=float(last_loss),
                max_abs_gradients=dict(gradients),rank_supervised_batches=rank_batches,
                engineering_training_pass=bool(gradients) and rank_batches>0,
                controller_audit_is_NOT_independent_generalization=True,
                science_association_gain='NOT_RUN_FULL_NATIVE',JEV_independent_value='NOT_ESTABLISHED',
                seconds=time.monotonic()-begin)
    save(folder/'RESULT.json',result)
    save(REPORTS/f'TRAIN_{arm}_{phase}_SEED{seed}.json',result)
    print('ONEWEEK_TRAIN_COMPLETE',arm,phase,seed,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=ARMS,required=True)
    parser.add_argument('--phase',choices=['tiny','pilot','formal'],required=True)
    parser.add_argument('--seed',type=int,default=20261009);a=parser.parse_args()
    try:main(a.arm,a.phase,a.seed)
    except Exception:
        failure('training',traceback.format_exc());raise
