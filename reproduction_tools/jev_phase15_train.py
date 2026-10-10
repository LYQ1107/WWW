"""Bounded equal-budget training; reserved TRAIN blocks never receive gradients."""
import argparse
import collections
import math
import random
import time
import traceback
import numpy as np
import torch
from jev_phase15_common import *
from jev_phase14_artifacts import load_dense
from jev_phase13_learning import collate, atomic_torch, profile
from jev_phase15_losses import objective
from gtr.modeling.jev_phase15.model import PersistentIdentityPolicy, ARMS
from gtr.modeling.jev_stage2.assignment import lawful_choice


def load_records(version=2):
    records=[]; manifests=[]
    for video in TRAIN:
        folder=OUT/f'commitment_dataset_v{version}'/f'video{video:02d}'
        result=read(folder/'RESULT.json'); assert result['status']=='PASS'
        descriptor=result['DATASET']; assert sha(descriptor['path'])==descriptor['SHA256']
        data=load_dense(descriptor['path'])
        for record in data['records']:
            assert record['key'][0] in TRAIN
            assert record['audit_block']==((record['key'][1]//64)%5==4)
        records.extend(data['records']); manifests.append(ref(folder/'RESULT.json'))
    return [r for r in records if not r['audit_block']], [r for r in records if r['audit_block']], manifests


def batch(records, device='cuda:0'):
    x,y=collate(records,device,'full');b,q=x['question_mask'].shape;k=x['identity_mask'].shape[1]
    x['commitment_features']=torch.zeros(b,q,k,20,device=device)
    y.update(availability=torch.full((b,q),-1,device=device), trust=torch.full((b,q,k),-1,device=device),
             safety=torch.full((b,q,k),-1,device=device),uncertainty=torch.full((b,q,k),-1,device=device),
             commit_positive=torch.zeros(b,q,k+1,dtype=torch.bool,device=device),
             commit_kind=torch.zeros(b,q,dtype=torch.int8,device=device))
    for index,r in enumerate(records):
        qq=len(r['rows']);kk=len(r['refs'])
        x['commitment_features'][index,:qq,:kk]=r['inputs']['commitment_features'][0].to(device)
        y['availability'][index,:qq]=r['availability'].to(device)
        y['trust'][index,:qq,:kk]=r['trust'].to(device)[None].expand(qq,-1)
        for key in ['safety','uncertainty']:y[key][index,:qq,:kk]=r[key].to(device)
        y['commit_positive'][index,:qq,:kk]=r['commit_positive'][:,:kk].to(device)
        y['commit_positive'][index,:qq,k]=r['commit_positive'][:,kk].to(device)
        y['commit_kind'][index,:qq]=r['commit_kind'].to(device)
    return x,y


def initialize(arm, seed=20261009):
    policy=PersistentIdentityPolicy(arm).cuda()
    variant='fixed_question' if arm=='D_fixed' else 'set_transformer' if arm=='E_set' else 'multi_question'
    old,checkpoint,manifest=load_old_policy(variant,seed)
    policy.identity.load_state_dict(old.state_dict(),strict=True)
    if hasattr(policy,'option'):
        torch.nn.init.zeros_(policy.option.value[-1].weight);torch.nn.init.zeros_(policy.option.value[-1].bias)
    if hasattr(policy,'reliability'):
        torch.nn.init.zeros_(policy.reliability.head[-1].weight);torch.nn.init.zeros_(policy.reliability.head[-1].bias)
    return policy,checkpoint,manifest


@torch.no_grad()
def assess(model,records,limit=256):
    model.eval();count=collections.Counter();risk=[]
    chosen=np.linspace(0,len(records)-1,min(limit,len(records)),dtype=int).tolist() if records else []
    selected=set(chosen)
    # Deterministic correction strata, not outcome-selected favourable cases.
    correction=[i for i,r in enumerate(records) if (r['commit_kind']==2).any()]
    for i in np.linspace(0,len(correction)-1,min(limit,len(correction)),dtype=int).tolist() if correction else []:
        selected.add(correction[i])
    items=[records[i] for i in sorted(selected)]
    for begin in range(0,len(items),4):
        part=items[begin:begin+4];x,y=batch(part);details=model.details(x)
        for index,r in enumerate(part):
            q=len(r['rows']);k=len(r['refs']);z=torch.cat([details['logits'][index,:q,:k],details['logits'][index,:q,-1:]],-1)
            choice=lawful_choice(z,x['legal'][index,:q,:k]);confidence=z.softmax(-1).max(-1).values.cpu().tolist()
            for row,col in enumerate(choice):
                col=k if col<0 else col;count['rows']+=1
                if r['positive'][row,:k].any():
                    count['normal_supported']+=1;count['normal_retained']+=int(col<k and r['positive'][row,col])
                    count['false_defer']+=int(col==k)
                kind=int(r['commit_kind'][row])
                if kind:
                    count['commit_certified']+=1;count['commit_correct']+=int(r['commit_positive'][row,col])
                    count[f'kind{kind}_support']+=1;count[f'kind{kind}_correct']+=int(r['commit_positive'][row,col])
                if r['supervised'][row]:
                    known=bool(r['known_options'][row,col]);correct=bool(r['positive'][row,col])
                    count['certified_row_queries']+=1;count['selected_UNKNOWN']+=int(not known)
                    count['selected_certified_wrong']+=int(known and not correct)
                    risk.append((confidence[row],known,correct))
            if 'purity_logits' in details and q:
                prediction=(details['purity_logits'][index,0,:k]>=0).cpu()
                for label in [0,1]:
                    mask=r['trust']==label;count[f'purity{label}_support']+=int(mask.sum())
                    count[f'purity{label}_correct']+=int((prediction[mask]==label).sum())
    coverage=[]
    for threshold in [.5,.7,.9]:
        selected=[item for item in risk if item[0]>=threshold];known=[item for item in selected if item[1]]
        coverage.append(dict(confidence_threshold=threshold,query_coverage=len(selected)/max(1,len(risk)),
            selected_UNKNOWN=sum(not item[1] for item in selected),
            certified_selection_risk=sum(not item[2] for item in known)/len(known) if known else None,
            certified_selection_support=len(known)))
    rates=lambda a,b:count[a]/count[b] if count[b] else None
    return dict(counts=dict(count),commitment_accuracy=rates('commit_correct','commit_certified'),
        safe_continuation_accuracy=rates('kind1_correct','kind1_support'),
        necessary_correction_accuracy=rates('kind2_correct','kind2_support'),
        normal_retention=rates('normal_retained','normal_supported'),selective_risk_coverage=coverage,
        scope='reserved TRAIN temporal blocks, unchanged actual predicted-state payloads; not full-video tracking metrics')


def main(arm, phase, seed=20261009, version=1):
    protect();storage_guard();assert arm in ARMS and phase in ['tiny','pilot','formal'] and seed in SEEDS
    protocol=read(REPORTS/'PREREGISTRATION.json')['training'];assert version<=protocol['max_evidence_driven_versions']
    if phase=='formal':
        gate=read(REPORTS/'PILOT_RESULTS.json');assert gate['formal_qualified'] is True
    if arm!='F_full':
        assert read(REPORTS/'PILOT_RESULTS.json')['full_pilot_qualified'] is True
    if phase!='tiny':assert read(REPORTS/'COMMITMENT_FEASIBILITY_GO_NO_GO.json')['mechanism_GO'] is True
    steps=protocol['Tiny_updates' if phase=='tiny' else 'Pilot_updates' if phase=='pilot' else 'formal_updates']
    out=OUT/f'training_v{version}'/arm/f'seed{seed}'/phase;out.mkdir(parents=True,exist_ok=True)
    assert not (out/'RESULT.json').exists()
    torch.set_num_threads(1);torch.manual_seed(seed);np.random.seed(seed);rng=random.Random(seed)
    train,audit,manifests=load_records();assert train and audit
    grouped=collections.defaultdict(list)
    for r in train:grouped[r['key'][0],r['key'][1]//64].append(r)
    safe=[r for r in train if (r['commit_kind']==1).any()];correct=[r for r in train if (r['commit_kind']==2).any()]
    assert safe,'no certified safe continuation support'
    keys=sorted(grouped)
    model,checkpoint,initializer_manifest=initialize(arm,seed)
    if phase=='tiny':
        tiny=([correct[0]] if correct else [])+[safe[0]]+[grouped[keys[0]][0],grouped[keys[-1]][-1]]
        tiny=tiny[:4]
    optimizer=torch.optim.AdamW(model.parameters(),lr=protocol['learning_rate'],weight_decay=protocol['weight_decay'])
    source=binding(seed=seed,checkpoints=[checkpoint],dataset=manifests,
        evaluator='UNKNOWN-safe supervised objective and reserved TRAIN constrained assignment',scope=f'{phase}, version{version}, no DEVELOPMENT gradients')
    source['initializer_manifest']=initializer_manifest
    probe=batch([train[0]])[0];model.eval()
    with torch.no_grad():
        initial=model(probe);old=model.identity({k:v for k,v in probe.items() if k!='commitment_features'})
        assert torch.allclose(initial,old,atol=2e-6),'new initial heads changed frozen WHO choices'
    save(out/'START.json',dict(binding=source,arm=arm,phase=phase,version=version,updates=steps,
        initializer='matching Phase XIV 20k; zero new value/risk final layers retain original initial scores',
        initial_scores_parity=True,development_GT=False,training_payloads=len(train),reserved_payloads=len(audit),
        train_safe_payloads=len(safe),train_correction_payloads=len(correct),executed_profile=profile(model,probe)))
    before=assess(model,audit,128);begin=time.monotonic();losses=[];history=[];gradients=collections.defaultdict(float)
    first_loss=None;step=0
    try:
        for step in range(1,steps+1):
            part=tiny if phase=='tiny' else [rng.choice(grouped[rng.choice(keys)]) for _ in range(2)]+[
                rng.choice(safe),rng.choice(correct) if correct else rng.choice(grouped[rng.choice(keys)])]
            assert all(not r['audit_block'] and r['key'][0] in TRAIN for r in part)
            x,y=batch(part);model.train();optimizer.zero_grad(set_to_none=True)
            details=model.details(x);loss,parts=objective(details,x,y,arm);assert torch.isfinite(loss)
            if first_loss is None:first_loss=float(loss.detach())
            loss.backward()
            for name,param in model.named_parameters():
                if param.grad is not None:
                    group=name.split('.')[0];gradients[group]=max(gradients[group],float(param.grad.detach().abs().max()))
            norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.));assert math.isfinite(norm)
            optimizer.step();losses.append(float(loss.detach()))
            if step%250==0 or step==steps:
                measurement=assess(model,audit,128);history.append(dict(update=step,assessment=measurement,loss=parts))
                save(out/'PROGRESS.json',dict(status='RUNNING',update=step,total=steps,seconds=time.monotonic()-begin,
                    measurement=measurement,last_loss=float(loss.detach())))
                print('PHASE15_UPDATE',arm,phase,version,step,measurement['commitment_accuracy'],
                      measurement['necessary_correction_accuracy'],flush=True)
        final=assess(model,audit,256);threshold=protocol['pilot_gate']
        support=final['counts'].get('kind1_support',0)>0 and final['counts'].get('kind2_support',0)>0
        passed=support and final['commitment_accuracy']>=threshold['certified_commitment_accuracy_min'] and (
            final['safe_continuation_accuracy']>=threshold['safe_continuation_accuracy_min']) and (
            final['necessary_correction_accuracy']>=threshold['necessary_correction_accuracy_min'])
        artifact=out/'LAST_FROZEN.pth'
        atomic_torch(artifact,dict(model=model.state_dict(),arm=arm,seed=seed,version=version,actual_updates=step,binding=source))
        result=dict(status='COMPLETE',binding=source,arm=arm,phase=phase,version=version,seed=seed,
            actual_updates=step,checkpoint=ref(artifact),before=before,final=final,history=history,
            loss_before=first_loss,loss_last=float(np.mean(losses[-min(16,len(losses)):])),gradient_health=dict(gradients),
            label_assessment_qualified=bool(passed),native_pilot_safety='PENDING_REAL_MUTATED_BRANCHES',
            formal_qualified=False,seconds=time.monotonic()-begin,manifests=manifests)
        save(out/'RESULT.json',result);save(out/'PROGRESS.json',dict(status='COMPLETE',actual_updates=step,qualified=bool(passed)))
        print('PHASE15_TRAIN_COMPLETE',arm,phase,version,passed,final,flush=True)
    except Exception:
        save(out/'FAILED.json',dict(status='FAIL',binding=source,actual_update=step,traceback=traceback.format_exc()));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=ARMS,default='F_full');p.add_argument('--phase',choices=['tiny','pilot','formal'],required=True)
    p.add_argument('--seed',type=int,default=20261009);p.add_argument('--version',type=int,default=1);a=p.parse_args()
    main(a.arm,a.phase,a.seed,a.version)
