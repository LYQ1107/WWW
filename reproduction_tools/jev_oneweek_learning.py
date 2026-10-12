"""Reuse dense causal inputs; all controls receive identical offline targets."""
import collections
import math
import torch
from torch.nn import functional as F
from jev_oneweek_common import *
from jev_phase14_artifacts import load_dense
from jev_phase15_train import batch as historical_batch
from gtr.modeling.jev_oneweek.model import JevTrack


def load_records():
    audit=read(REPORTS/'SUPERVISION_AUDIT.json')
    assert audit['status']=='COMPLETE'
    descriptor=audit['labels'];assert sha(descriptor['path'])==descriptor['SHA256']
    labels={tuple(r['key']):r for r in load_dense(descriptor['path'])['labels']}
    train=[];validation=[];source=[]
    for video in TRAIN:
        manifest=read(PHASE15/'commitment_dataset_v2'/f'video{video:02d}'/'RESULT.json')
        assert sha(manifest['DATASET']['path'])==manifest['DATASET']['SHA256']
        records=load_dense(manifest['DATASET']['path'])['records']
        for r in records:
            annotation=labels[tuple(r['key'])]
            r=dict(r,oneweek_labels=annotation)
            if annotation['train_gradient']:
                assert video in [12,13,14] and not r['audit_block'];train.append(r)
            elif video==16:validation.append(r)
        source.append(manifest['DATASET'])
    assert train and validation
    return train,validation,source


def batch(records):
    x,y=historical_batch(records)
    b,q=x['question_mask'].shape;k=x['identity_mask'].shape[1]
    labels={name:torch.full((b,q,k),-1,dtype=torch.int8,device='cuda:0') for name in ['cross','stability']}
    for index,r in enumerate(records):
        qq=len(r['rows']);kk=len(r['refs'])
        for name in labels:labels[name][index,:qq,:kk]=r['oneweek_labels'][name].to('cuda:0')
    y.update(labels)
    return x,y


def initialize(arm,seed):
    seed_all(seed)
    result=read(PHASE14/'formal_v2/multi_question/availability_joint'/f'seed{seed}'/'RESULT.json')
    checkpoint=result['checkpoint'];assert sha(checkpoint['path'])==checkpoint['SHA256']
    model=JevTrack(arm).cuda();model.identity.load_state_dict(torch.load(checkpoint['path'],map_location='cpu')['model'],strict=True)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad)<=500000
    return model,checkpoint


def comparisons(details,y,records):
    """Only fully certified changed rows label a current-plan comparison.

    Unchanged UNKNOWN rows cancel in the comparison; a changed UNKNOWN row
    disqualifies that pair. Continuity adds a current-ID preference only when
    both choices have a certified continuity target. No future utility labels.
    """
    pairs=[]
    for index,r in enumerate(records):
        k=len(r['refs']);alternatives=details['plans'][index]
        for a in range(len(alternatives)):
            for b in range(a+1,len(alternatives)):
                changed=[row for row,(u,v) in enumerate(zip(alternatives[a],alternatives[b])) if u!=v]
                if not changed:continue
                ua=[k if alternatives[a][row]<0 else alternatives[a][row] for row in changed]
                ub=[k if alternatives[b][row]<0 else alternatives[b][row] for row in changed]
                if any(not r['known_options'][row,u] or not r['known_options'][row,v] for row,u,v in zip(changed,ua,ub)):continue
                difference=sum(int(r['positive'][row,u])-int(r['positive'][row,v]) for row,u,v in zip(changed,ua,ub))
                stable=r['oneweek_labels']['stability']
                for row,u,v in zip(changed,ua,ub):
                    if u<k and v<k and stable[row,u]>=0 and stable[row,v]>=0:
                        difference+=.25*(int(stable[row,u])-int(stable[row,v]))
                if difference:pairs.append((index,a,b,1 if difference>0 else -1,len(changed)))
    return pairs


def loss(details,x,y,records,pos_weights):
    terms={};zero=details['features'].sum()*0
    for name,out in [('cross','cross_logits'),('stability','stability_logits')]:
        labels=y[name];valid=labels>=0
        disabled=(name=='cross' and details.get('arm')=='stability_only') or (name=='stability' and details.get('arm')=='cross_only')
        terms[name]=F.binary_cross_entropy_with_logits(details[out][valid],labels[valid].float(),
                       pos_weight=details[out].new_tensor(pos_weights[name])) if valid.any() and not disabled else zero
    pairs=comparisons(details,y,records)
    ranking=[F.softplus(-sign*(details['plan_scores'][index][a]-details['plan_scores'][index][b])/math.sqrt(changed))
             for index,a,b,sign,changed in pairs]
    terms['current_plan_ranking']=torch.stack(ranking).mean() if ranking else zero
    # Same current compatibility supervision for every architecture. The edge
    # component permits an honest same-score Hungarian 2x2; the joint component
    # is audited separately rather than pretending Hungarian optimizes it.
    margin=details['edge_logits'][:,:,:-1]-details['edge_logits'][:,:,-1:]
    known=y['known_options'][:,:,:-1]&x['legal']&x['question_mask'][:,:,None]
    terms['current_edge_compatibility']=F.binary_cross_entropy_with_logits(margin[known],y['positive'][:,:,:-1][known].float(),
        pos_weight=margin.new_tensor(pos_weights['compatibility'])) if known.any() else zero
    value=terms['current_plan_ranking']+.25*(terms['cross']+terms['stability'])+.1*terms['current_edge_compatibility']
    return value,{name:float(v.detach()) for name,v in terms.items()},pairs


@torch.no_grad()
def assess(model,records,limit=128):
    model.eval();counts=collections.Counter();question={name:collections.Counter() for name in ['cross','stability']}
    selected=[records[i] for i in torch.linspace(0,len(records)-1,min(limit,len(records))).long().tolist()]
    for start in range(0,len(selected),4):
        part=selected[start:start+4];x,y=batch(part);details=model.details(x)
        pairs=comparisons(details,y,part)
        for i,a,b,sign,n in pairs:
            counts['rank_pairs']+=1
            counts['rank_pair_correct']+=int(sign*float(details['plan_scores'][i][a]-details['plan_scores'][i][b])>0)
        for index,r in enumerate(part):
            p=int(details['plan_scores'][index].argmax());choice=details['plans'][index][p];k=len(r['refs'])
            counts['plans_generated']+=len(details['plans'][index]);counts['nonbase_plan_selected']+=p!=0
            for row,col in enumerate(choice):
                col=k if col<0 else col;counts['rows']+=1
                if r['known_options'][row,col]:counts['known_selected']+=1;counts['known_correct']+=int(r['positive'][row,col])
                else:counts['UNKNOWN_selected']+=1
        for name,out in [('cross','cross_logits'),('stability','stability_logits')]:
            target=y[name];pred=details[out]>=0
            for label in [0,1]:
                mask=target==label;question[name][f'{label}_support']+=int(mask.sum())
                question[name][f'{label}_correct']+=int((pred[mask]==label).sum())
    return dict(counts=dict(counts),questions={k:dict(v) for k,v in question.items()},
                known_correct_rate=counts['known_correct']/max(1,counts['known_selected']),
                rank_pair_accuracy=counts['rank_pair_correct']/counts['rank_pairs'] if counts['rank_pairs'] else None,
                scope='Deterministic reused TRAIN16 controller audit; correlated rows, not full MOT or independent generalization')
