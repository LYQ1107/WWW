"""Equal-budget TRAIN-only pilots; reserved temporal blocks never receive gradients."""
import argparse
import collections
import copy
import gc
import math
import random
import time
import traceback
import numpy as np
import torch
from jev_phase14_common import *
from jev_phase13_learning import collate, atomic_torch
from gtr.modeling.jev_phase14.model import ReliableIdentityPolicy
from gtr.modeling.jev_stage2.assignment import lawful_choice
from jev_phase14_losses import objective

MODELS=['full','fixed_question','multi_question','set_transformer','motip','shared_mlp']
LOSSES=['standard_choice','legacy_structured','cost_sensitive','availability_joint']

def load_records():
    eligibility=read(REPORTS/'QUESTION_SUPERVISION_ELIGIBILITY.json');assert eligibility['status']=='PASS'
    train=[];audit=[];paired_train=[];paired_audit=[];manifests=[]
    for item in eligibility['videos']:
        data=item['source_DATASET']; lab=item['label_artifact']
        assert sha(data['path'])==data['SHA256'] and sha(lab['path'])==lab['SHA256']
        rs=torch.load(data['path'],map_location='cpu')['records'];ls=torch.load(lab['path'],map_location='cpu')['labels']
        for meta in ls:
            r=rs[meta['source_record_index']]
            if r['task']!=0:continue
            r=dict(r);r['availability']=meta['availability'];r['trust']=meta['trust'];r['withholdable_rows']=meta['withholdable_rows'];r['audit_block']=meta['audit_block']
            # Absence does not become certified just because a positive was not found.
            uncertain=(r['availability']<0)&r['positive'][:,-1]
            r['supervised']=r['supervised'].clone();r['supervised'][uncertain]=False
            (audit if meta['audit_block'] else train).append(r)
            if r['withholdable_rows']:(paired_audit if meta['audit_block'] else paired_train).append(r)
        manifests.append({'video':item['video'],'DATASET':data,'labels':lab})
    return train,audit,paired_train,paired_audit,manifests

def remove_options(r,removed):
    keep=~removed;k=int(keep.sum())
    out={key:value for key,value in r.items() if key!='inputs'}
    out['inputs']={key:value.clone() for key,value in r['inputs'].items()}
    for key in ['history_visual','history_mask','identity_meta','identity_mask']:out['inputs'][key]=out['inputs'][key][:,keep]
    for key in ['pair_evidence','legal']:out['inputs'][key]=out['inputs'][key][:,:,keep]
    out['refs']=[ref for ref,kept in zip(r['refs'],keep.tolist()) if kept]
    out['positive']=torch.cat([r['positive'][:,:-1][:,keep],r['positive'][:,-1:]],1).clone()
    out['known_options']=torch.cat([r['known_options'][:,:-1][:,keep],r['known_options'][:,-1:]],1).clone()
    out['availability']=r['availability'].clone();out['trust']=r['trust'][keep].clone()
    out['supervised']=r['supervised'].clone();out['targets']=torch.full_like(r['targets'],k)
    for i in range(len(r['rows'])):
        pos=out['positive'][i,:k]
        if pos.any():out['targets'][i]=int(torch.where(pos)[0][0]);out['availability'][i]=1
        elif r['availability'][i]==1 and bool(r['known_options'][i,:-1].all()):
            out['positive'][i]=False;out['positive'][i,-1]=True;out['known_options'][i,-1]=True;out['availability'][i]=0;out['supervised'][i]=True
    out['intervened']=True;out['withheld_original_refs']=[r['refs'][i] for i in torch.where(removed)[0].tolist()]
    return out

def withhold(r,row):
    assert row in r['withholdable_rows'] and bool(r['known_options'][row,:-1].all())
    positives=r['positive'][row,:-1];assert positives.any()
    out=remove_options(r,positives)
    assert out['availability'][row]==0 and out['positive'][row,-1]
    out['withheld_row']=row
    return out

def matched_presence(r,row):
    # Remove the same number of certified wrong identities, leaving all target
    # candidates present. This defeats candidate-cardinality-only absence cues.
    size=int(r['positive'][row,:-1].sum())
    wrong=torch.where(r['known_options'][row,:-1]&~r['positive'][row,:-1])[0]
    if len(wrong)<size:return None
    removed=torch.zeros(len(r['refs']),dtype=torch.bool);removed[wrong[:size]]=True
    out=remove_options(r,removed);out['withheld_row']=row;out['count_matched_presence']=True
    assert out['availability'][row]==1 and len(out['refs'])==len(r['refs'])-size
    return out

def batch(records,device='cuda:0'):
    x,y=collate(records,device,'full');b,q=x['question_mask'].shape;k=x['identity_mask'].shape[1]
    availability=torch.full((b,q),-1,dtype=torch.float32,device=device)
    trust=torch.full((b,q,k),-1,dtype=torch.float32,device=device)
    for i,r in enumerate(records):
        qq=len(r['rows']);kk=len(r['refs']);availability[i,:qq]=r['availability'].to(device)
        trust[i,:qq,:kk]=r['trust'].to(device)[None].expand(qq,-1)
    y.update(availability=availability,trust=trust)
    return x,y

def difficulty(r,row):
    if r['availability'][row]==1:return 'count_matched_presence' if r.get('count_matched_presence') else 'intervened_presence' if r.get('intervened') else 'natural_presence'
    if r['availability'][row]<0:return 'ambiguous'
    if not r.get('intervened'):return 'natural_absence'
    if len(r['refs'])>=19:return 'many_candidate_absence'
    values=r['inputs']['pair_evidence'][0,row,:,:4]
    maximum=float(values.max()) if values.numel() else -1.
    return 'hard_absence' if maximum>=.5 else 'easy_absence'

@torch.no_grad()
def assess(model,records,pairs,limit=256):
    model.eval();count=collections.Counter();strata=collections.defaultdict(collections.Counter)
    # Fixed systematic sampling covers the complete reserved TRAIN-block list.
    indices=np.linspace(0,len(records)-1,min(limit,len(records)),dtype=int).tolist() if records else []
    items=[records[i] for i in indices]
    paired_indices=np.linspace(0,len(pairs)-1,min(limit//2,len(pairs)),dtype=int).tolist() if pairs else []
    for i in paired_indices:
        r=pairs[i];row=r['withholdable_rows'][0];items.extend([r,withhold(r,row)])
        matched=matched_presence(r,row)
        if matched is not None:items.append(matched)
    for start in range(0,len(items),4):
        part=items[start:start+4];x,y=batch(part);out=model.details(x)
        for b,r in enumerate(part):
            qq=len(r['rows']);kk=len(r['refs']);z=torch.cat([out['logits'][b,:qq,:kk],out['logits'][b,:qq,-1:]],1)
            choices=lawful_choice(z,x['legal'][b,:qq,:kk])
            for row,c in enumerate(choices):
                count['rows']+=1;col=kk if c<0 else c
                supported=bool(r['positive'][row,:kk].any())
                if supported:
                    count['normal_supported']+=1;count['normal_retained']+=int(c>=0 and bool(r['positive'][row,col]))
                    count['false_defer']+=int(c<0)
                if bool(r['supervised'][row]):
                    count['certified_rows']+=1
                    label='UNKNOWN_choice' if not r['known_options'][row,col] else 'correct_choice' if r['positive'][row,col] else 'wrong_choice'
                    count[label]+=1
                if r['availability'][row]>=0:
                    a=int(r['availability'][row]);pred=int(out['availability_logits'][b,row]>=0)
                    count['availability_'+str(a)+'_support']+=1;count['availability_'+str(a)+'_correct']+=int(pred==a)
                    group=strata[difficulty(r,row)];group['rows']+=1;group['availability_correct']+=int(pred==a);group['false_absence' if a else 'false_presence']+=int(pred!=a)
                    if not a:group['DEFER_correct']+=int(c<0);group['incorrect_match']+=int(c>=0)
            tknown=r['trust']>=0
            if tknown.any():
                pred=(out['trust_logits'][b,0,:kk]>=0).cpu().long();truth=r['trust']
                for label in [0,1]:
                    mask=truth==label;count[f'trust_{label}_support']+=int(mask.sum());count[f'trust_{label}_correct']+=int((pred[mask]==label).sum())
    avail_rates=[count[f'availability_{label}_correct']/count[f'availability_{label}_support'] for label in [0,1] if count[f'availability_{label}_support']]
    trust_rates=[count[f'trust_{label}_correct']/count[f'trust_{label}_support'] for label in [0,1] if count[f'trust_{label}_support']]
    return dict(counts=dict(count),normal_retention=count['normal_retained']/max(1,count['normal_supported']),
                availability_balanced_accuracy=float(np.mean(avail_rates)) if len(avail_rates)==2 else None,
                trust_balanced_accuracy=float(np.mean(trust_rates)) if len(trust_rates)==2 else None,
                difficulty={k:dict(v) for k,v in strata.items()},scope='reserved TRAIN temporal blocks, natural and explicit paired withheld inputs; not a video tracking metric')

def main(variant,kind,seed=20261009,phase='pilot'):
    protect();assert read(REPORTS/'ERROR_ATTRIBUTION.json')['status']=='COMPLETE'
    assert variant in MODELS and kind in LOSSES
    protocol=read(REPORTS/'PREREGISTRATION.json');steps=protocol[phase]['updates']
    out=OUT/f'{phase}_v1'/variant/kind/f'seed{seed}';out.mkdir(parents=True,exist_ok=True)
    assert not (out/'RESULT.json').exists()
    torch.set_num_threads(1);torch.manual_seed(seed);np.random.seed(seed);rng=random.Random(seed)
    save(out/'PROGRESS.json',dict(status='LOADING_VERIFIED_TRAIN_DATA',variant=variant,loss=kind,seed=seed))
    train,audit,pairs,audit_pairs,manifests=load_records();assert train and pairs and audit
    groups=collections.defaultdict(list);pairgroups=collections.defaultdict(list)
    for r in train:
        if r['supervised'].any():groups[r['key'][0],r['key'][1]//64].append(r)
    for r in pairs:pairgroups[r['key'][0],r['key'][1]//64].append(r)
    keys=sorted(groups);pkeys=sorted(pairgroups)
    model=ReliableIdentityPolicy(variant,kind).cuda();opt=torch.optim.AdamW(model.parameters(),lr=.0001,weight_decay=.01)
    source=binding();save(out/'START.json',dict(binding=source,variant=variant,loss=kind,seed=seed,steps=steps,manifests=manifests,
        parameters=sum(p.numel() for p in model.parameters()),fresh_initialization=True,development_GT_training=False,
        training_groups=len(train),reserved_audit_groups=len(audit),withholding_train_groups=len(pairs),withholding_audit_groups=len(audit_pairs)))
    begin=time.monotonic();health=[];max_grad=0.;history=[];before=assess(model,audit,audit_pairs,128);step=0
    try:
        for step in range(1,steps+1):
            normal=[rng.choice(groups[rng.choice(keys)]) for _ in range(2)]
            paired=rng.choice(pairgroups[rng.choice(pkeys)]);row=rng.choice(paired['withholdable_rows'])
            matched=matched_presence(paired,row)
            part=normal+[matched if matched is not None else paired,withhold(paired,row)]
            x,y=batch(part);model.train();opt.zero_grad(set_to_none=True);details=model.details(x)
            loss,parts=objective(details,x,y,kind);assert torch.isfinite(loss),parts;loss.backward()
            grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.));assert math.isfinite(grad);max_grad=max(max_grad,grad);opt.step()
            if step%250==0:
                sample=assess(model,audit,audit_pairs,128);history.append(dict(update=step,assessment=sample,loss=parts))
                save(out/'PROGRESS.json',dict(status='RUNNING',update=step,total=steps,seconds=time.monotonic()-begin,assessment=sample))
                print('PHASE14_UPDATE',variant,kind,seed,step,sample['normal_retention'],sample['availability_balanced_accuracy'],flush=True)
        final=assess(model,audit,audit_pairs,256)
        ck=out/'LAST_FROZEN.pth';atomic_torch(ck,dict(model=model.state_dict(),variant=variant,objective=kind,seed=seed,actual_updates=step,binding=source,manifests=manifests))
        qualified=final['normal_retention']>=protocol['pilot']['normal_retention_min']
        if kind=='availability_joint':qualified=qualified and final['availability_balanced_accuracy'] is not None and final['availability_balanced_accuracy']>=.6
        result=dict(status='COMPLETE',binding=source,variant=variant,loss=kind,seed=seed,phase=phase,
                    actual_updates=step,checkpoint={'path':str(ck),'SHA256':sha(ck)},parameters=sum(p.numel() for p in model.parameters()),
                    before=before,final=final,pilot_qualified=qualified,history=history,gradient_health={'finite':True,'max_norm':max_grad},
                    seconds=time.monotonic()-begin,source_manifests=manifests,semantic_scope='WHO legal joint choice + optional availability/trust; native REACT and WRITE remain frozen fallback')
        save(out/'RESULT.json',result);save(out/'PROGRESS.json',dict(status='COMPLETE',actual_updates=step,pilot_qualified=qualified))
        print('PHASE14_TRAIN_COMPLETE',variant,kind,seed,qualified,final,flush=True)
    except Exception:
        save(out/'FAILED.json',dict(status='FAIL',binding=source,update=step,traceback=traceback.format_exc()));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--variant',choices=MODELS,required=True);p.add_argument('--loss',choices=LOSSES,required=True);p.add_argument('--seed',type=int,default=20261009);p.add_argument('--phase',choices=['pilot','formal'],default='pilot');a=p.parse_args();main(a.variant,a.loss,a.seed,a.phase)
