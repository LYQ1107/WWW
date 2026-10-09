"""One matched extra4k round: own-state/new, own-state/old, off-policy/new."""
import argparse
import collections
import math
import random
import time
import traceback
import numpy as np
import torch
from jev_phase14_common import *
from jev_phase14_train import load_records,batch,assess,withhold,matched_presence
from jev_phase14_online import load_policy
from jev_phase14_losses import objective
from jev_phase13_learning import atomic_torch
from jev_phase14_forensics import native_choice
from jev_phase14_artifacts import load_dense

ARMS=['onpolicy','oldloss_onpolicy','offpolicy']

def group_labels(record):
    if 'strata_OFFLINE_ONLY' in record:return
    r=record;values=r['inputs']['pair_evidence'][0,:,:,[0,1,2]].max(-1).values
    z=torch.cat([values,values.new_full((len(values),1),.75)],-1)
    selected=native_choice(z.numpy(),r['inputs']['legal'][0].numpy());groups=[]
    for row,c in enumerate(selected):
        correct=r['positive'][row,:-1].any()
        if r['availability'][row]==0:g='certified_correct_absent'
        elif correct and c<0:g='false_split_prefix'
        elif c>=0 and r['known_options'][row,c] and not r['positive'][row,c]:g='false_merge_prefix'
        elif correct and (r['trust']==0).any():g='polluted_with_reliable_candidate'
        elif correct:g='normal_continuation'
        else:g='ambiguous_or_unavailable'
        groups.append(g)
    r['strata_OFFLINE_ONLY']=groups
    r['identity_time_groups_OFFLINE_ONLY']=[(r['key'][0],gt,r['key'][1]//64) for gt in r['GT_labels_OFFLINE_ONLY']]

class BalancedSampler:
    def __init__(self,records):
        self.groups=collections.defaultdict(lambda:collections.defaultdict(list));self.pairs=collections.defaultdict(list)
        for r in records:
            if not r['supervised'].any():continue
            group_labels(r)
            for group,key in set(zip(r['strata_OFFLINE_ONLY'],r['identity_time_groups_OFFLINE_ONLY'])):
                self.groups[group][tuple(key)].append(r)
            if r['withholdable_rows']:self.pairs[r['key'][0],r['key'][1]//64].append(r)
        self.strata=sorted(self.groups);self.keys={s:sorted(self.groups[s],key=str) for s in self.strata};self.pkeys=sorted(self.pairs)
        assert self.strata
    def normal(self,rng):
        s=rng.choice(self.strata);k=rng.choice(self.keys[s]);return rng.choice(self.groups[s][k])
    def next(self,rng):
        normal=[self.normal(rng) for _ in range(2)]
        if not self.pkeys:return normal+[self.normal(rng) for _ in range(2)]
        r=rng.choice(self.pairs[rng.choice(self.pkeys)]);row=rng.choice(r['withholdable_rows']);present=matched_presence(r,row)
        return normal+[present if present is not None else r,withhold(r,row)]

def own_records(variant,seed):
    train=[];audit=[];manifests=[];positive=0;contaminated=0;contaminated_videos=0
    for video in TRAIN:
        path=OUT/'onpolicy_collect_v2'/variant/f'seed{seed}'/f'video{video:02d}/RESULT.json';r=read(path);d=r['DATASET'];assert r['status']=='COMPLETE' and sha(d['path'])==d['SHA256']
        rs=load_dense(d['path'])['records'];mix=0
        for item in rs:
            if item['task']!=0:continue
            item['audit_block']=(item['key'][1]//64)%4==3
            (audit if item['audit_block'] else train).append(item)
            if not item['audit_block']:
                positive+=int((item['availability']==1).sum());mix+=int((item['trust']==0).sum())
        contaminated+=mix;contaminated_videos+=int(mix>0);manifests.append(dict(video=video,result=dict(path=str(path),SHA256=sha(path)),DATASET=d))
    qualification=dict(positive_rows=positive,contaminated_options=contaminated,contaminated_videos=contaminated_videos)
    qualified=positive>=100 and contaminated>=50 and contaminated_videos>=2
    return train,audit,manifests,qualified,qualification

def main(variant,seed,arm):
    protect();assert arm in ARMS;protocol=read(REPORTS/'ON_POLICY_PROTOCOL.json');source=binding();torch.set_num_threads(1)
    torch.manual_seed(seed);np.random.seed(seed);rng=random.Random(seed)
    out=OUT/f'{arm}_v2'/variant/'availability_joint'/f'seed{seed}';out.mkdir(parents=True,exist_ok=True)
    assert not (out/'RESULT.json').exists()
    own,own_audit,own_manifest,qualified,support=own_records(variant,seed)
    if not qualified:
        save(out/'NOT_RUN.json',dict(status='QUALIFICATION_FAILED',binding=source,own_state_label_support=support,manifests=own_manifest,no_new_training=True));return
    original,audit,original_pairs,audit_pairs,off_manifest=load_records()
    records=original if arm=='offpolicy' else own;sampler=BalancedSampler(records)
    model,initial,_=load_policy(variant,seed,'formal');model.train();kind='legacy_structured' if arm=='oldloss_onpolicy' else 'availability_joint'
    # Same inference mapping, parameters and solver across all arms. The old
    # objective receives the same lawful candidates and source-specific labels.
    assert model.objective=='availability_joint'
    opt=torch.optim.AdamW(model.parameters(),lr=protocol['lr'],weight_decay=protocol['weight_decay']);steps=protocol['extra_updates']
    save(out/'START.json',dict(binding=source,arm=arm,initial=initial,steps=steps,objective=kind,
        sampler_groups={s:len(sampler.keys[s]) for s in sampler.strata},paired_blocks=len(sampler.pkeys),
        own_manifest=own_manifest,original_manifest=off_manifest,own_state_label_support=support,
        development_GT_training=False,optimizer_reset_identical_all_arms=True,inference_objective=model.objective))
    history=[];begin=time.monotonic();max_grad=0.;step=0
    before=assess(model,audit,audit_pairs,128)
    try:
        for step in range(1,steps+1):
            part=sampler.next(rng);x,y=batch(part);model.train();opt.zero_grad(set_to_none=True)
            loss,parts=objective(model.details(x),x,y,kind);assert torch.isfinite(loss)
            loss.backward();grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),protocol['grad_clip']));assert math.isfinite(grad)
            max_grad=max(max_grad,grad);opt.step()
            if step%250==0:
                a=assess(model,audit,audit_pairs,128);history.append(dict(update=step,assessment=a,loss=parts))
                save(out/'PROGRESS.json',dict(status='RUNNING',extra_update=step,total=steps,assessment=a,seconds=time.monotonic()-begin))
                print('PHASE14_MATCHED_EXTRA_UPDATE',variant,seed,arm,step,a['normal_retention'],flush=True)
        final=assess(model,audit,audit_pairs,256);own_pairs=[r for r in own_audit if r['withholdable_rows']];own_final=assess(model,own_audit,own_pairs,256)
        ck=out/'LAST_FROZEN.pth';atomic_torch(ck,dict(model=model.state_dict(),variant=variant,objective='availability_joint',training_objective=kind,seed=seed,actual_updates=20000+step,binding=source,initial=initial))
        result=dict(status='COMPLETE',binding=source,variant=variant,seed=seed,phase=arm,arm=arm,loss='availability_joint',training_objective=kind,
            actual_updates=20000+step,extra_updates=step,initial=initial,checkpoint=dict(path=str(ck),SHA256=sha(ck)),
            source_manifests={'own_state':own_manifest,'offpolicy':off_manifest},before=before,final=final,own_reserved_final=own_final,
            history=history,gradient_health=dict(finite=True,max_norm=max_grad),own_state_label_support=support,
            sampler_groups={s:len(sampler.keys[s]) for s in sampler.strata},seconds=time.monotonic()-begin,
            inference_mapping_identical_all_arms=True,semantic_scope='20k+4k budget-matched, learned MATCH; frozen native REACT and MEMORY fallbacks')
        save(out/'RESULT.json',result);save(out/'PROGRESS.json',dict(status='COMPLETE',extra_update=step));print('PHASE14_MATCHED_EXTRA_COMPLETE',variant,seed,arm,flush=True)
    except Exception:
        save(out/'FAILED.json',dict(status='FAIL',binding=source,extra_update=step,traceback=traceback.format_exc()));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--variant',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--arm',choices=ARMS,required=True);a=p.parse_args();main(a.variant,a.seed,a.arm)
