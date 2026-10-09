"""GT labels saved only after real learned-policy logits, never enter native actor."""
import argparse
import collections
import time
import torch
from build_dense_jev_stage2_dataset import OfflineLabels
from jev_phase13_runtime import build_tracker,cache_inputs,run
from jev_phase14_common import *
from jev_phase14_online import load_policy
from jev_phase14_artifacts import save_dense
from jev_phase14_forensics import native_choice
from gtr.modeling.jev_phase14.memory import CachedIdentityMemory

class OwnStateLabels(OfflineLabels):
    def __init__(self,video,reader):
        super().__init__(video,reader);self.votes=collections.defaultdict(collections.Counter);self.total=collections.Counter();self.booted=False;self.strata=collections.Counter()
    def boot(self,frame,view):
        if self.booted:return
        for ref,gt in enumerate(self.current(frame,view),1):
            self.total[ref]+=1
            if gt is not None:self.votes[ref][gt]+=1
        self.booted=True
    def before(self,**d):
        c=d['context']
        if c.get('first'):self.boot(0,1-c['view'])
        length=len(self.records);super().before(**d)
        if len(self.records)==length:return
        r=self.records[-1];q=len(r['rows']);k=len(r['refs'])
        availability=torch.full((q,),-1,dtype=torch.int8);trust=torch.full((k,),-1,dtype=torch.int8)
        for col,ref in enumerate(r['refs']):
            votes=self.votes[ref];known=sum(votes.values());coverage=known/max(1,self.total[ref])
            if known>=3 and coverage>=.8:
                if len(votes)==1:trust[col]=1
                elif sum(n>=2 for n in votes.values())>=2:trust[col]=0
        choice=native_choice(d['logits'].cpu().numpy(),d['batch']['legal'][0].cpu().numpy())
        eligible=[];groups=[];anchors=[]
        for row,gt in enumerate(r['GT_labels_OFFLINE_ONLY']):
            positive=r['positive'][row,:k];known=r['known_options'][row,:k]
            if gt is not None:
                if positive.any():availability[row]=1
                elif known.all():availability[row]=0
                if positive.any() and known.all():eligible.append(row)
            if r['task']==1 and positive.any():group='certified_stale_recovery'
            elif availability[row]==0:group='certified_correct_absent'
            elif positive.any() and choice[row]<0:group='false_split_prefix'
            elif choice[row]>=0 and known[choice[row]] and not positive[choice[row]]:group='false_merge_prefix'
            elif positive.any() and any(len(self.votes[t])>1 for t in r['refs']):group='polluted_with_reliable_candidate'
            elif positive.any():group='normal_continuation'
            else:group='ambiguous_or_unavailable'
            groups.append(group);anchors.append(gt);self.strata[group]+=1
        uncertain=(availability<0)&r['positive'][:,-1]
        r['supervised']=r['supervised'].clone();r['supervised'][uncertain]=False
        r.update(availability=availability,trust=trust,withholdable_rows=eligible,
            audit_block=(r['key'][1]//64)%4==3,strata_OFFLINE_ONLY=groups,
            identity_time_groups_OFFLINE_ONLY=[(self.video,gt,r['key'][1]//64) for gt in anchors],
            native_actual_choice_OFFLINE_ONLY=choice)
    def after(self,**d):
        if d['first']:self.boot(d['frame'],1-d['view'])
        super().after(**d)
        for ref,gt in zip(d['instances'][-1].track_ids.tolist(),self.current(d['frame'],d['view'])):
            self.total[ref]+=1
            if gt is not None:self.votes[ref][gt]+=1

def main(video,variant,seed):
    protect();assert video in TRAIN;assert read(REPORTS/'ON_POLICY_PROTOCOL.json')['status']=='FROZEN_BEFORE_NEW_ONPOLICY_OUTCOMES'
    assert read(REPORTS/'NATIVE_CACHE_PARITY.json')['status']=='PASS'
    source=binding();torch.set_num_threads(1);torch.manual_seed(20261009)
    policy,trained,_=load_policy(variant,seed,'formal');values,frames,reader=cache_inputs(video)
    out=OUT/'onpolicy_collect_v2'/variant/f'seed{seed}'/f'video{video:02d}';out.mkdir(parents=True,exist_ok=True)
    assert not (out/'RESULT.json').exists(),'completed own-state data immutable'
    labels=OwnStateLabels(video,reader);model=build_tracker(video,policy=policy,variant='full',react_learned=False)
    ex=model.jev_stage2_executor;ex.memory_factory=CachedIdentityMemory;ex.observer=labels.before;ex.commit_observer=labels.after
    save(out/'START.json',dict(binding=source,trained=trained,frames=256,GT_actor_inputs=False));begin=time.monotonic()
    with torch.no_grad():raw,_=run(model,values,frames,stop=255)
    ck=out/'DATASET.pth.xz';save_dense(ck,dict(records=labels.records,video=video,trained=trained,
        scope='actual policy-mutated TRAIN first256 scene frames; offline labels independent of actor; no teacher-forced histories'))
    result=dict(status='COMPLETE',binding=source,video=video,variant=variant,seed=seed,trained=trained,
        frames=256,records=len(labels.records),DATASET=dict(path=str(ck),SHA256=sha(ck),bytes=ck.stat().st_size),
        counts=dict(labels.counts),strata=dict(labels.strata),withholdable_rows=sum(len(r['withholdable_rows']) for r in labels.records if r['task']==0),
        GT_actor_inputs=False,native_REACT_training=False,native_MEMORY_reward_training=False,seconds=time.monotonic()-begin)
    save(out/'RESULT.json',result);print('PHASE14_OWN_STATE_DATA_COMPLETE',variant,seed,video,result['strata'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--video',type=int,required=True);p.add_argument('--variant',required=True);p.add_argument('--seed',type=int,required=True);a=p.parse_args();main(a.video,a.variant,a.seed)
