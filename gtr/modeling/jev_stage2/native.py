"""Direct scores replace GTA; preserve original Instances/Gallery/bank/ID commits."""
import time,torch
from detectron2.structures import Instances
from .memory import IdentityMemory
from .assignment import lawful_choice

class NativeDirectExecutor:
    def __init__(self,policy,mode='JEV_DIRECT',temperature=(1.,1.),variant='full',react_learned=True):
        assert mode in ['OFF','SHADOW','JEV_DIRECT'];self.mode=mode;self.policy=policy;self.temperature=temperature;self.variant=variant;self.react_learned=react_learned;self.memory=IdentityMemory();self.observer=None;self.commit_observer=None;self.latency=[]
    def reset(self):self.memory=IdentityMemory();self.latency=[]
    def initialize(self,instances,frame,view,first):
        if first and not self.memory.meta:
            for inst in instances[:-1]:
                for row,ref in enumerate(inst.track_ids.tolist()):self.memory.update(ref,inst.reid_features[row,:1024],inst.pred_boxes.tensor[row],inst.image_size,frame,1-view)
    def batch(self,current,refs,galleries,frame,view,active):
        return self.memory.build(current,refs,galleries,frame,view,active,no_cross=self.variant=='no_cross_camera',no_long=self.variant=='no_long_term')
    def scores(self,batch,refs,context,task):
        batch['question_type'].fill_(task)
        if self.policy is None or task==1 and not self.react_learned:
            values=batch['pair_evidence'][0,:,:,[0,1,2]].max(-1).values;logits=torch.cat([values,values.new_full((len(values),1),.75)],1)
        else:logits=self.policy(batch)[0]/self.temperature[task]
        if self.observer is not None:self.observer(batch=batch,refs=refs,context=context,task=task,logits=logits)
        return logits
    def shadow(self,instances,k,galleries,frame,view,first):
        current=instances[k];refs=sorted({int(ref) for inst in instances[:k] for ref in inst.track_ids.tolist()});self.initialize(instances,frame,view,first)
        if len(current):self.scores(self.batch(current,refs,galleries,frame,view,set(refs)),refs,{'frame':frame,'view':view,'shadow':True},0)
    def promote_bank(self,model,active,galleries,instances_old):
        from ..meta_arch.gtr_rcnn import poss_ids,old_reids
        if not len(instances_old):return
        for ref in poss_ids.poss_ids.copy():
            if ref not in active and len(galleries.get(ref,[]))>=model.bank_size:
                poss_ids.poss_ids.remove(ref);average=sum(galleries[ref][-1-i].reid_features for i in range(model.bank_size))/model.bank_size
                old_reids.old_reids.append(galleries[ref][0]);old_reids.old_reids[0 if len(old_reids.old_reids)==1 else 1].reid_features=average;old_reids.old_reids=[Instances.cat(old_reids.old_reids)]
    def commit(self,model,instances,k,id_count,hits,galleries,*,frame,view,first,instances_old=()):
        from ..meta_arch.gtr_rcnn import poss_ids,old_reids
        self.initialize(instances,frame,view,first);current=instances[k];d=len(current);active=sorted({int(t) for inst in instances[:k] for t in inst.track_ids.tolist()});active_set=set(active)
        torch.cuda.synchronize();started=time.perf_counter();build_start=started;x=self.batch(current,active,galleries,frame,view,active_set);torch.cuda.synchronize();built=time.perf_counter()
        with torch.no_grad():logits=self.scores(x,active,{'frame':frame,'view':view,'first':first},0)
        choices=lawful_choice(logits,x['legal'][0]);ids=current.reid_features.new_full((d,),-1,dtype=torch.long);actions=['DEFER']*d
        for row,col in enumerate(choices):
            if col>=0:ids[row]=active[col];actions[row]='ASSOCIATE_EXISTING'
        unmatched=[r for r in range(d) if ids[r]<0];stale=[]
        if unmatched and not first and model.with_bank:
            self.promote_bank(model,active_set,galleries,instances_old)
            if old_reids.old_reids:
                stale=sorted(set(old_reids.old_reids[0].track_ids.tolist())-set(ids[ids>=0].tolist()))
        if unmatched:
            sub=current[unmatched];rx=self.batch(sub,stale,galleries,frame,view,active_set)
            with torch.no_grad():react=self.scores(rx,stale,{'frame':frame,'view':view,'first':first,'rows':unmatched},1)
            restore=lawful_choice(react,rx['legal'][0])
            for row,col in zip(unmatched,restore):
                if col>=0:
                    ref=stale[col];ids[row]=ref;actions[row]='REACTIVATE';poss_ids.poss_ids.add(ref)
                    pooled=old_reids.old_reids[0];old_reids.old_reids=[pooled[pooled.track_ids!=ref]]
                    if len(old_reids.old_reids[0])==0:old_reids.old_reids=[]
        current.track_ids=ids;events=[]
        for row in range(d):
            ref=int(ids[row]);before=len(galleries[ref]) if ref>=0 else 0
            if ref<0:
                id_count+=1;ref=id_count;ids[row]=ref;current.track_ids=ids;hits[ref]=1;galleries[ref]=current[row];actions[row]='START_NEW'
            else:
                hits[ref]+=1;current.track_ids=ids;galleries[ref]=Instances.cat([galleries[ref],current[row]])
                if not first and len(galleries[ref])==model.bank_size+1:poss_ids.poss_ids.add(ref)
            self.memory.update(ref,current.reid_features[row,:1024],current.pred_boxes.tensor[row],current.image_size,frame,view)
            events.append({'row':row,'identity':ref,'action':actions[row],'memory':'WRITE_FROZEN_NATIVE_FALLBACK','gallery_before':before,'gallery_after':len(galleries[ref]),'hits':hits[ref]})
        current.track_ids=ids;assert len(ids)==len(ids.unique())
        torch.cuda.synchronize();end=time.perf_counter();self.latency.append({'frame':frame,'view':view,'build_ms':(built-build_start)*1000,'total_stage2_ms':(end-started)*1000,'MATCH_Q':d,'REACT_Q':len(unmatched),'active_K':len(active),'stale_K':len(stale)})
        model._jev_candidate_last=None;model._candidate_observe_commit(instances,id_count,hits,galleries,view=view,frame_index=frame,first=first)
        if self.commit_observer is not None:self.commit_observer(instances=instances,id_count=id_count,hits=hits,galleries=galleries,frame=frame,view=view,first=first,events=events)
        return instances,id_count,hits,galleries
