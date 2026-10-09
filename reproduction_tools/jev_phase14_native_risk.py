"""Past-only offline consequences of actual commits; no GT actor inputs."""
import collections
from build_dense_jev_stage2_dataset import OfflineLabels
from jev_phase14_forensics import native_choice

class NativeRisk:
    def __init__(self,video,reader,prefix=None):
        self.labels=OfflineLabels(video,reader);self.votes=collections.defaultdict(collections.Counter)
        self.observations=collections.Counter();self.ever=set();self.latest={};self.counts=collections.Counter();self.queries={};self.trace=[];self.booted=False
        if prefix is not None:
            boundary=tuple(prefix['key'][1:])
            for index,inst in enumerate(prefix['instances']):
                key=divmod(index,2)
                if key>=boundary:break
                assert inst.has('track_ids') or len(inst)==0
                self.update(key,inst.track_ids.tolist() if inst.has('track_ids') else [],self.labels.current(*key))
            self.booted=True
    def update(self,key,ids,gts):
        assert len(ids)==len(gts)
        for ref,gt in zip(ids,gts):
            self.observations[ref]+=1
            if gt is not None:self.votes[ref][gt]+=1;self.ever.add(gt);self.latest[gt,key[1]]=(key[0],ref)
    def boot(self,view):
        if self.booted:return
        gts=self.labels.current(0,view);self.update((0,view),list(range(1,len(gts)+1)),gts);self.booted=True
    def before(self,**d):
        c=d['context'];key=(c['frame'],c['view'])
        if c.get('first'):self.boot(1-c['view'])
        if not len(d['logits']):return
        choice=native_choice(d['logits'].cpu().numpy(),d['batch']['legal'][0].cpu().numpy());targets=self.labels.current(*key)
        for qi,(row,col) in enumerate(zip(c.get('rows',list(range(len(choice)))),choice)):
            gt=targets[row];refs=d['refs'];legal=d['batch']['legal'][0,qi].tolist()
            support=[r for j,r in enumerate(refs) if legal[j] and gt is not None and set(self.votes[r])=={gt}]
            selected=None if col<0 else refs[col]
            known=selected is not None and len(self.votes[selected])==1
            tag='GT_UNKNOWN' if gt is None else 'DEFER' if selected is None else 'CERTIFIED_CORRECT' if known and gt in self.votes[selected] else 'CERTIFIED_WRONG' if known else 'SELECTED_UNKNOWN'
            task='MATCH' if d['task']==0 else 'REACT';self.counts[task+'_'+tag]+=1
            if d['task']==0:
                self.queries[key,row]=(support,selected,tag)
                self.counts['normal_supported']+=bool(support)
                self.counts['normal_correct_retained']+=bool(support) and tag=='CERTIFIED_CORRECT'
                self.counts['false_defer']+=bool(support) and selected is None
                self.counts['UNKNOWN_selected_with_correct_available']+=bool(support) and tag=='SELECTED_UNKNOWN'
    def after(self,**d):
        key=(d['frame'],d['view'])
        if d['first']:self.boot(1-d['view'])
        ids=d['instances'][-1].track_ids.tolist();gts=self.labels.current(*key);self.counts['payloads']+=1
        assert len(ids)==len(set(ids))
        for row,(ref,gt,e) in enumerate(zip(ids,gts,d['events'])):
            support,selected,tag=self.queries.get((key,row),([],None,'NO_QUERY'))
            before=self.votes[ref];known=len(before)==1;self.counts['detections']+=1;self.counts[e['action']]+=1
            if gt is not None:
                self.counts['wrong_anchor_certified_pure']+=known and gt not in before
                self.counts['commit_to_already_polluted']+=len(before)>1
                self.counts['new_cross_GT_gallery_mix']+=bool(before) and gt not in before
                self.counts['false_birth']+=e['action']=='START_NEW' and gt in self.ever
                self.counts['false_split_birth']+=e['action']=='START_NEW' and bool(support)
                self.counts['wrong_reactivation_certified']+=e['action']=='REACTIVATE' and known and gt not in before
                other=self.latest.get((gt,1-key[1]))
                self.counts['recent_cross_camera_queries']+=other is not None and key[0]-other[0]<=40
                self.counts['recent_cross_camera_ID_mismatch']+=other is not None and key[0]-other[0]<=40 and other[1]!=ref
                previous=self.latest.get((gt,key[1]))
                self.counts['observed_same_camera_fragment_hop']+=previous is not None and previous[1]!=ref
            self.trace.append(dict(key=list(key),row=row,identity=ref,GT_OFFLINE_ONLY=gt,action=e['action'],selected_tag=tag,pure_correct_refs=support))
        self.update(key,ids,gts)
    def summary(self):
        return dict(counts=dict(self.counts),normal_retention=self.counts['normal_correct_retained']/max(1,self.counts['normal_supported']),
            contaminated_observed_IDs=sum(len(v)>1 for v in self.votes.values()),
            scope='actual commits with IoU>=.5 past observed GT only; unknown excluded from wrong-anchor certification; fragment hops and40-frame cross-view mismatch are diagnostics, not CLEAR IDSW or CVIDF1')
