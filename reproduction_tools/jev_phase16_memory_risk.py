"""Observer-only anchored ownership, strict split risk and observed propagation."""
import collections
from jev_phase16_common import duplicate_masked_labels
from jev_phase14_native_risk import NativeRisk
from gtr.modeling.jev_phase16.evidence_ledger import EvidenceLedger


class MemoryRisk(NativeRisk):
    def __init__(self, video, reader, prefix=None):
        super().__init__(video,reader)
        self.labels=duplicate_masked_labels(video,reader)
        self.booted=prefix is not None;self.owner={};self.qualified_support={};self.strong_trace=[]
        if prefix is None:return
        ledger=EvidenceLedger.from_prefix(prefix)
        for identity,records in ledger.records.items():
            for r in records:
                gt=self.labels.current(*r['key'][:2])[r['key'][2]]
                self.observations[identity]+=1
                if gt is not None:
                    self.votes[identity][gt]+=1;self.ever.add(gt)
                    old=self.latest.get((gt,r['key'][1]))
                    if old is None or r['key'][0]>=old[0]:self.latest[gt,r['key'][1]]=(r['key'][0],identity)
                self.certify(identity)
        self.prefix_votes={i:dict(v) for i,v in self.votes.items()}
        self.prefix_latest=dict(self.latest);self.prefix_ever=set(self.ever)

    def certify(self, identity):
        v=self.votes[identity]
        if identity not in self.owner and len(v)==1 and sum(v.values())>=3 and sum(v.values())/self.observations[identity]>=.8:
            self.owner[identity]=next(iter(v))

    def before(self, **d):
        super().before(**d)
        for identity in list(self.votes):self.certify(identity)
        if d['task']!=0:return
        c=d['context'];key=c['frame'],c['view'];gts=self.labels.current(*key)
        legal=d['batch']['legal'][0].cpu().tolist()
        for row,gt in enumerate(gts):
            self.qualified_support[key,row]=[r for j,r in enumerate(d['refs']) if legal[row][j] and gt is not None and
                set(self.votes[r])=={gt} and sum(self.votes[r].values())>=3 and
                sum(self.votes[r].values())/max(1,self.observations[r])>=.8]

    def after(self, **d):
        key=d['frame'],d['view'];gts=self.labels.current(*key);inst=d['instances'][-1]
        for row,(identity,gt,event) in enumerate(zip(inst.track_ids.tolist(),gts,d['events'])):
            v=self.votes[identity];owner=self.owner.get(identity);previous=self.latest.get((gt,key[1])) if gt is not None else None
            self.strong_trace.append(dict(key=list(key),row=row,identity=identity,GT_OFFLINE_ONLY=gt,
                immutable_owner_GT_OFFLINE_ONLY=owner,owner_assessed=owner is not None and gt is not None,
                wrong_owner=owner is not None and gt is not None and owner!=gt,
                new_cross_GT_gallery_mix=gt is not None and bool(v) and gt not in v,
                already_polluted_write=gt is not None and len(v)>1,
                false_split_birth=event['action']=='START_NEW' and bool(self.qualified_support.get((key,row))),
                false_birth_seen_GT_proxy=event['action']=='START_NEW' and gt is not None and gt in self.ever,
                action=event['action'],owner_correct=owner is not None and gt==owner,
                previous_same_camera_ID=previous[1] if previous else None,
                independent_cluster=[d.get('video',0),gt,key[1],key[0]//64]))
        super().after(**d)
        for identity in inst.track_ids.tolist():self.certify(identity)

    def horizon(self, boundary, end):
        rows=[r for r in self.strong_trace if boundary<=tuple(r['key']) and r['key'][0]<end]
        counts=collections.Counter();episodes=[];active={}
        for r in rows:
            for field in ['owner_assessed','wrong_owner','new_cross_GT_gallery_mix','already_polluted_write','false_split_birth','false_birth_seen_GT_proxy','owner_correct']:
                counts[field]+=int(r[field])
            if r['GT_OFFLINE_ONLY'] is None:continue
            key=r['GT_OFFLINE_ONLY'],r['key'][1];frame=r['key'][0]
            old=active.get(key)
            if r['wrong_owner']:
                if old is None or frame!=old['end']+1:
                    if old:old['right_censored']=True;old['end_reason']='observation gap';episodes.append(old)
                    active[key]=dict(GT_OFFLINE_ONLY=key[0],view=key[1],start=frame,end=frame,observed_wrong_frames=1,right_censored=False)
                else:old['end']=frame;old['observed_wrong_frames']+=1
            elif old:
                old=active.pop(key);old['right_censored']=frame!=old['end']+1
                old['end_reason']='next observed non-wrong owner' if not old['right_censored'] else 'gap before next non-wrong observation'
                episodes.append(old)
        for e in active.values():e['right_censored']=True;e['end_reason']='horizon/video end';episodes.append(e)
        return dict(counts=dict(counts),wrong_owner_observed_frames=counts['wrong_owner'],
            wrong_owner_episode_max_frames=max((e['observed_wrong_frames'] for e in episodes),default=0),
            error_propagation_intervals=episodes,observed_episode_scope='IoU>=.5 detector-observed owner errors; gaps split episodes; absence and video/horizon end censored, not continuous invisible duration',
            owners='Immutable earliest past-only single-GT >=3 and>=.8observed coverage; mixed histories retain previously established owner; uncertifiable IDs UNKNOWN',
            UNKNOWN_current=sum(r['GT_OFFLINE_ONLY'] is None for r in rows),
            UNKNOWN_owner=sum(r['GT_OFFLINE_ONLY'] is not None and not r['owner_assessed'] for r in rows))
