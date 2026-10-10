"""All-query frozen reader comparisons on an unchanged real native trajectory."""
import argparse
import collections
import gzip
import itertools
import time
from jev_phase16_common import *
import numpy as np
import torch
from gtr.modeling.jev_phase16.multi_prototype_memory import MultiPrototypeMemory,VARIANTS
from gtr.modeling.jev_phase15.native_commit_adapter import attach
from gtr.modeling.jev_stage2.assignment import lawful_choice
from gtr.modeling.jev_native_state import fingerprint
from jev_phase13_runtime import build_tracker,cache_inputs,run
from jev_phase15_common import perception_provenance


class TraceCursor:
    def __init__(self,path):
        self.stream=gzip.open(path,'rt')
        self.groups=itertools.groupby((json.loads(x) for x in self.stream),lambda r:tuple(r['key']))
        self.key=None;self.records=[];self.advance()
    def advance(self):
        key,group=next(self.groups,(None,[]));self.key=key;self.records=list(group)
    def at(self,key):
        while self.key is not None and self.key<key:self.advance()
        return self.records if self.key==key else []


def quantile(values):
    return dict(zip(['p50','p95','max'],map(float,np.quantile(values,[.5,.95,1])))) if values else None


class PrototypeAudit:
    def __init__(self,video,reader,policy,cursor,stream):
        self.video=video;self.labels=duplicate_masked_labels(video,reader);self.policy=policy
        self.cursor=cursor;self.stream=stream;self.views={v:MultiPrototypeMemory(v) for v in VARIANTS[1:]}
        self.counts=collections.defaultdict(collections.Counter);self.clusters=collections.defaultdict(set)
        self.read_ms=collections.defaultdict(list);self.forward_ms=collections.defaultdict(list)
        self.votes=collections.defaultdict(collections.Counter);self.observed=collections.Counter();self.owner={}
        self.past_GT={};self.booted=False;self.actual_payloads=0;self.raw_vectors_max=0

    def append(self,inst,frame,view):
        for m in self.views.values():m.append_instance(inst,frame,view)
        targets=self.labels.current(frame,view)
        for row,(identity,gt) in enumerate(zip(inst.track_ids.tolist() if inst.has('track_ids') else [],targets)):
            self.past_GT[frame,view,row]=gt;self.observed[identity]+=1
            if gt is not None:self.votes[identity][gt]+=1
            votes=self.votes[identity]
            if identity not in self.owner and len(votes)==1 and sum(votes.values())>=3 and sum(votes.values())/self.observed[identity]>=.8:
                self.owner[identity]=next(iter(votes))

    def native_before(self,**d):
        if d['first'] and not self.booted:
            other=1-d['view'];self.append(d['instances'][other],0,other);self.booted=True

    def before(self,**d):
        if not len(d['logits']):return
        context=d['context'];key=(self.video,context['frame'],context['view']);refs=list(d['refs'])
        task='MATCH' if d['task']==0 else 'REACT';rows=context.get('rows',list(range(len(d['logits']))))
        source={r['row']:r for r in self.cursor.at(key) if r.get('record_type')!='COMMIT' and r['task']==task}
        assert set(rows)==set(source),(key,task,'query rows differ from frozen P0')
        for ri,row in enumerate(rows):
            assert source[row]['actual_candidates']==refs
            assert source[row]['actual_frozen_logits']==d['logits'][ri].cpu().tolist(),'changed original frozen scores'
        saved_details=getattr(self.policy,'runtime_details',None);results={};provenance={}
        try:
            for variant in VARIANTS:
                x=d['batch'];torch.cuda.synchronize();begin=time.perf_counter()
                if variant!=VARIANTS[0]:x,provenance[variant]=self.views[variant].batch(x,refs,context['view'])
                torch.cuda.synchronize();read_end=time.perf_counter()
                if task=='MATCH':logits=self.policy(x)[0]
                else:
                    value=x['pair_evidence'][0,:,:,:3].max(-1).values
                    logits=torch.cat([value,value.new_full((len(value),1),.75)],1)
                torch.cuda.synchronize();end=time.perf_counter()
                if variant==VARIANTS[0]:assert torch.equal(logits,d['logits']),'A reforward changed native scores'
                self.read_ms[variant].append((read_end-begin)*1000);self.forward_ms[variant+'_'+task].append((end-read_end)*1000)
                details=getattr(self.policy,'runtime_details',None) if task=='MATCH' else None
                who=details['choice_logits'][0] if details is not None else None
                results[variant]=(x,logits,lawful_choice(logits,x['legal'][0]),
                    lawful_choice(who,x['legal'][0]) if who is not None else None,who)
        finally:
            if saved_details is not None:self.policy.runtime_details=saved_details
        for ri,row in enumerate(rows):
            record=source[row];gt=record['current_GT_OFFLINE_ONLY'];cluster=(self.video,gt,key[2],key[1]//64)
            legal={r for col,r in enumerate(refs) if bool(d['batch']['legal'][0,ri,col])}
            eligible=[r for r in record['anchored_correct_owner_IDs'] if r in legal and record['raw_and_summary_scores'][str(r)]['raw_target']>=.75]
            pure_wrong=[r for r in legal if gt is not None and len(self.votes[r])==1 and
                sum(self.votes[r].values())>=3 and sum(self.votes[r].values())/self.observed[r]>=.8 and gt not in self.votes[r]]
            owner_wrong=[r for r in legal if gt is not None and r in self.owner and self.owner[r]!=gt]
            paired={}
            for variant,(x,logits,choices,who_choices,who_logits) in results.items():
                col=choices[ri];chosen=refs[col] if col>=0 else None;c=self.counts[variant+'_'+task]
                scores=x['pair_evidence'][0,ri,:,:4].max(-1).values
                support=bool(eligible) and any(float(scores[refs.index(r)])>=.75 for r in eligible)
                false=any(float(scores[refs.index(r)])>=.75 for r in pure_wrong)
                wrong_owner=any(float(scores[refs.index(r)])>=.75 for r in owner_wrong)
                c['queries']+=1;c['GT_known']+=gt is not None
                c['negative_denominator_GT_known_legal']+=gt is not None and bool(legal)
                c['positive_eligible']+=bool(eligible);c['positive_supported']+=support
                c['pure_wrong_activated_queries']+=false;c['known_wrong_owner_activated_queries']+=wrong_owner
                c['WHO_pure_correct_selected']+=chosen in record['legal_pure_correct_IDs']
                c['WHO_pure_wrong_selected']+=chosen in pure_wrong
                c['immutable_correct_owner_selected']+=chosen is not None and gt is not None and self.owner.get(chosen)==gt
                c['immutable_wrong_owner_selected']+=chosen in owner_wrong
                selected_votes=self.votes[chosen] if chosen is not None else {}
                certified=chosen is not None and len(selected_votes)==1 and sum(selected_votes.values())>=3 and sum(selected_votes.values())/self.observed[chosen]>=.8
                c['WHO_certificate_UNKNOWN_joint_selection']+=gt is not None and chosen is not None and not certified
                c['mixed_history_joint_selection']+=gt is not None and chosen is not None and len(selected_votes)>1
                who_chosen=refs[who_choices[ri]] if who_choices is not None and who_choices[ri]>=0 else None
                if who_choices is not None:
                    c['WHO_head_certified_correct_selected']+=who_chosen in record['legal_pure_correct_IDs']
                    c['WHO_head_certified_wrong_selected']+=who_chosen in pure_wrong
                c['DEFER']+=chosen is None
                c['changed_MATCH_column']+=chosen!=record['selected_ID']
                c['high_risk_positive_eligible']+=bool(eligible) and record['high_risk']
                c['high_risk_positive_supported']+=support and record['high_risk']
                local=False
                if variant!=VARIANTS[0]:
                    for i,r in enumerate(refs):
                        if r not in eligible:continue
                        for slot,item in enumerate(provenance[variant][i]):
                            if item is None:continue
                            targets=[self.past_GT[k] for k in item['keys']];v=collections.Counter(g for g in targets if g is not None)
                            qualified=sum(v.values())>=3 and sum(v.values())/len(targets)>=.8 and set(v)=={gt}
                            local=local or qualified and float(x['pair_evidence'][0,ri,i,slot])>=.75
                c['pure_local_segment_supported']+=local
                if eligible and gt is not None:self.clusters[variant+'_'+task].add(cluster)
                a_support=paired[VARIANTS[0]]['positive_supported'] if variant!=VARIANTS[0] else support
                if support and not a_support:self.clusters[variant+'_'+task+'_new_support_vs_A'].add(cluster)
                paired[variant]=dict(selected_ID=chosen,positive_eligible=bool(eligible),positive_supported=support,
                    pure_wrong_activation=false,known_wrong_owner_activation=wrong_owner,pure_local_support=local,
                    logits=logits[ri].cpu().tolist(),WHO_head_selected_ID=who_chosen,
                    WHO_head_logits=who_logits[ri].cpu().tolist() if who_logits is not None else None)
            self.stream.write(json.dumps(dict(key=list(key),row=row,task=task,GT_OFFLINE_ONLY=gt,
                actual_candidates=refs,high_risk=record['high_risk'],variants=paired),allow_nan=False)+'\n')

    def after(self,**d):
        key=(self.video,d['frame'],d['view']);inst=d['instances'][-1]
        expected=[r['committed_ID'] for r in self.cursor.at(key) if r.get('record_type')=='COMMIT']
        assert inst.track_ids.tolist()==expected,(key,'shadow trials changed native commitments')
        self.append(inst,d['frame'],d['view']);self.actual_payloads+=1
        self.raw_vectors_max=max(self.raw_vectors_max,sum(len(g) for g in d['galleries'].values()))


def main(video):
    protect();storage_guard();assert video in TRAIN+DEV
    assert read(REPORTS/'PHASE16_P0_GO_NO_GO.json')['P1_frozen_prototype_trial_qualified']
    torch.set_num_threads(1);torch.manual_seed(20261009);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    policy,checkpoint,training=load_policy('v3');values,frames,reader=cache_inputs(video)
    p0=OUT/'P0_r2/full/v3'/f'video{video:02d}/RESULT.json';baseline=read(p0)['cases'][0]['queries']
    assert sha(baseline['path'])==baseline['SHA256'];cursor=TraceCursor(baseline['path'])
    model=build_tracker(video,policy,react_learned=False);executor=attach(model)
    out=OUT/'P1'/f'video{video:02d}';out.mkdir(parents=True,exist_ok=True);assert not (out/'RESULT.json').exists()
    source=binding(checkpoints=[checkpoint],inputs=[training,ref(p0),ref(REPORTS/'P1_PROTOCOL.json')],
        native_state=dict(kind='actual original frozen whole native query/commit trace',trace=baseline,restorable_start_snapshot_captured=False),
        evaluator='paired four-slot all-query frozen model/prototype diagnostics; original native outputs unchanged',
        scope='strict TRAIN eligibility, TRAIN14 and reused DEV isolated diagnostics; zero training')
    source['perception_input_provenance']=perception_provenance(video)
    source['concurrent_GPU_processes_at_start']=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True)
    start=time.monotonic();last=[0.]
    with gzip.open(out/'PAIRED_READERS.jsonl.gz','wt') as stream:
        audit=PrototypeAudit(video,reader,policy,cursor,stream)
        model.jev_native_prefix_observer=audit.native_before;executor.observer=audit.before
        def after(**d):
            audit.after(**d)
            if time.monotonic()-last[0]>=15:
                save(out/'PROGRESS.json',dict(status='FROZEN_PAIRED_READER_AUDIT',key=[video,d['frame'],d['view']],frames=frames,
                    original_actual_payloads=audit.actual_payloads,seconds=time.monotonic()-start));stream.flush();last[0]=time.monotonic()
        executor.commit_observer=after
        with torch.no_grad():raw,_=run(model,values,frames)
    cursor.stream.close();assert audit.actual_payloads==2*frames-1
    memory={v:dict(IDs=len(m.identities),max_vectors_per_ID=max((m.stored_vectors(i) for i in m.identities),default=0),
        stored_vectors=sum(m.stored_vectors(i) for i in m.identities),persisted_feature_MiB=sum(m.stored_vectors(i) for i in m.identities)*1024*4/2**20,
        state_SHA256=fingerprint(m.state_dict())) for v,m in audit.views.items()}
    assert all(v['max_vectors_per_ID']<=16 for v in memory.values())
    save(out/'RESULT.json',dict(status='COMPLETE',binding=source,video=video,frames=frames,
        counts={v:dict(c) for v,c in audit.counts.items()},clusters={v:[list(k) for k in sorted(c)] for v,c in audit.clusters.items()},
        read_latency_ms={v:quantile(t) for v,t in audit.read_ms.items()},forward_latency_ms={v:quantile(t) for v,t in audit.forward_ms.items()},
        bounded_memory=memory,original_raw_Gallery_max_vectors=audit.raw_vectors_max,
        original_raw_Gallery_visual_MiB=audit.raw_vectors_max*1024*4/2**20,
        final_original_identity_and_commitment_memory_SHA256=fingerprint(executor.memory.state_dict()),
        paired_reader_trace=ref(out/'PAIRED_READERS.jsonl.gz'),every_original_logit_and_committed_ID_exact_vs_P0=True,
        shadow_decisions_not_executed_and_not_native_benefit=True,actual_training_updates=0,
        joint_policy_scores_include_availability_trust_Q4_reliability=True,WHO_head_diagnosed_separately=True,
        cache_audit_is_not_full_image_FPS=True,strict_train_gate_eligible=video in TRAIN and video!=14,seconds=time.monotonic()-start))
    save(out/'PROGRESS.json',dict(status='COMPLETE',frames=frames));print('PHASE16_P1_COMPLETE',video,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--video',type=int,required=True);main(p.parse_args().video)
