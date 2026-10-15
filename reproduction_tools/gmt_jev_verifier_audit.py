"""Full approved TRAIN original-GMT states, frozen-MLP counterfactual plans."""
import collections,gzip,heapq,sys,torch
import gmt_jev_opportunity as opportunity
from gmt_jev_opportunity import OfflineCertificates
from gmt_jev_runtime import metadata,seed
from gmt_jev_learned_checkpoint import load_actor
from gmt_jev_verifier_common import *
from gmt_jev_verifier_labels import history_label,paired_plan,dependency_components
from gmt_jev_verifier_q1_diagnostic import run_diagnostic

class History(OfflineCertificates):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.view_votes=collections.defaultdict(lambda:collections.defaultdict(collections.Counter))
    def update(self,inst,frame,view):
        for t,g in zip(inst.track_ids.tolist(),self.labels(inst,frame,view)):
            self.total[t]+=1
            if g is not None:self.votes[t][g]+=1;self.view_votes[t][view][g]+=1
            if t not in self.owner and self.votes[t]:
                person,n=self.votes[t].most_common(1)[0]
                if n>=3 and n/self.total[t]>=.8:self.owner[t]=person

def main():
    assert read(REPORTS/'GOAL.json')['budget']['new_GPU_wall_hours_max']==4
    seed();actor,checkpoint=load_actor('mlp');counts=collections.Counter();cases=[];events=[];examples=[]
    q1groups=set();q1ids=collections.defaultdict(set);heaps={k:[] for k in ['RELIABLE','CONFLICTED','UNKNOWN']}
    serial=0;buffer_bytes=0;group_bytes=0;peak=0;diagnostic_seen=collections.Counter();q2_probes={}
    for video in [12,13,14]:
        _,frames,images=metadata(video);local=collections.Counter();holder={};before_result=[]
        class CapturedHistory(History):
            def __init__(self,*a,**kw):super().__init__(*a,**kw);holder['history']=self
        original_class=opportunity.OfflineCertificates;original_run=opportunity.run
        def compare_run(*a,**kw):
            from run_jev_phase10_closed_loop import raw_predictions
            out=original_run(*a,**kw)
            reference=legacy.read(legacy.REPORTS/f'TRAIN_MATCHED_original_VIDEO{video:02d}.json')['inference']['predictions']['RAW']
            assert sha(reference['path'])==reference['SHA256']
            with gzip.open(reference['path'],'rt') as f:old=json.load(f)
            assert raw_predictions(out[0],images)==old,'P0 observer changed original complete native predictions'
            before_result.append(dict(status='PASS_EXACT_COMPLETE_RAW_RECORDS',reference=reference,records=len(old)))
            return out
        def observe(s):
            nonlocal serial,buffer_bytes,group_bytes,peak
            history=holder['history'];refs=s['refs'];f,v=s['key'][1:];heldout=s['audit_block']
            device=next(actor.parameters()).device;x=s['features'].to(device);raw=s['raw_scores'].to(device);lengths=s['lengths'].to(device)
            with torch.no_grad():
                gate,delta=actor.head(x);corrected=raw+(.05*gate*delta)*lengths[None]
            base=opportunity.native_assignment(raw,lengths);mlp=opportunity.native_assignment(corrected,lengths)
            assert base==s['baseline_accepted_cols']
            assert all(s['native_ids'][r]==refs[c] for r,c in enumerate(base) if c>=0)
            plan=paired_plan(base,mlp,refs,s['targets'].tolist(),s['categories'],s['GT_labels_OFFLINE_ONLY'],s['owners'])
            local['plans']+=1;local['plan_'+plan['category']]+=1;local.update({'row_'+k:n for k,n in collections.Counter(s['categories']).items()})
            local['rows']+=len(base);local['changed_rows']+=len(plan['changed']);local.update(plan.get('unknown_reasons',{}))
            if plan['label'] is not None:
                event=dict(video=video,frame=f,view=v,heldout=heldout,people=plan['people'],action=plan['category'])
                events.append(event);local[('HELDOUT_' if heldout else 'TRAIN_')+plan['category']]+=1
                if not heldout and plan['label'] not in q2_probes:
                    q2_probes[plan['label']]=s['features'][plan['changed']].mean((0,1)).clone()
            if plan['changed'] and len(examples)<64:
                examples.append(dict(key=s['key'],**plan,base_cols=base,mlp_cols=mlp,refs=refs,
                    current_people_OFFLINE_ONLY=s['GT_labels_OFFLINE_ONLY'],targets=s['targets'].tolist()))
            compatible=collections.Counter()
            for t,votes in history.votes.items():
                for g,n in votes.items():
                    if n/history.total[t]>=.8:compatible[g]+=1
            if not refs:return
            features=s['features'].mean(0)
            for col,t in enumerate(refs):
                owner=history.owner.get(t);label=history_label(history.total[t],history.votes[t],owner,compatible.get(owner,0))
                local['Q1_'+label]+=1;q1ids[label].add((video,t))
                view_owners=[]
                for votes in history.view_votes[t].values():
                    if votes:
                        g,n=votes.most_common(1)[0]
                        if n>=3 and n/sum(votes.values())>=.8:view_owners.append(g)
                local['Q1_cross_camera_certified_owner_conflict']+=len(set(view_owners))>1
                group=(video,t,v,f//64,label)
                if group in q1groups:continue
                q1groups.add(group)
                group_bytes+=sys.getsizeof(group)+sum(sys.getsizeof(a) for a in group)
                priority=int(hashlib.sha256(repr(group).encode()).hexdigest(),16)
                # Q1 uses ONLY committed-history channels; current matching evidence is masked.
                feature=torch.zeros(24);keep=[6,7,8,9,10,15,18,20,23];feature[keep]=features[col,keep]
                item=dict(video=video,identity=t,frame=f,view=v,label=label,heldout=heldout,x=feature)
                weight=sys.getsizeof(item)+sys.getsizeof(feature)+feature.numel()*feature.element_size()+512
                part=heaps[label];serial+=1
                if len(part)<2000:heapq.heappush(part,(-priority,serial,weight,item));buffer_bytes+=weight
                elif priority < -part[0][0]:
                    old=heapq.heapreplace(part,(-priority,serial,weight,item));buffer_bytes+=weight-old[2]
                peak=max(peak,buffer_bytes+group_bytes+sys.getsizeof(q1groups)+sum(sys.getsizeof(p)+len(p)*160 for p in heaps.values()))
                assert peak<32*2**20,'Training buffer and grouping metadata exceed 32MiB'
            if f%64==0:
                save(OUT/'P0_PROGRESS.json',dict(status='PAIRED_GMT_NATIVE_STATE_AUDIT',video=video,frame=f,frames=frames,counts=dict(local),buffer_bytes=peak))
        opportunity.OfflineCertificates=CapturedHistory;opportunity.run=compare_run
        try:
            samples,result=opportunity.collect(video,frames=frames,records=False,report=False,diagnostic_sample_observer=observe)
        finally:opportunity.OfflineCertificates=original_class;opportunity.run=original_run
        assert not samples and result['frames']==frames and before_result
        counts.update(local);case=dict(video=video,frames=frames,counts=dict(local),original_prediction_parity=before_result[0],source_result_counts=result['counts'])
        cases.append(case);save(REPORTS/f'PAIRED_PLAN_VIDEO{video:02d}.json',case)
        print('V2_PAIRED_VIDEO_COMPLETE',video,dict(local),flush=True)
    support={}
    for action in ['KEEP_GMT','APPLY_MLP']:
        train=[e for e in events if e['action']==action and not e['heldout']]
        support[action]=dict(plans=len(train),person_dependency_components=dependency_components(train),
           unique_video_person_groups=len({(e['video'],p) for e in train for p in e['people']}),videos=sorted({e['video'] for e in train}),
           temporal_blocks=len({(e['video'],e['view'],e['frame']//64) for e in train}))
    eligible=all(s['plans']>=8 and s['person_dependency_components']>=8 and len(s['videos'])>=2 for s in support.values())
    summary=dict(status='COMPLETE_ALL_APPROVED_TRAIN_PAIRED_PLAN_AUDIT',counts=dict(counts),cases=cases,
        Q2_training_eligible=eligible,Q2_support=support,Q1_unique_native_ID_groups={k:len(v) for k,v in q1ids.items()},
        Q1_unique_time_camera_ID_label_groups=len(q1groups),buffer_accounted_peak_bytes=peak,
        examples=examples,examples_truncated_at=64,events=events,
        same_state='Two complete Hungarian+native-threshold accepted plans on one immutable original-GMT pre-commit score/candidate/support packet; actual original commits checked. No mixed row plan or mutated-trajectory numeric-ID comparison.',
        Bank_birth='Any changed rejected row makes whole-plan certification UNKNOWN; no unexecuted Bank or newborn outcome assumed.',
        history='Only earlier committed observations; current commit not yet inserted in offline votes when sample observer runs. Q1 input history channels [6,7,8,9,10,15,18,20,23], no current/future GT input.',
        temporal_dependency='Original time buckets carried through history certification. Heldout matrices excluded from Q2 training counts; this is NOT strict label isolation or independent validation.',
        eligibility='Per action >=8 certified plans, >=8 person-connected dependency components, >=2 TRAIN videos; minimum engineering smoke qualification only, not statistical sufficiency.',
        inference_gradient_updates=0,frozen_MLP_checkpoint=checkpoint,scientific_success=False,source=binding())
    probes={}
    for label,x in q2_probes.items():
        head=torch.nn.Linear(24,2);loss=torch.nn.functional.cross_entropy(head(x[None]),torch.tensor([label]));loss.backward()
        norm=sum(float(p.grad.square().sum()) for p in head.parameters())**.5
        assert norm>0
        probes[str(label)]=dict(gradient_norm=norm,parameter_updates=0,scope='Real certified whole-plan label, CPU linear autograd smoke; not trained typed verifier or evidence of two-class support.')
    summary['Q2_real_label_gradient_smoke']=probes
    save(REPORTS/'PAIRED_PLAN_OPPORTUNITY.json',summary)
    save(REPORTS/'DECISION_LABEL_ELIGIBILITY.json',dict(status='ELIGIBLE_FOR_BOUNDED_SMOKE' if eligible else 'INSUFFICIENT_INDEPENDENT_Q2_SUPERVISION',support=support,source=ref(REPORTS/'PAIRED_PLAN_OPPORTUNITY.json')))
    save(REPORTS/'CURRENT_IDENTITY_CONFLICT_AUDIT.json',dict(Q1_unique_native_ID_groups=summary['Q1_unique_native_ID_groups'],counts={k:v for k,v in counts.items() if k.startswith('Q1_')},history_definition=summary['history']))
    save(REPORTS/'SUPERVISION_ELIGIBILITY.json',dict(Q2_eligible=eligible,Q2_support=support,Q1_support=summary['Q1_unique_native_ID_groups'],evidence=ref(REPORTS/'PAIRED_PLAN_OPPORTUNITY.json'),scientific_success=False))
    if not eligible:
        # Existing retained audits contain no frozen-MLP pre-state features. Review,
        # cite them, and do not pretend old rule E1 witnesses are APPLY labels.
        diagnostic_seen.update({k:read(legacy.REPORTS/f'FULL_TRAIN_NATIVE_RULE_VIDEO{k:02d}.json')['paired_commit_counts'].get('N01',0) for k in [12,13,14]})
        save(REPORTS/'HISTORICAL_EVENT_REAUDIT.json',dict(status='NO_ADDITIONAL_CERTIFIED_MLP_PLAN_LABELS_IN_RETAINED_RULE_AUDITS',rule_N01_by_video=dict(diagnostic_seen),
           evidence=[ref(legacy.REPORTS/f'FULL_TRAIN_NATIVE_RULE_VIDEO{k:02d}.json') for k in [12,13,14]],
           reason='Rule native events and capped ORACLE witnesses are not frozen-MLP paired plan supervision; no persisted full decision feature corpus exists. No relabeling of UNKNOWN.'))
        run_diagnostic([t[3] for part in heaps.values() for t in part],peak)
    print('V2_P0_COMPLETE',eligible,support,flush=True)
if __name__=='__main__':main()
