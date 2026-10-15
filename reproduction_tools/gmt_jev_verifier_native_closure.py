"""Close rejected MATCH plans through actual Bank/birth calls, not guesses."""
import argparse,collections,gzip,inspect,torch
import gmt_jev_opportunity as opportunity
import gtr.modeling.gmt_jev as package
from gmt_jev_runtime import metadata
from gmt_jev_learned_checkpoint import load_actor
from gmt_jev_verifier_common import *
from gmt_jev_verifier_labels import dependency_components
from gtr.modeling.gmt_jev.native_plan_replay import replay,call_digest,RestorationProbe

def classify(sample,base,mlp):
    initial=sample['id_count_before'];refs=sample['refs'];history=sample['past_history']
    semantic=lambda t:('NEW',) if t>initial else ('EXISTING',t)
    changed=[r for r,(a,b) in enumerate(zip(base,mlp)) if semantic(a)!=semantic(b)]
    if not changed:return dict(category='EQUIVALENT_NATIVE_PLAN',label=None,changed=[],people=[])
    local=[];reasons=collections.Counter();people=set()
    for r in changed:
        person=sample['GT_labels_OFFLINE_ONLY'][r];target=sample['global_targets'][r]
        if person is not None:people.add(person)
        for t in [base[r],mlp[r]]:
            if history.get(t) is not None:people.add(history[t])
        if target is None:reasons['UNKNOWN_CURRENT_OR_ALIASED_OR_UNCERTIFIED_GLOBAL_TARGET']+=1;continue
        correct=[]
        for identity in [base[r],mlp[r]]:
            if identity>initial:correct.append(False) # Certified existing target: new ID is a fragment.
            elif identity==target:correct.append(True)
            elif history.get(identity) is not None and history[identity]!=person:correct.append(False)
            else:correct.append(None)
        if None in correct:reasons['UNKNOWN_FINAL_OWNER_OR_COMPATIBLE_ALIAS']+=1
        else:local.append(correct)
    if reasons:category='UNKNOWN_NATIVE_PLAN';label=None
    elif all(a for a,b in local) and any(not b for a,b in local):category='KEEP_GMT';label=0
    elif all(b for a,b in local) and any(not a for a,b in local):category='APPLY_MLP';label=1
    elif all(a and b for a,b in local):category='BOTH_CORRECT';label=None
    elif not any(a or b for a,b in local):category='BOTH_WRONG';label=None
    else:category='MIXED_NATIVE_TRADEOFF_NO_LABEL';label=None
    return dict(category=category,label=label,changed=changed,people=sorted(people),unknown_reasons=dict(reasons))

def main(videos,stop):
    if stop is None:
        contract=read(REPORTS/'NATIVE_PLAN_CLOSURE_PREFIX_CONTRACT.json')
        assert contract['status']=='COMPLETE_NATIVE_PLAN_CLOSURE_PREFIX_CONTRACT' and contract['counts']['actual_paired_native_replays']>0
    else:assert 0<stop<=256
    original_correction=package.IdentityCorrection
    actor,checkpoint=load_actor('mlp');all_counts=collections.Counter();cases=[];events=[];examples=[];contract_cases=[];probes={}
    for video in videos:
        _,total,images=metadata(video);frames=total if stop is None else min(total,stop)
        holder={};pending={};counts=collections.Counter();parity=[]
        original_model=opportunity.model;original_run=opportunity.run
        original_certificates=opportunity.OfflineCertificates
        class Certificates(original_certificates):
            def __init__(self,*a,**kw):super().__init__(*a,**kw);holder['history']=self
        class Observed(original_correction):
            def forward(self,*a,**kw):
                out=super().forward(*a,**kw);p=self.last
                raw=p['base_scores'];lengths=p['lengths'];refs=p['ids'].tolist()
                with torch.no_grad():gate,delta=actor.head(p['features']);new=raw+(.05*gate*delta)*lengths[None]
                b=opportunity.native_assignment(raw,lengths);c=opportunity.native_assignment(new,lengths)
                counts['plans']+=1
                if b==c:counts['MATCH_EQUIVALENT_NO_REPLAY']+=1;return out
                model=holder['model'];fn,args,kwargs=holder['call']
                if not contract_cases:
                    try:replay(model,fn,args,kwargs,raw,new,p['ids'],lengths,probe_exception=True)
                    except RestorationProbe:counts['exception_restoration_probe_pass']+=1
                    else:raise AssertionError('Restoration test did not raise')
                base=replay(model,fn,args,kwargs,raw,raw,p['ids'],lengths)
                alternative=replay(model,fn,args,kwargs,raw,new,p['ids'],lengths)
                assert base['pre_SHA256']==alternative['pre_SHA256']
                key=(p['frame'],p['view']);pending[key]=dict(base=base,mlp=alternative)
                counts['actual_paired_native_replays']+=1
                if len(contract_cases)<16:contract_cases.append(dict(key=[video,*key],same_pre_SHA256=base['pre_SHA256'],score_interface_exact=True,restoration_exact=True))
                return out
        def captured_model(*a,**kw):
            out=original_model(*a,**kw);m=out[0];holder['model']=m
            assert not m.jev_enabled and m.jev_policy is None and m.jev_candidate_policy is None
            def wrap(fn):
                def call(*args,**kwargs):
                    holder['call']=(fn,args,kwargs);result=fn(*args,**kwargs)
                    # The observer records the key, avoiding assumptions about bootstrap view.
                    proof=holder.pop('verify_actual',None)
                    if proof is not None:assert call_digest(m,args,kwargs,result)==proof,'Cloned GMT post-state differs from actual native execution'
                    holder.pop('call',None);return result
                return call
            m.run_first_tracker_plus=wrap(m.run_first_tracker_plus);m.run_global_tracker_plus=wrap(m.run_global_tracker_plus)
            return out
        def compare_run(*a,**kw):
            from run_jev_phase10_closed_loop import raw_predictions
            out=original_run(*a,**kw)
            reference=legacy.read(legacy.REPORTS/f'TRAIN_MATCHED_original_VIDEO{video:02d}.json')['inference']['predictions']['RAW']
            assert sha(reference['path'])==reference['SHA256']
            with gzip.open(reference['path'],'rt') as f:old=json.load(f)
            image_ids={i['id'] for (frame,view),i in images.items() if frame<frames}
            assert raw_predictions(out[0],images)==[r for r in old if r['image_id'] in image_ids]
            parity.append(dict(status='PASS_EXACT_RAW_RECORDS',frames=frames,reference=reference));return out
        def observed(s):
            key=tuple(s['key'][1:]);pair=pending.pop(key,None)
            if pair is None:return
            assert s['native_ids']==pair['base']['native_ids']
            holder['verify_actual']=pair['base']['post_SHA256']
            history=holder['history'];compatible=collections.defaultdict(list)
            for t,votes in history.votes.items():
                for g,n in votes.items():
                    if n/history.total[t]>=.8:compatible[g].append(t)
            targets=[]
            for person in s['GT_labels_OFFLINE_ONLY']:
                ids=compatible.get(person,[]) if person is not None else []
                targets.append(ids[0] if len(ids)==1 and history.clean(ids[0])==person else None)
            s=dict(s,past_history={t:history.clean(t) for t in history.total},global_targets=targets)
            choice=classify(s,pair['base']['native_ids'],pair['mlp']['native_ids'])
            counts['plan_'+choice['category']]+=1;counts.update(choice.get('unknown_reasons',{}))
            counts['cloned_GMT_ids_equal_actual']+=1
            if choice['label'] is not None:
                events.append(dict(video=video,frame=s['key'][1],view=s['key'][2],heldout=s['audit_block'],people=choice['people'],action=choice['category']))
                if not s['audit_block'] and choice['label'] not in probes:
                    probes[choice['label']]=s['features'][choice['changed']].mean((0,1)).clone()
            if len(examples)<64:examples.append(dict(key=s['key'],**choice,actual_GMT_ids=pair['base']['native_ids'],actual_MLP_clone_ids=pair['mlp']['native_ids'],current_GT_OFFLINE_ONLY=s['GT_labels_OFFLINE_ONLY'],global_targets=targets))
            if s['key'][1]%32==0:save(OUT/'NATIVE_CLOSURE_PROGRESS.json',dict(video=video,frame=s['key'][1],frames=frames,counts=dict(counts)))
        opportunity.model=captured_model;opportunity.run=compare_run;opportunity.OfflineCertificates=Certificates;package.IdentityCorrection=Observed
        try:
            _,result=opportunity.collect(video,frames=frames,records=False,report=False,diagnostic_sample_observer=observed)
        finally:
            opportunity.model=original_model;opportunity.run=original_run;opportunity.OfflineCertificates=original_certificates;package.IdentityCorrection=original_correction
        assert not pending and parity
        case=dict(video=video,frames=frames,counts=dict(counts),raw_parity=parity[0]);cases.append(case);all_counts.update(counts)
        save(REPORTS/f'NATIVE_CLOSURE_{"PREFIX" if stop else "FULL"}_VIDEO{video:02d}.json',case)
    support={}
    for action in ['KEEP_GMT','APPLY_MLP']:
        train=[e for e in events if e['action']==action and not e['heldout']]
        support[action]=dict(plans=len(train),person_dependency_components=dependency_components(train),unique_video_person_groups=len({(e['video'],p) for e in train for p in e['people']}),videos=sorted({e['video'] for e in train}),temporal_blocks=len({(e['video'],e['view'],e['frame']//64) for e in train}))
    eligible=all(s['plans']>=8 and s['person_dependency_components']>=8 and len(s['videos'])>=2 for s in support.values())
    report=dict(status='COMPLETE_NATIVE_PLAN_CLOSURE_PREFIX_CONTRACT' if stop else 'COMPLETE_NATIVE_PLAN_CLOSURE_ALL_APPROVED_TRAIN',
        cases=cases,counts=dict(all_counts),events=events,examples=examples,Q2_support=support,Q2_training_eligible=eligible,
        contract_examples=contract_cases,no_new_gradient_updates=True,frozen_MLP_checkpoint=checkpoint,scientific_success=False,
        scope='Full native tracker call replay, including Bank, birth, Gallery WRITE and global Bank sets. Shared actual pre-state and exact restoration; no hand-edited IDs or legacy JEV enablement. Local same-state causal outcomes only, not full-video intervention propagation.',source=binding())
    gradient_checks={}
    for label,x in probes.items():
        head=torch.nn.Linear(24,2);loss=torch.nn.functional.cross_entropy(head(x[None]),torch.tensor([label]));loss.backward()
        norm=sum(float(p.grad.square().sum()) for p in head.parameters())**.5;assert norm>0
        gradient_checks[str(label)]=dict(gradient_norm=norm,parameter_updates=0,scope='CPU actual certified native-plan label smoke; not a trained Typed verifier')
    report['Q2_real_label_gradient_smoke']=gradient_checks
    assert all_counts['actual_paired_native_replays']>0
    save(REPORTS/('NATIVE_PLAN_CLOSURE_PREFIX_CONTRACT.json' if stop else 'NATIVE_PAIRED_PLAN_OPPORTUNITY.json'),report)
    print('NATIVE_PLAN_CLOSURE_COMPLETE',stop,support,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--videos',type=int,nargs='+',default=[12,13,14]);p.add_argument('--stop',type=int);a=p.parse_args()
    assert all(v in [12,13,14] for v in a.videos);main(a.videos,a.stop)
