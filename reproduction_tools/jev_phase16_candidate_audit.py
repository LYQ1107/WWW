"""All-query actual Gallery/Bank audit; annotation use is observer-only."""
import argparse
import collections
import gzip
import time
from jev_phase16_common import *
import torch
from gtr.modeling.jev_phase16.evidence_ledger import EvidenceLedger
from gtr.modeling.jev_phase15.native_commit_adapter import attach
from gtr.modeling.jev_stage2.assignment import lawful_choice
from gtr.modeling.jev_native_state import fingerprint
from jev_phase13_runtime import build_tracker,cache_inputs,run
from jev_phase15_pilot_native import converted_prefix
from jev_phase14_artifacts import load_dense
from run_jev_phase10_closed_loop import raw_predictions

class CandidateAudit:
    def __init__(self,video,reader,model,stream,prefix=None):
        self.video=video;self.labels=duplicate_masked_labels(video,reader);self.model=model
        self.executor=model.jev_stage2_executor;self.stream=stream
        self.ledger=EvidenceLedger() if prefix is None else EvidenceLedger.from_prefix(prefix)
        self.counts=collections.Counter();self.task_counts=collections.defaultdict(collections.Counter)
        self.flags=collections.Counter();self.primary=collections.Counter();self.clusters=collections.defaultdict(set)
        self.owner={};self.gt_cache={};self.last_records={};self.commit_rows=[];self.pending={}
        self.native_context=None
        if prefix is not None:self.ledger.verify_raw_galleries(prefix['galleries'])

    def labels_for(self,identity):
        rows=self.ledger.records[identity];old=self.gt_cache.get(identity)
        if old is None or old[0]!=len(rows):
            targets=[self.labels.current(*r['key'][:2])[r['key'][2]] for r in rows]
            votes=collections.Counter(gt for gt in targets if gt is not None)
            if identity not in self.owner:
                past=collections.Counter()
                for observed_count,gt in enumerate(targets,1):
                    if gt is not None:past[gt]+=1
                    if len(past)>1:break  # No earlier pure anchor existed.
                    if sum(past.values())>=3 and sum(past.values())/observed_count>=.8:
                        self.owner[identity]=next(iter(past));break
            encoded=torch.tensor([-1 if gt is None else gt for gt in targets],device=rows[0]['feature'].device,dtype=torch.long)
            old=(len(rows),targets,votes,encoded);self.gt_cache[identity]=old
        return old

    def native_before(self,**d):
        self.native_context=d
        if not self.ledger.records and d['first']:
            other=1-d['view'];self.ledger.append_instance(d['instances'][other],0,other)
        if d['frame']%128==0:self.ledger.verify_raw_galleries(d['galleries'])

    def before(self,**d):
        from gtr.modeling.meta_arch.gtr_rcnn import old_reids,poss_ids
        c=d['context'];key=(c['frame'],c['view']);task=int(d['task']);refs=list(map(int,d['refs']))
        rows=c.get('rows',list(range(len(d['logits']))));targets=[self.labels.current(*key)[r] for r in rows]
        selected=lawful_choice(d['logits'],d['batch']['legal'][0])
        dv=d['batch']['detection_visual'][0];q=len(rows);allrefs=sorted(self.ledger.records)
        if not q:return
        gt_tensor=torch.tensor([-1 if gt is None else gt for gt in targets],device=dv.device)
        raw_scores={};summary_scores={};evidence={};pure_correct=[[] for _ in rows]
        meta=self.executor.memory.meta;galleries=self.native_context['galleries']
        bank=set(old_reids.old_reids[0].track_ids.tolist()) if old_reids.old_reids else set()
        for identity in allrefs:
            n,labels,votes,encoded=self.labels_for(identity);raw,norm=self.ledger.matrix(identity)
            score=dv@norm.T;correct=(encoded[None,:]==gt_tensor[:,None])&(gt_tensor[:,None]>=0)
            foreign=(encoded[None,:]>=0)&(~correct)&(gt_tensor[:,None]>=0)
            positive=score.masked_fill(~correct,-2).max(-1).values
            negative=score.masked_fill(~foreign,-2).max(-1).values
            maximum=score.max(-1).values
            raw_scores[identity]=torch.stack([positive,negative,maximum],-1).cpu().tolist()
            vals=self.executor.memory.history_values(identity,galleries[identity],meta[identity],key[1])
            present=[v for v in vals if v is not None]
            summary_scores[identity]=(dv@torch.stack(present).T).max(-1).values.cpu().tolist()
            pure=sum(votes.values())>=3 and sum(votes.values())/n>=.8 and len(votes)==1
            evidence[identity]=dict(votes=votes,unknown=n-sum(votes.values()),pure=pure,owner=self.owner.get(identity))
            if pure:
                gt=next(iter(votes))
                for ri,target in enumerate(targets):
                    if target==gt:pure_correct[ri].append(identity)
        legal_rows=d['batch']['legal'][0].cpu().tolist()
        raw_known_total=sum(sum(e['votes'].values()) for e in evidence.values())
        raw_unknown_total=sum(e['unknown'] for e in evidence.values())
        for ri,(row,target,col) in enumerate(zip(rows,targets,selected)):
            available={identity for identity,legal in zip(refs,legal_rows[ri]) if legal}
            chosen=None if col<0 else refs[col];pure=pure_correct[ri]
            supported=[identity for identity in pure if identity in available]
            known_chosen=chosen is not None and evidence[chosen]['pure']
            selected_correct=chosen in pure
            content=[identity for identity,e in evidence.items() if target is not None and target in e['votes']]
            qualified=[identity for identity in content if evidence[identity]['votes'][target]>=3]
            owners=[identity for identity in qualified if evidence[identity]['owner']==target]
            mixed=[identity for identity in content if len(evidence[identity]['votes'])>1]
            hidden=[identity for identity in qualified if raw_scores[identity][ri][0]>=.75 and
                (summary_scores[identity][ri]<.75 or raw_scores[identity][ri][0]-raw_scores[identity][ri][1]-
                 (summary_scores[identity][ri]-raw_scores[identity][ri][1])>=.02)]
            owner_hidden=[identity for identity in hidden if identity in owners]
            raw_max=max((raw_scores[i][ri][0] for i in qualified),default=None)
            summary_max=max((summary_scores[i][ri] for i in qualified),default=None)
            pending_bank=[identity for identity in pure if identity not in available and
                (identity in bank or identity in poss_ids.poss_ids and len(galleries[identity])>=self.model.bank_size)]
            legal_omitted=[identity for identity in pure if identity not in available and identity not in pending_bank]
            flags=dict(A_ABSENT_FROM_ALL_HISTORY=target is not None and not content,
                B_PRESENT_IN_RAW_GALLERY=bool(hidden),C_PRESENT_IN_STALE_BANK=bool(pending_bank),
                D_CANDIDATE_ELIGIBILITY_FAILURE=bool(legal_omitted),
                E_CANDIDATE_PRESENT_BUT_WRONG_SELECTION=bool(supported) and not selected_correct,
                F_HISTORY_CONTAMINATED=bool(mixed),
                G_AMBIGUOUS=target is None or not pure and (bool(content) or raw_unknown_total>0))
            highrisk=not selected_correct or not supported or bool(mixed) and chosen in mixed
            primary=next((name for name in ['E_CANDIDATE_PRESENT_BUT_WRONG_SELECTION','C_PRESENT_IN_STALE_BANK',
                'D_CANDIDATE_ELIGIBILITY_FAILURE','F_HISTORY_CONTAMINATED','B_PRESENT_IN_RAW_GALLERY',
                'A_ABSENT_FROM_ALL_HISTORY','G_AMBIGUOUS'] if flags[name]),'CERTIFIED_CORRECT_SELECTION')
            kind='MATCH' if task==0 else 'REACT'
            self.counts['all_queries']+=1;self.counts['high_risk_queries']+=highrisk
            self.counts['high_risk_GT_known_queries']+=highrisk and target is not None
            self.counts['current_GT_known']+=target is not None
            self.counts['observed_correct_content_exists']+=bool(content)
            self.counts['qualified_raw_content_exists']+=bool(qualified)
            self.counts['anchored_correct_owner_exists']+=bool(owners)
            self.counts['legal_pure_correct_available']+=bool(supported)
            self.counts['selected_certified_correct']+=selected_correct
            self.counts['selected_certified_wrong']+=known_chosen and not selected_correct and target is not None
            self.counts['selected_UNKNOWN_history']+=chosen is not None and not known_chosen and target is not None
            self.counts['selected_globally_mixed']+=chosen is not None and len(evidence[chosen]['votes'])>1 and target is not None
            self.counts['summary_hidden_queries']+=bool(hidden)
            self.counts['owner_anchored_summary_hidden_queries']+=bool(owner_hidden)
            self.counts['raw_known_observations_at_query']+=raw_known_total
            self.counts['raw_UNKNOWN_observations_at_query']+=raw_unknown_total
            self.task_counts[kind]['queries']+=1
            self.task_counts[kind]['high_risk']+=highrisk
            if highrisk:
                self.primary[primary]+=1
                for name,value in flags.items():self.flags[name]+=value
            cluster=(self.video,target,key[1],key[0]//64)
            if target is not None and owner_hidden:
                self.clusters['owner_anchored_summary_loss'].add(cluster)
            if highrisk and target is not None:self.clusters['high_risk_known'].add(cluster)
            record=dict(key=[self.video,*key],row=row,task=kind,current_GT_OFFLINE_ONLY=target,
                selected_ID=chosen,selected_certified_correct=selected_correct,high_risk=highrisk,
                actual_candidates=refs,legal_pure_correct_IDs=supported,raw_target_content_IDs=content,
                qualified_raw_target_IDs=qualified,anchored_correct_owner_IDs=owners,
                mixed_target_content_IDs=mixed,summary_hidden_IDs=hidden,owner_anchored_summary_hidden_IDs=owner_hidden,
                pending_native_Bank_IDs=pending_bank,ineligible_pure_IDs=legal_omitted,
                raw_best_target_cosine=raw_max,summary_best_target_ID_cosine=summary_max,
                raw_and_summary_scores={str(i):dict(raw_target=raw_scores[i][ri][0],
                    raw_foreign=raw_scores[i][ri][1],raw_max=raw_scores[i][ri][2],four_summary=summary_scores[i][ri],
                    owner_GT_OFFLINE_ONLY=evidence[i]['owner'],target_count=evidence[i]['votes'].get(target,0),
                    known_GT_count=len(evidence[i]['votes'])) for i in allrefs},
                raw_unknown_count=raw_unknown_total,flags=flags,primary=primary,
                absence_scope='No observed target content in past raw Gallery; UNKNOWN past observations may hide additional identity evidence',
                correct_person_content_is_not_correct_Global_ID=True)
            self.stream.write(json.dumps(record,allow_nan=False)+'\n')
            self.pending[key,row]=record

    def after(self,**d):
        key=(d['frame'],d['view']);inst=d['instances'][-1]
        for row,event in enumerate(d['events']):
            record=self.pending.get((key,row))
            if record is not None:
                record=dict(key=[self.video,*key],row=row,action=event['action'],committed_ID=event['identity'],
                    high_risk=record['high_risk'],primary=record['primary'],flags=record['flags'])
                self.stream.write(json.dumps(dict(record_type='COMMIT',**record))+'\n')
                self.last_records[key,row]=record;self.counts[event['action']]+=1
        self.ledger.append_instance(inst,*key)
        for item in [k for k in self.pending if k[0]<=key]:del self.pending[item]
        for item in [k for k in self.last_records if k[0][0]<key[0]-1]:del self.last_records[item]
        self.commit_rows.append(dict(key=list(key),ids=inst.track_ids.cpu().tolist()))
        self.counts['payloads']+=1

    def summary(self):
        return dict(counts=dict(self.counts),task_counts={k:dict(v) for k,v in self.task_counts.items()},
            primary_high_risk=dict(self.primary),multi_label_high_risk=dict(self.flags),
            cluster_counts={k:len(v) for k,v in self.clusters.items()},
            clusters={k:[list(x) for x in sorted(v)] for k,v in self.clusters.items()},
            overlap_is_not_independent=True,GT_used_only_after_native_scores=True,
            raw_target_content_does_not_certify_Global_ID=True)

def main(video,policy_name='v3',scope='full'):
    protect();storage_guard();assert video in TRAIN+DEV and (video in TRAIN or policy_name=='v3' and scope=='full')
    torch.set_num_threads(1);torch.manual_seed(20261009)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    policy,checkpoint,training=load_policy(policy_name);values,frames,reader=cache_inputs(video)
    model=build_tracker(video,policy,react_learned=False);executor=attach(model)
    out=OUT/'P0'/scope/policy_name/f'video{video:02d}';out.mkdir(parents=True,exist_ok=True)
    assert not (out/'RESULT.json').exists()
    source=binding(checkpoints=[checkpoint],inputs=[training],evaluator='all actual raw Gallery/Bank observer, frozen solver',
        scope=f'{scope} {policy_name} actual mutated native states; annotations never enter decisions')
    source['actual_recovery']=dict(react_learned=False,cosine_slots=[0,1,2],DEFER_logit=.75,
        bank_size=model.bank_size,test_len=model.test_len,with_bank=model.with_bank,
        Bank_only_after_MATCH_DEFER=True,START_NEW='native allocation after unavailable/rejected Bank candidates')
    cases=[];begin=time.monotonic();last=[0.]
    entries=[None] if scope=='full' else read(XV/'train_commitment_prefixes_v1'/f'video{video:02d}/RESULT.json')['counterfactual_prefixes']
    for entry in entries:
        tag='FULL' if entry is None else f"frame{entry['key'][1]:06d}_view{entry['key'][2]}_row{entry['row']}"
        prefix=None
        if entry is not None:
            descriptor=entry['prefix'];assert sha(descriptor['path'])==descriptor['SHA256']
            prefix=load_dense(descriptor['path'],map_location='cuda:0');assert fingerprint(prefix)==entry['starting_state_SHA256']
            prefix=converted_prefix(prefix)
        with gzip.open(out/(tag+'_QUERIES.jsonl.gz'),'wt') as stream:
            audit=CandidateAudit(video,reader,model,stream,prefix)
            model.jev_native_prefix_observer=audit.native_before;executor.observer=audit.before
            def after(**d):
                audit.after(**d)
                if time.monotonic()-last[0]>=15:
                    save(out/'PROGRESS.json',dict(status='AUDITING_ACTUAL_NATIVE',tag=tag,key=[video,d['frame'],d['view']],
                        counts=dict(audit.counts),seconds=time.monotonic()-begin));stream.flush();last[0]=time.monotonic()
            executor.commit_observer=after
            with torch.no_grad():raw,_=run(model,values,frames,stop=None if entry is None else min(frames-1,entry['key'][1]+31),prefix=prefix)
        if entry is not None:
            previous=read(XV/f'pilot_native_{policy_name}/F_full/pilot'/f'video{video:02d}/RESULT.json')
            old=next(r for r in previous['cases'] if r['key']==entry['key'] and r['row']==entry['row'])
            with gzip.open(old['trace']['path'],'rt') as stream:expected=list(map(json.loads,stream))
            actual={(tuple(r['key']),row):identity for r in audit.commit_rows for row,identity in enumerate(r['ids'])}
            assert all(actual[tuple(r['key']),r['row']]==r['identity'] for r in expected),'audit changed XV native branch'
            parity='ALL_XV_BRANCH_IDS_EXACT'
        else:
            predictions=raw_predictions(raw,audit.labels.images)
            reference=None
            if policy_name=='original':reference=XV/'commitment_dataset_v2'/f'video{video:02d}/RAW_PREDICTIONS.json'
            if policy_name=='v2':reference=XV/'onpolicy_pilot_dataset_v3_r2'/f'video{video:02d}/RAW_PREDICTIONS.json'
            if video in DEV:reference=XV/'pilot_online_v3/F_full_seed20261009/live'/f'video{video:02d}/RAW_PREDICTIONS.json'
            if reference is not None:assert predictions==read(reference),('audit full predictions differ',reference)
            parity='FULL_RAW_PREDICTIONS_EXACT' if reference is not None else 'NEW_FULL_FROZEN_POLICY_DIAGNOSTIC'
        # The raw Gallery remains the original production container.
        result=dict(tag=tag,summary=audit.summary(),queries=ref(out/(tag+'_QUERIES.jsonl.gz')),parity=parity,
            native_start=entry['prefix'] if entry is not None else 'unchanged native scene bootstrap',
            native_start_SHA256=entry['starting_state_SHA256'] if entry is not None else None,
            strict_train_causal_gate_eligible=video in TRAIN and video!=14,
            final_observed_identity_memory_SHA256=fingerprint(executor.memory.state_dict()))
        cases.append(result)
        print('PHASE16_P0_CASE',video,policy_name,tag,result['summary']['counts'],flush=True)
        del audit,prefix,raw;torch.cuda.empty_cache()
    save(out/'RESULT.json',dict(status='COMPLETE',binding=source,video=video,policy=policy_name,scope=scope,
        frames=frames,cases=cases,all_queries_not_selected_examples=True,seconds=time.monotonic()-begin,
        any_new_training_updates=False))
    save(out/'PROGRESS.json',dict(status='COMPLETE',cases=len(cases)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--video',type=int,required=True)
    p.add_argument('--policy',choices=['original','v1','v2','v3'],default='v3')
    p.add_argument('--scope',choices=['full','windows'],default='full');a=p.parse_args()
    main(a.video,a.policy,a.scope)
