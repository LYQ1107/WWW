"""Same-prefix/RNG 2x2, single targeted reader factor and ongoing native writes."""
from jev_phase16_common import *
import argparse
import copy
import gzip
import time
import torch
from jev_phase13_runtime import build_tracker,cache_inputs,run
from jev_phase15_pilot_native import converted_prefix
from jev_phase15_causal_metrics import FrozenClearFuture
from jev_phase14_artifacts import load_dense
from jev_phase16_memory_risk import MemoryRisk
from gtr.modeling.jev_phase15.native_commit_adapter import attach as attach_xv
from gtr.modeling.jev_phase16.native_memory_commit_adapter import attach as attach_safe
from gtr.modeling.jev_phase16.multi_prototype_memory import MultiPrototypeMemory,VARIANTS
from gtr.modeling.jev_native_state import fingerprint
from gtr.modeling.jev_stage2.assignment import lawful_choice


class FirstReader(torch.nn.Module):
    def __init__(self, policy, prefix, entry, evidence):
        super().__init__();self.policy=policy;self.entry=entry;self.evidence=evidence
        self.reader=MultiPrototypeMemory.from_prefix(VARIANTS[4],prefix) if evidence else None
        self.context=None;self.refs=None;self.triggered=False;self.intervention=None

    def forward(self,x):
        scores=self.policy(x);details=self.policy.runtime_details
        target=tuple(self.entry['key'][1:]);key=self.context['frame'],self.context['view']
        if key==target and not self.triggered:
            row=self.entry['row'];result=scores.clone();alt=None
            if self.evidence:
                ax,_=self.reader.batch(x,self.refs,key[1]);alt=self.policy(ax)
                alternate_details=self.policy.runtime_details
                result[0,row]=alt[0,row]
                details={k:v.clone() for k,v in details.items()}
                for k,v in details.items():
                    if v.ndim>=2 and v.shape[1]==scores.shape[1]:v[:,row]=alternate_details[k][:,row]
            base_choices=lawful_choice(scores[0],x['legal'][0]);choices=lawful_choice(result[0],x['legal'][0])
            self.intervention=dict(first_original_batch_SHA256=fingerprint(x),first_original_logits_SHA256=fingerprint(scores),
                actual_first_logits_SHA256=fingerprint(result),refs=list(self.refs),row=row,
                original_assignment=base_choices,executed_assignment=choices,
                non_target_rows_redistributed_by_native_capacity=[i for i,(a,b) in enumerate(zip(base_choices,choices)) if i!=row and a!=b],
                E_reader_single_decision_only=self.evidence,GT_or_future_inputs=False)
            self.triggered=True;scores=result
        self.runtime_details=details
        return scores


def main(video,arm):
    protect();storage_guard();assert video in TRAIN and arm in 'ABCD';torch.set_num_threads(1)
    torch.manual_seed(20261009);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    protocol=read(REPORTS/'P2_P3_PROTOCOL.json');assert read(REPORTS/'NATIVE_MEMORY_PARITY.json')['status']=='PASS_EARLIEST_NATIVE_CONTRACT'
    values,frames,reader=cache_inputs(video);case=XV/'train_commitment_prefixes_v1'/f'video{video:02d}'
    manifest=read(case/'RESULT.json');metric=FrozenClearFuture('multi_question',20261009,video,case_override=case,duplicate_gt_diagnostic=video==14)
    frozen,ck,training=load_policy('v3');out=OUT/'P3_factorial'/arm/f'video{video:02d}'
    out.mkdir(parents=True,exist_ok=True);assert not(out/'RESULT.json').exists()
    source=binding(checkpoints=[ck],inputs=[training,ref(case/'RESULT.json'),ref(REPORTS/'P2_P3_PROTOCOL.json')],
        evaluator=metric.evaluator,scope='single targeted failed-reader negative control + ongoing admission in real mutated native future; no GT actor input')
    results=[];start=time.monotonic()
    old_results=read(XV/'pilot_native_v3/F_full/pilot'/f'video{video:02d}/RESULT.json')
    for entry in manifest['counterfactual_prefixes']:
        assert sha(entry['prefix']['path'])==entry['prefix']['SHA256']
        prefix=load_dense(entry['prefix']['path'],map_location='cuda:0');assert fingerprint(prefix)==entry['starting_state_SHA256']
        prefix=converted_prefix(prefix);initial=fingerprint(prefix);boundary=tuple(entry['key'][1:])
        rng=fingerprint({k:v for k,v in prefix.items() if 'rng' in k});wrapper=FirstReader(frozen,prefix,entry,arm in 'CD')
        model=build_tracker(video,wrapper,react_learned=False)
        executor=attach_safe(model,'bounded_quarantine',prefix) if arm in 'BD' else attach_xv(model)
        original_scores=executor.scores
        def scores(batch,refs,context,task):
            wrapper.context=context;wrapper.refs=refs
            if arm in 'CD' and task==1 and (context['frame'],context['view'])==boundary and entry['row'] in context.get('rows',[]):
                row=context['rows'].index(entry['row']);alt,_=wrapper.reader.batch(batch,refs,context['view'])
                batch=dict(batch);pair=batch['pair_evidence'].clone();pair[:,row,:,:4]=alt['pair_evidence'][:,row,:,:4];batch['pair_evidence']=pair
            return original_scores(batch,refs,context,task)
        executor.scores=scores;risk=MemoryRisk(video,reader,prefix)
        executor.observer=risk.before;admissions=[]
        def after(**d):
            d['video']=video;risk.after(**d)
            if arm in 'BD':admissions.extend(copy.deepcopy(executor.memory.last_admission))
        executor.commit_observer=after
        h64=any(e['key']==entry['key'] and e['row']==entry['row'] for e in protocol['H64_prefixes'])
        maxh=64 if h64 else 32
        unchanged={k:list(v) for k,v in metric.ids(prefix['instances']).items() if k<boundary}
        with torch.no_grad():raw,_=run(model,values,frames,stop=min(frames-1,boundary[0]+maxh-1),prefix=copy.deepcopy(prefix))
        assert wrapper.triggered;ids=metric.ids(raw);assert all(ids[k]==v for k,v in unchanged.items())
        H={str(h):dict(**metric.horizon(ids,boundary,h,frames),identity_consequences=risk.horizon(boundary,min(frames,boundary[0]+h))) for h in [8,16,32]+([64] if h64 else [])}
        old=next(r for r in old_results['cases'] if r['key']==entry['key'] and r['row']==entry['row'])
        original_parity=None
        if arm=='A':
            with gzip.open(old['trace']['path'],'rt') as stream:old_trace=list(map(json.loads,stream))
            actual={(tuple(r['key']),r['row']):r['identity'] for r in risk.trace}
            assert all(actual[tuple(r['key']),r['row']]==r['identity'] for r in old_trace)
            original_parity='ALL_XV_V3_H32_COMMITTED_IDS_EXACT'
        name=f'frame{boundary[0]:06d}_view{boundary[1]}_row{entry["row"]}'
        with gzip.open(out/(name+'.jsonl.gz'),'wt') as stream:
            for r in risk.strong_trace:stream.write(json.dumps(r,allow_nan=False)+'\n')
        with gzip.open(out/(name+'_admission.jsonl.gz'),'wt') as stream:
            for r in admissions:stream.write(json.dumps(r,allow_nan=False)+'\n')
        result=dict(status='COMPLETE',arm=arm,video=video,key=entry['key'],row=entry['row'],category=entry['category'],
            prefix=entry['prefix'],original_state_SHA256=entry['starting_state_SHA256'],derived_state_SHA256=initial,
            actual_RNG_SHA256=rng,same_saved_RNG=True,intervention=wrapper.intervention,H=H,
            trace=ref(out/(name+'.jsonl.gz')),admission_trace=ref(out/(name+'_admission.jsonl.gz')),
            admissions=collections_counts(admissions),final_memory_SHA256=fingerprint(executor.memory.state_dict()),
            raw_Gallery_SHA256=fingerprint(executor.memory.raw_galleries) if arm in 'BD' else None,
            native_original_parity=original_parity,actual_mutated_future=True,past_IDs_unchanged=True,
            first_actual_ID=next(r['identity'] for r in risk.strong_trace if r['key']==list(boundary) and r['row']==entry['row']),
            strict_gate_eligible=video!=14)
        save(out/(name+'.json'),dict(result,binding=source));results.append(result)
        save(out/'PROGRESS.json',dict(status='RUNNING',completed_prefixes=len(results),total=6,last_key=entry['key'],arm=arm,seconds=time.monotonic()-start))
        print('PHASE16_NATIVE_MEMORY_FACTORIAL',video,arm,entry['key'],{h:v['identity_consequences']['counts'] for h,v in H.items()},flush=True)
        del prefix,model,executor,risk,wrapper;torch.cuda.empty_cache()
    save(out/'RESULT.json',dict(status='COMPLETE',binding=source,arm=arm,video=video,cases=results,seconds=time.monotonic()-start))
    save(out/'PROGRESS.json',dict(status='COMPLETE',completed_prefixes=6,arm=arm))


def collections_counts(events):
    import collections
    c=collections.Counter()
    for r in events:c[r['status']]+=1;c['admitted_observations']+=len(r['admitted_indices']);c['raw_observations']+=1
    return dict(c)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--video',type=int,required=True);p.add_argument('--arm',choices=list('ABCD'),required=True)
    a=p.parse_args();main(a.video,a.arm)
