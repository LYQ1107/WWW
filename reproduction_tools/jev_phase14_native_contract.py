"""Actual GTA-throw native bank recycling, caches and lossless resume proof."""
import io
import torch
from jev_phase14_common import *
from jev_phase13_runtime import build_tracker,cache_inputs,run
from audit_jev_stage2_gta_free import phase13_prefix
from gtr.modeling.jev_native_state import fingerprint
from gtr.modeling.jev_phase14.model import ReliableIdentityPolicy
from gtr.modeling.jev_phase14.memory import CachedIdentityMemory
from gtr.modeling.jev_stage2.memory import IdentityMemory

class CoveragePolicy(torch.nn.Module):
    def __init__(self):
        super().__init__();self.net=ReliableIdentityPolicy('multi_question').cuda().eval();self.refs=[];self.calls=0;self.inputs=[];self.values=[]
    def forward(self,x):
        z=self.net(x);self.calls+=1;self.inputs.append(fingerprint(x));self.values.append(fingerprint(z))
        q=z.shape[1];frame=round(float(x['detection_meta'][0,0,6])*2000) if q else 0;task=int(x['question_type'][0,0]) if q else 0
        forced=torch.full_like(z,-100.);forced[:,:,-1]=100.
        if q and (task==0 and frame<12 or task==1 and frame>=55) and 1 in self.refs:forced[0,0,self.refs.index(1)]=200.
        return forced

def execute(cached,resume=None,output=None):
    torch.manual_seed(20261009);policy=CoveragePolicy();model=build_tracker(12,policy=policy,react_learned=True)
    ex=model.jev_stage2_executor;ex.memory_factory=CachedIdentityMemory if cached else IdentityMemory
    values,frames,_=cache_inputs(12);states=[];events=[];prefix=[None];builder=ex.batch;tensor_hash_cache={}
    def batch(current,refs,*args):policy.refs=list(refs);return builder(current,refs,*args)
    ex.batch=batch
    def after(**d):
        from gtr.modeling.meta_arch.gtr_rcnn import poss_ids,old_reids
        ids=d['instances'][-1].track_ids.tolist();assert len(ids)==len(set(ids))
        events.extend(dict(key=[d['frame'],d['view']],**e) for e in d['events'])
        states.append(dict(key=[d['frame'],d['view']],id_count=d['id_count'],ids=ids,
            all_hits=dict(d['hits']),possible_ids=sorted(poss_ids.poss_ids),
            old_reids=fingerprint(old_reids.old_reids),memory=fingerprint(ex.memory.state_dict()),
            full_gallery_instances=fingerprint({'instances':d['instances'],'galleries':d['galleries']},tensor_cache=tensor_hash_cache),
            trajectory_RNG=fingerprint(dict(state=model._jev_trajectory_rng.getstate(),draws=model._jev_trajectory_rng._trajectory_draws))))
    def before(**d):
        if (d['frame'],d['view'])==(25,0):
            # Prefix containers share storage with the production loop. Freeze
            # at the boundary, before later commits mutate those references.
            data=io.BytesIO();torch.save(phase13_prefix(model,d),data)
            if output is not None:output.write_bytes(data.getvalue())
            data.seek(0);prefix[0]=torch.load(data,map_location='cuda:0')
    ex.commit_observer=after;model.jev_native_prefix_observer=before
    with torch.no_grad():raw,_=run(model,values,frames,stop=105,prefix=resume)
    committed=raw[:2*106]
    assert all(i.has('track_ids') or len(i)==0 for i in committed)
    ids=[i.track_ids.cpu().tolist() if i.has('track_ids') else [] for i in committed]
    return dict(states=states,events=events,input_SHA=policy.inputs,NN_logit_SHA=policy.values,ids=ids,
                memory_cache_hits=getattr(ex.memory,'cache_hits',None),memory_cache_misses=getattr(ex.memory,'cache_misses',None)),prefix[0]

def main():
    protect();source=binding();torch.set_num_threads(1);out=OUT/'native_contract_v2';out.mkdir(parents=True,exist_ok=True)
    assert not (out/'TRACE.json').exists(),'native proof artifacts are immutable'
    save(out/'START.json',dict(binding=source,scope='actual loaded source captured before execution'))
    baseline,prefix=execute(False,output=out/'PREFIX25_UNCACHED.pth')
    cached,cprefix=execute(True,output=out/'PREFIX25_CACHED.pth')
    for key in ['states','events','input_SHA','NN_logit_SHA','ids']:assert baseline[key]==cached[key],key
    assert cached['memory_cache_hits']>0
    resumed,_=execute(True,resume=cprefix)
    assert resumed['states']==[s for s in cached['states'] if tuple(s['key'])>=(25,0)]
    assert resumed['ids']==cached['ids']
    recoveries=[e['key'] for e in cached['events'] if e['action']=='REACTIVATE' and e['identity']==1]
    assert len(recoveries)>=2 and recoveries[0][0]<recoveries[1][0]
    save(out/'TRACE.json',cached)
    save(REPORTS/'NATIVE_CACHE_PARITY.json',dict(status='PASS',binding=source,
        exact_causal_input_tensor_SHA=True,exact_actual_NN_logit_SHA=True,exact_native_IDS_Gallery_bank_hits_RNG_meta=True,
        exact_resume=True,ID1_repeated_real_bank_recycling=recoveries,GTA_throw=True,
        cache_hits=cached['memory_cache_hits'],cache_misses=cached['memory_cache_misses'],numeric_difference=0.,
        full_trace_SHA256=sha(out/'TRACE.json'),prefixes=[{'path':str(p),'SHA256':sha(p)} for p in [out/'PREFIX25_UNCACHED.pth',out/'PREFIX25_CACHED.pth']],
        scope='106 real TRAINvideo12 scene frames, new multi-question NN executed before forced lawful existing-reference actions for coverage; no fabricated IDs/options or GT; not tracking performance or full-video cached parity'))
    print('PHASE14_NATIVE_CACHE_CONTRACT_PASS',recoveries,cached['memory_cache_hits'],flush=True)

if __name__=='__main__':main()
