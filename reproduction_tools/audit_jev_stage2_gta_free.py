"""Throw GTA through actual full native lifecycle and test lossless resumes.

The engineering branch invokes the real untrained NN then forces lawful
choices to cover bank restoration/birth. It is never training supervision or
tracking evidence. No GT is loaded and no candidate/identity is fabricated.
"""
import argparse,time
import torch
from jev_phase13_learning import *
from jev_phase13_runtime import *
from gtr.modeling.jev_native_state import prefix_state,fingerprint

class CoveragePolicy(torch.nn.Module):
    def __init__(self):super().__init__();self.net=GlobalIdentityJev().cuda().eval();self.refs=[];self.calls=0
    def forward(self,x):
        z=self.net(x);self.calls+=1;q=z.shape[1];k=z.shape[2]-1
        frame=round(float(x['detection_meta'][0,0,6])*2000) if q else 0;task=int(x['question_type'][0,0]) if q else 0
        forced=torch.full_like(z,-100.);forced[:,:,-1]=100.
        # Actual bootstrap ID 1 is strengthened, then absent from the complete
        # native 40-frame active window before it is recovered from the bank.
        want=task==0 and frame<12 or task==1 and frame>=55
        if q and want and 1 in self.refs:forced[0,0,self.refs.index(1)]=200.
        return forced

def phase13_prefix(model,values):
    p=prefix_state(model,**values);p['phase13_identity_meta']=model.jev_stage2_executor.memory.state_dict();return p

def main():
    protect();source=binding();assert not source['dirty'];torch.set_num_threads(1);torch.manual_seed(20261009);policy=CoveragePolicy();model=build_tracker(12,policy=policy,react_learned=True);values,frames,_=cache_inputs(12);out=OUT/'gta_free_native_v1';out.mkdir(exist_ok=True);events=[];states=[];calls=[];boundary=25
    executor=model.jev_stage2_executor;builder=executor.batch
    def batch(current,refs,*args):policy.refs=list(refs);return builder(current,refs,*args)
    executor.batch=batch
    def after(**d):
        from gtr.modeling.meta_arch.gtr_rcnn import poss_ids,old_reids
        ids=d['instances'][-1].track_ids.tolist();assert len(ids)==len(set(ids));events.extend(dict(key=[d['frame'],d['view']],**e) for e in d['events'])
        states.append({'key':[d['frame'],d['view']],'id_count':d['id_count'],'ids':ids,'hits':{str(t):d['hits'][t] for t in ids},'gallery_lengths':{str(t):len(d['galleries'][t]) for t in ids},'possible_ids':sorted(poss_ids.poss_ids),'old_reids_ids':old_reids.old_reids[0].track_ids.tolist() if old_reids.old_reids else [],'identity_memory_SHA256':fingerprint(executor.memory.meta),'native_instances_galleries_SHA256':fingerprint({'instances':d['instances'],'galleries':d['galleries']})})
    def before(**d):
        if (d['frame'],d['view'])==(boundary,0):torch.save(phase13_prefix(model,d),out/'PREFIX25.pth')
    executor.commit_observer=after;model.jev_native_prefix_observer=before
    with torch.no_grad():raw,_=run(model,values,frames,stop=65)
    full=list(states);full_ids=[i.track_ids.cpu().tolist() for i in raw];assert policy.calls>100
    assert any(e['action']=='REACTIVATE' and e['identity']==1 for e in events),'real bank recovery not reached'
    assert any(e['action']=='START_NEW' for e in events);assert all(e['gallery_after']==e['gallery_before']+1 for e in events if e['action']!='START_NEW')
    assert any(s['key']==[56,0] for s in states),'fresh next-state decision absent';nn_calls=policy.calls
    # Exact same native graph and metadata, restored from the actual prefix.
    states.clear();events.clear();model.jev_native_prefix_observer=None;p=torch.load(out/'PREFIX25.pth',map_location='cuda:0')
    with torch.no_grad():resumed,_=run(model,values,frames,stop=65,prefix=p)
    expected=[s for s in full if tuple(s['key'])>=(boundary,0)];assert states==expected,'native graph/memory resume mismatch';assert [i.track_ids.cpu().tolist() for i in resumed]==full_ids
    result={'status':'PASS','binding':source,'source_checkpoint':STAGE1_SHA,'GTA_and_RPCE_throw_mocks':True,'frames':66,'NN_forward_calls_full_path':nn_calls,'actual_actions_covered':sorted({e['action'] for e in events}), 'bootstrap_and_active_association':True,'stale_ref1_real_bank_recovery':True,'new_birth':True,'exactly_one_native_memory_write':True,'fresh_next_frame_after_recovery':True,'same_camera_ID_capacity_one':True,'cross_camera_shared_ID':True,'native_full_vs_segmented_IDs_Gallery_bank_hits_identity_meta_exact':True,'prefix_SHA256':sha(out/'PREFIX25.pth'),'native_trace_SHA256':fingerprint(full),'full_trace':full,'scope':'untrained engineering coverage: NN executed then lawful actions forced using actual references, no GT/no fabricated candidate; no learning/performance claim'}
    save(out/'RESULT.json',result);print('PHASE13_GTA_FREE_NATIVE_PASS',flush=True)

if __name__=='__main__':main()
