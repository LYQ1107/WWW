"""Actual native special-action stages; deliberately untrained engineering actors."""
import torch
from jev_phase12_common import *
from gtr.modeling.jev_native_state import NativeProductionPrefixRecorder,NativeStateForkAdapter
from gtr.modeling.jev_candidate_assignment import CandidateAssignment
from gtr.modeling.visual_jev_mcmot import VisualJev
from gtr.modeling.visual_jev_mcmot.lifecycle_controller import VisualLifecycleController

class EqualLogits(VisualJev):
    def forward(self,*args):
        out=super().forward(*args);out['choice_logits']=torch.zeros_like(out['choice_logits']);return out

class DeferEngineering:
    def __init__(self):self.calls=[]
    def native_context(self,*a,**k):pass
    def match(self,batch,original,galleries,observations):
        self.calls.append({'ID_allocated_in_MATCH':False,'row_count':len(observations),'actual_active_candidate_count':len(batch.candidate_ids)})
        m=len(observations);return CandidateAssignment(tuple([-1]*m),tuple(range(m)),(),False),batch.scores.new_zeros(batch.scores.shape)
    def memory(self,*a,**k):return None
    def reactivation(self,*a,**k):return None

def main():
    protect();source=binding();assert not source['dirty'];torch.set_num_threads(1);video=13;key=(13,664,1)
    manifest=json.loads((PREVIOUS/'native_capture_v1/video13/compat/RESULT.json').read_text());entry=next(e for e in manifest['prefixes'] if tuple(e['key'])==key)
    model=build_model(video);rows=inputs(video,manifest['frames']);out=OUT/'native_special_actions_v1';original_bank=model.memory_bank;original_react=model._jev_reactivation_action;traces={};stages={}
    for mode in ['OFF','ABSTAIN_GROUP_FALLBACK','DEFER_NATIVE_BANK','DEFER_THEN_START_NEW']:
        model.visual_jev_enabled=mode!='OFF';bank=[];react=[]
        if mode=='ABSTAIN_GROUP_FALLBACK':model.visual_jev_controller=VisualLifecycleController(EqualLogits().cuda(),'MATCH_ONLY',supervision='CE',risk=True,qualified_tasks=('MATCH',))
        elif mode!='OFF':model.visual_jev_controller=DeferEngineering()
        def bank_call(hits,galleries,*a,**kw):
            before=max(hits,default=0);result=original_bank(hits,galleries,*a,**kw);after=max(hits,default=0)
            bank.append({'ID_counter_before':before,'ID_counter_after_bank':after,'allocation_in_bank_query':False,'returned_identity_refs':None if result[0] is None else result[0].cpu().tolist()});assert before==after
            return result
        def react_call(**kw):
            action=original_react(**kw)
            if mode=='DEFER_THEN_START_NEW':action='START_NEW'
            react.append({'native_stale_candidates':kw.get('candidate_track_ids',[]),'assigned_stale_reference':kw['track_id'],'native_action':action});return action
        model.memory_bank=bank_call;model._jev_reactivation_action=react_call;recorder=NativeProductionPrefixRecorder(model,[],out/mode);model.jev_native_prefix_observer=recorder.before;model.jev_candidate_commit_observer=recorder.after
        with torch.no_grad():NativeStateForkAdapter(model).run(entry['path'],rows,stop_frame=key[1])
        traces[mode]=recorder.trace;stages[mode]={'bank_queries':bank,'reactivation_actions':react,'controller_calls':getattr(model.visual_jev_controller,'calls',None) if mode!='OFF' else None}
        assert all(len(r['ids'])==len(set(r['ids'])) for r in recorder.trace)
        if mode=='ABSTAIN_GROUP_FALLBACK':
            record=next(r for r in model.visual_jev_controller.records if r['task']=='MATCH');assert record['abstain_rows']>0 and record['fallback_rows'];stages[mode]['risk_record']=record
    assert traces['OFF']==traces['ABSTAIN_GROUP_FALLBACK'],'ABSTAIN altered native state/RNG/births'
    assert stages['DEFER_NATIVE_BANK']['bank_queries'] and stages['DEFER_THEN_START_NEW']['bank_queries']
    assert any(q['native_stale_candidates'] for q in stages['DEFER_THEN_START_NEW']['reactivation_actions']),'no actual stale pool for START_NEW test'
    current=traces['DEFER_THEN_START_NEW'][0];births=[e for e in current['events'] if 'NEW_ID_COMMIT' in e['events']]
    assert births and all(e['gallery_before']==0 and e['gallery_after']==1 for e in births)
    save(out/'RESULT.json',{'status':'PASS','binding':source,'prefix_SHA256':entry['sha256'],'key':key,'ABSTAIN_full_state_and_RNG_and_birth_counter_equal_OFF':True,'DEFER_queries_real_bank_before_any_birth':True,'START_NEW_commits_new_ID_and_initializes_native_Gallery':True,'per_camera_capacity_valid':True,'stages':stages,'traces':traces,'scope':'forced untrained engineering output interventions, not learned lifecycle correctness or qualified supervision','research_performance_evidence':False});print('ACTUAL_NATIVE_ABSTAIN_DEFER_NEW_PASS',flush=True)
if __name__=='__main__':main()
