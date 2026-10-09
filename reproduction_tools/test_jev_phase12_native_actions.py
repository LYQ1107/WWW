"""Engineering interventions on genuine options; no training/performance claim."""
import torch
from jev_phase12_common import *
from gtr.modeling.jev_native_state import NativeProductionPrefixRecorder,NativeStateForkAdapter
from gtr.modeling.visual_jev_mcmot import VisualJev
from gtr.modeling.visual_jev_mcmot.lifecycle_controller import VisualLifecycleController

class ForcedOptionEngineeringModel(VisualJev):
    def __init__(self,row,col):
        super().__init__();self.forced_row=row;self.forced_col=col;self.armed=True
    def forward(self,state,questions,options):
        out=super().forward(state,questions,options)
        if bool((questions.types==0).all()) and options.evidence.shape[2]:
            # Controlled test output intervention, not a fitted network outcome.
            logits=options.evidence[...,0].clone()
            if self.armed:
                logits[0,self.forced_row]=logits[0,self.forced_row]-100
                logits[0,self.forced_row,self.forced_col]=100
                self.armed=False
            out['choice_logits']=logits
        return out

def main():
    protect();source=binding();assert not source['dirty']
    key=(12,85,0);row=1
    capture=PREVIOUS/'native_capture_v1/video12/compat';manifest=json.loads((capture/'RESULT.json').read_text());entry=next(r for r in manifest['prefixes'] if tuple(r['key'])==key)
    packet=torch.load(entry['packet_path'],map_location='cpu');batch=packet['packet']['batch'];factual=packet['commit']['ids'][row]
    columns=[j for j,ref in enumerate(batch.candidate_ids) if batch.legal_mask[row,j] and ref!=factual]
    assert columns;col=columns[0];reference=batch.candidate_ids[col]
    model=build_model(12);rows=inputs(12,manifest['frames']);out=OUT/'native_actions_v1';out.mkdir(exist_ok=True)
    traces={};scores={};events={}
    for mode in ['OFF','FORCED_GENUINE_OPTION']:
        model.visual_jev_enabled=mode!='OFF'
        if mode!='OFF':
            controller=VisualLifecycleController(ForcedOptionEngineeringModel(row,col).cuda(),'MATCH_ONLY',supervision='CE',risk=False,qualified_tasks=('MATCH',))
            # Qualification here grants engineering test execution only; report
            # explicitly forbids using this random/test actor for research eval.
            model.visual_jev_controller=controller
        recorder=NativeProductionPrefixRecorder(model,[],out/mode);native_scores=[]
        def after(**values):
            candidate=values['candidate']
            if candidate is not None:native_scores.append({'key':[values['frame'],values['view']],'scores':candidate['batch'].scores.detach().cpu().clone(),'candidate_ids':candidate['batch'].candidate_ids})
            recorder.after(**values)
        model.jev_native_prefix_observer=recorder.before;model.jev_candidate_commit_observer=after
        with torch.no_grad():NativeStateForkAdapter(model).run(entry['path'],rows,stop_frame=key[1]+1)
        traces[mode]=recorder.trace;scores[mode]=native_scores;events[mode]=[e for r in recorder.trace for e in r['events']]
    actual=traces['FORCED_GENUINE_OPTION'][0]
    assert actual['ids'][row]==reference and actual['ids'][row]!=factual
    assert all(len(r['ids'])==len(set(r['ids'])) for r in traces['FORCED_GENUINE_OPTION'])
    event=actual['events'][row];assert event['track']==reference and event['gallery_after']==event['gallery_before']+1
    assert 'MEMORY_WRITE' in event['events']
    next_changed=False;comparisons=[]
    for a,b in zip(scores['OFF'][1:],scores['FORCED_GENUINE_OPTION'][1:]):
        same=a['candidate_ids']==b['candidate_ids'] and a['scores'].shape==b['scores'].shape
        error=float((a['scores']-b['scores']).abs().max()) if same and a['scores'].numel() else None
        changed=not same or (error is not None and error>1e-6);next_changed|=changed
        comparisons.append({'key':a['key'],'candidate_refs_and_shape_same':same,'score_max_error':error,'changed':changed})
    assert next_changed,'native next-step evidence did not respond to actual identity mutation'
    result={'status':'PASS','binding':source,'key':key,'row':row,'factual_id':factual,'forced_genuine_option_reference':reference,'committed_id_changed':True,'correct_single_memory_write':event,'fresh_next_native_candidates_changed':comparisons,'all_per_camera_IDs_unique':True,'controller_is_untrained_engineering_test_only':True,'research_performance_evidence':False,'supervision_qualification_not_granted':True,'typed_memory_reactivation_status':'UNTRAINED/FROZEN_FALLBACK','traces':traces,'scientific_gate':'native executor functionality, not learned correctness'}
    save(out/'RESULT.json',result);print('NATIVE_REAL_OPTION_COMMIT_AND_FRESH_NEXT_SCORES_PASS',flush=True)

if __name__=='__main__':main()
