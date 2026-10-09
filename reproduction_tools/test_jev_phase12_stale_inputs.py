"""Real stale semantic inputs, with identical forced-MATCH native control."""
import torch
from jev_phase12_common import *
from gtr.modeling.jev_native_state import NativeStateForkAdapter,NativeProductionPrefixRecorder
from gtr.modeling.visual_jev_mcmot import VisualJev
from gtr.modeling.visual_jev_mcmot.lifecycle_controller import VisualLifecycleController
from test_jev_phase12_special_actions import DeferEngineering

class ReadOnlyStaleObserver(DeferEngineering):
    def __init__(self):
        super().__init__();self.reader=VisualLifecycleController(VisualJev().cuda(),'SHADOW');self.current=None;self.questions=[]
    def native_context(self,*a,**kw):self.reader.native_context(*a,**kw);self.current=self.reader.current
    def reactivation(self,batch,observation,bank):
        assert not batch.evidence12[...,8].any() and batch.evidence12[...,9].all()
        for col,ref in enumerate(batch.candidate_ids):assert (batch.evidence12[:,col,5]==len(self.current['galleries'][ref])).all()
        self.reader.reactivation(batch,observation,bank)
        self.questions.append({'references':batch.candidate_ids,'actual_Gallery_lengths':[len(self.current['galleries'][ref]) for ref in batch.candidate_ids],'stale_not_active':True,'actual_bank_eligible':True,'typed_forward_executed':True})
    def memory(self,*a,**kw):self.reader.memory(*a,**kw)

def main():
    protect();source=binding();assert not source['dirty'];torch.set_num_threads(1);key=(13,664,1)
    manifest=json.loads((PREVIOUS/'native_capture_v1/video13/compat/RESULT.json').read_text());entry=next(e for e in manifest['prefixes'] if tuple(e['key'])==key);model=build_model(13);rows=inputs(13,manifest['frames']);out=OUT/'stale_inputs_v1';traces={};observer=None
    for tag in ['FORCED_DEFER_CONTROL','FORCED_DEFER_READONLY_TYPED']:
        model.visual_jev_enabled=True;controller=DeferEngineering() if tag.endswith('CONTROL') else ReadOnlyStaleObserver();model.visual_jev_controller=controller
        recorder=NativeProductionPrefixRecorder(model,[],out/tag);model.jev_native_prefix_observer=recorder.before;model.jev_candidate_commit_observer=recorder.after
        with torch.no_grad():NativeStateForkAdapter(model).run(entry['path'],rows,stop_frame=key[1])
        traces[tag]=recorder.trace
        if isinstance(controller,ReadOnlyStaleObserver):observer=controller
    assert traces['FORCED_DEFER_CONTROL']==traces['FORCED_DEFER_READONLY_TYPED'];assert observer.questions
    save(out/'RESULT.json',{'status':'PASS','binding':source,'key':key,'actual_stale_questions':observer.questions,'native_full_Gallery_bank_RNG_commit_equal_forced_control':True,'scope':'genuine native stale bank reached by engineering-only forced DEFER; readonly untrained typed forward does not alter memory/bank transitions; no artificial stale ID or recovery label, primary frozen online source unchanged'})
    print('GENUINE_STALE_INPUTS_AND_READONLY_NATIVE_STATE_PASS',flush=True)
if __name__=='__main__':main()
