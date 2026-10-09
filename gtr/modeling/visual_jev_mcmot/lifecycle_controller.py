import time
import torch
from .identity_history_tokens import build_match_inputs,build_memory_inputs,native_time_metadata
from .calibration import risk_mask
from .native_action_adapter import lawful_match
from .schemas import masked_softmax

class VisualLifecycleController:
    def __init__(self,model,mode='SHADOW',*,supervision='joint',risk=True,qualified_tasks=(),observer=None):
        if mode not in {'OFF','SHADOW','MATCH_ONLY','FULL_LIFECYCLE'}:raise ValueError('unknown lifecycle mode')
        if mode=='FULL_LIFECYCLE' and not {'MATCH','MEMORY','REACTIVATION'}.issubset(qualified_tasks):
            raise ValueError('FULL_LIFECYCLE requires all real native supervision gates')
        if mode=='MATCH_ONLY' and 'MATCH' not in qualified_tasks:raise ValueError('unqualified MATCH checkpoint')
        self.model=model.eval();self.mode=mode;self.supervision=supervision;self.risk=risk
        self.qualified_tasks=set(qualified_tasks);self.observer=observer
        self.current=None;self.records=[];self.timings=[]

    def native_context(self,instances,galleries,frame,view,first=False):
        self.current={'instances':instances,'galleries':galleries,'frame':frame,'view':view,'first':first}

    @torch.no_grad()
    def match(self,batch,original,galleries,observations):
        if self.mode=='OFF':return None
        ctx=self.current or {'frame':0,'view':batch.view,'instances':[],'first':False}
        torch.cuda.synchronize(observations.device) if observations.is_cuda else None
        begin=time.perf_counter()
        metadata=native_time_metadata(ctx['instances'],ctx['frame'],ctx['view'],first=ctx['first'])
        inp=build_match_inputs(batch,observations,galleries,frame=ctx['frame'],view=ctx['view'],metadata=metadata)
        built=time.perf_counter();out=self.model(*inp)
        k=len(batch.candidate_ids)
        pref=out['choice_logits'][0,:,:k]
        if self.supervision=='joint' and self.model.variant not in {'no_H32','no_consequence'}:
            pref=pref+.25*out['consequences'][0,:,:k,2]
        elif self.supervision=='H32':pref=out['consequences'][0,:,:k,2]
        legal=batch.legal_mask & inp[2].mask[0,:,:k]
        confidence=masked_softmax(pref,legal)
        abstain=risk_mask(confidence,legal) if self.risk else torch.zeros(len(pref),dtype=torch.bool,device=pref.device)
        if k:
            maximum=pref.masked_fill(~legal,-torch.inf).amax(-1)
            raw=(batch.scores-batch.thresholds[None]).masked_fill(~legal,-torch.inf).amax(-1)
            maximum=torch.where(legal.any(-1),maximum,torch.zeros_like(maximum))
            raw=torch.where(legal.any(-1),raw,torch.zeros_like(raw))
            values=pref-maximum[:,None]+raw[:,None]
        else:values=pref
        from dataclasses import replace
        valid_batch=replace(batch,legal_mask=legal)
        assignment,fallback=lawful_match(values,valid_batch,abstain)
        torch.cuda.synchronize(observations.device) if observations.is_cuda else None
        end=time.perf_counter()
        record={'task':'MATCH','frame':int(ctx['frame']),'view':int(ctx['view']),'questions':len(observations),'options':k,'mode':self.mode,'abstain_rows':int(abstain.sum()),'fallback_rows':fallback,'defer_rows':assignment.new_rows,'memory_reused_for_questions':len(observations),'state_tokens':int(inp[0].mask.sum()),'qualified':'MATCH' in self.qualified_tasks,'context_rebuilt':True}
        self.records.append(record);self.timings.append({'task':'MATCH','feature_ms':(built-begin)*1000,'policy_assignment_ms':(end-built)*1000,'total_ms':(end-begin)*1000})
        if self.observer:self.observer(record=record,inputs=inp,outputs=out,assignment=assignment,values=values)
        return None if self.mode=='SHADOW' else (assignment,values)

    @torch.no_grad()
    def memory(self,observation,gallery,context,view):
        if self.mode=='OFF':return None
        inp=build_memory_inputs(observation,gallery,context,view)
        out=self.model(*inp)
        self.records.append({'task':'MEMORY','mode':self.mode,'status':'UNTRAINED/FROZEN_FALLBACK','gallery_length':len(gallery),'context_rebuilt_after_identity_commit':True,'write_probability_diagnostic':float(out['write_probability'][0,0])})
        return None

    @torch.no_grad()
    def reactivation(self,batch,observation,bank_galleries):
        if self.mode=='OFF':return None
        inp=build_match_inputs(batch,observation,bank_galleries,view=batch.view,question_type=1)
        out=self.model(*inp)
        self.records.append({'task':'REACTIVATION','mode':self.mode,'status':'UNTRAINED/FROZEN_FALLBACK','actual_native_stale_candidates':len(batch.candidate_ids),'questions':len(observation),'typed_logits_shape':list(out['choice_logits'].shape)})
        return None
