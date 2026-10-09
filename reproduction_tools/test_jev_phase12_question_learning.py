"""Controlled structural sanity: same visual evidence, context-dependent answer."""
import dataclasses,time
import torch
import torch.nn.functional as F
from jev_phase12_common import *
from gtr.modeling.visual_jev_mcmot import *

def main():
    protect();assert json.loads((REPORTS/'STRUCTURAL_TESTS.json').read_text())['status']=='PASS'
    torch.set_num_threads(1);device='cuda:0';torch.manual_seed(20261009)
    # Exactly balanced signs; all visual/options/state otherwise identical.
    b=16
    sv=torch.randn(1,5,1152,device=device).expand(b,-1,-1).clone()
    sm=torch.zeros(b,5,8,device=device);sm[:,:,0]=1
    state=OnlineVisualState(sv,sm,torch.ones(b,5,device=device,dtype=torch.bool))
    qv=torch.randn(1,1,1152,device=device).expand(b,-1,-1).clone()
    qc=torch.zeros(b,1,64,device=device);qc[:b//2,0,0]=-1;qc[b//2:,0,0]=1
    q=QuestionDescriptor(qv,qc,torch.zeros(b,1,device=device,dtype=torch.long),torch.ones(b,1,device=device,dtype=torch.bool))
    ov=torch.randn(1,1,2,3,1152,device=device).expand(b,-1,-1,-1,-1).clone()
    o=OptionTensors(ov,torch.ones(b,1,2,3,device=device,dtype=torch.bool),torch.zeros(b,1,2,12,device=device),torch.zeros(b,1,2,device=device,dtype=torch.long),torch.ones(b,1,2,device=device,dtype=torch.bool))
    target=(qc[:,0,0]>0).long();results=[]
    for variant in ['full','fixed_question']:
        torch.manual_seed(20261009);model=VisualJev(variant).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        logs=[];start=time.perf_counter()
        for step in range(300):
            model.train();optimizer.zero_grad(set_to_none=True);out=model(state,q,o)
            loss=F.cross_entropy(out['choice_logits'][:,0],target);assert torch.isfinite(loss)
            loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),5));optimizer.step()
            if step%25==0 or step==299:
                logs.append({'step':step+1,'loss':float(loss),'accuracy':float((out['choice_logits'][:,0].argmax(-1)==target).float().mean()),'gradient_norm':norm})
        model.eval()
        with torch.no_grad():out=model(state,q,o);accuracy=float((out['choice_logits'][:,0].argmax(-1)==target).float().mean())
        results.append({'variant':variant,'accuracy':accuracy,'training_curve':logs,'seconds':time.perf_counter()-start,'params':parameters(model)})
    assert results[0]['accuracy']>=.95 and results[1]['accuracy']==.5
    previous=json.loads((REPORTS/'QUESTION_CONDITIONING_TESTS.json').read_text())
    previous.update(status='PASS',synthetic_learnability={'status':'PASS','binding':binding(),'results':results,'same_visual_options_state_both_contexts':True,'context_sign_only_changes_label':True,'real_MOT_evidence':False,'before_real_data_training':True})
    save(OUT/'question_sanity_v1/RESULT.json',previous);save(REPORTS/'QUESTION_CONDITIONING_TESTS.json',previous)
    print('DYNAMIC_QUERY_SANITY_PASS',[(r['variant'],r['accuracy']) for r in results],flush=True)

if __name__=='__main__':main()
