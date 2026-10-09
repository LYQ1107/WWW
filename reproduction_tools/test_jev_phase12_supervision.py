"""Meaningful partial-label boundary tests; no scientific fit performed."""
import torch
from jev_phase12_learning import *

def main():
    protect();rows,manifest=load_data();stats=normalization(rows)
    selected=next(r for r in rows if r['CE_eligible'] and r['ranking_eligible'] and (~r['known_mask']).any())
    args,labels=collate([selected]);k=len(selected['candidate_ids'])
    logits=torch.randn(1,1,k,requires_grad=True);q=torch.randn(1,1,k,3,requires_grad=True)
    output={'choice_logits':logits,'consequences':q}
    ce,_=losses(output,labels,stats,'CE','full');ce.sum().backward()
    assert torch.equal(logits.grad[0,0,~selected['known_mask']],torch.zeros_like(logits.grad[0,0,~selected['known_mask']]))
    assert q.grad is None
    output={'choice_logits':torch.randn(1,1,k,requires_grad=True),'consequences':q}
    loss,_=losses(output,labels,stats,'H32','full');assert torch.isfinite(loss).all();loss.sum().backward()
    assert torch.isfinite(q.grad).all()
    # H32 ranking can couple its executed columns only; Q regression masks
    # unknown at each horizon, so no unexplored action target enters the loss.
    unknown=~selected['consequence_mask'];assert torch.equal(q.grad[0,0][unknown],torch.zeros_like(q.grad[0,0][unknown]))
    assert not torch.isfinite(labels['targets'][~labels['utility_mask']]).any()
    model=build_network('full',stats).eval();model_args=args
    with torch.no_grad():output=model(*model_args)
    for x in output.values():
        if torch.is_tensor(x):assert torch.isfinite(x).all()
    assert torch.equal(model.normalized(args[1].context,'state')[...,stats['state_constant']],torch.zeros_like(model.normalized(args[1].context,'state')[...,stats['state_constant']]))
    result={'status':'PASS','binding':binding(),'visual_dataset_SHA256':manifest['SHA256'],'unknown_CE_gradients_zero':True,'unknown_Q_target_and_gradients_masked':True,'unexecuted_targets_NaN':True,'real_v3_batch_finite':True,'TRAIN_only_constant_normalization_correct':True,'all_models95_Tiny_gate_required':False,'real_training_started':False}
    save(OUT/'supervision_contract_v1/RESULT.json',result);save(REPORTS/'SUPERVISION_CONTRACT_TESTS.json',result);print('PARTIAL_SUPERVISION_CONTRACT_PASS',flush=True)

if __name__=='__main__':main()
