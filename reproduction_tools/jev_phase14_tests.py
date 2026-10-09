"""Test uncertainty semantics and full lawful capacity before training risk policies."""
import torch
from jev_phase14_common import *
from jev_phase14_losses import safe_joint_loss
from jev_phase14_train import withhold,matched_presence,batch
from jev_phase14_losses import objective
from gtr.modeling.jev_phase14.model import ReliableIdentityPolicy
from gtr.modeling.jev_stage2.assignment import lawful_choice

def main():
    torch.set_num_threads(1)
    z=torch.tensor([[[0.,8.,-1.]]],requires_grad=True)
    x={'legal':torch.ones(1,1,2,dtype=torch.bool),'question_mask':torch.ones(1,1,dtype=torch.bool)}
    y={'positive':torch.tensor([[[True,False,False]]]),'known_options':torch.tensor([[[True,False,True]]]),'supervised':torch.ones(1,1,dtype=torch.bool)}
    loss=safe_joint_loss(z,x,y);assert float(loss)==0
    assert lawful_choice(z[0],x['legal'][0])==[1]
    y['known_options'][:]=True
    loss=safe_joint_loss(z,x,y);assert float(loss)>0
    loss.backward();assert z.grad[0,0,1]>0 and z.grad[0,0,0]<0
    joint=lawful_choice(torch.tensor([[5.,4.,-1.],[5.,0.,-1.]]),torch.ones(2,2,dtype=torch.bool))
    assert joint==[1,0],'capacity must be solved jointly, not independent edge argmax'
    m=read(REPORTS/'QUESTION_SUPERVISION_ELIGIBILITY.json')['videos'][1]
    records=torch.load(m['source_DATASET']['path'],map_location='cpu')['records']
    labels=torch.load(m['label_artifact']['path'],map_location='cpu')['labels']
    meta=next(t for t in labels if t['task']==0 and t['withholdable_rows'])
    r=dict(records[meta['source_record_index']]);r.update(availability=meta['availability'],trust=meta['trust'],withholdable_rows=meta['withholdable_rows'])
    row=r['withholdable_rows'][0];original=r['inputs']['detection_visual'].clone();positive=r['positive'].clone();refs=list(r['refs'])
    removed=withhold(r,row)
    assert torch.equal(original,removed['inputs']['detection_visual'])
    assert torch.equal(r['positive'],positive) and r['refs']==refs
    assert not removed['positive'][row,:-1].any() and removed['positive'][row,-1] and removed['availability'][row]==0
    assert all(t not in removed['refs'] for t in removed['withheld_original_refs'])
    matched=matched_presence(r,row)
    if matched is not None:
        assert len(matched['refs'])==len(removed['refs']) and matched['availability'][row]==1
    capacities={}
    for name in ['full','fixed_question','multi_question','set_transformer','motip','shared_mlp']:
        net=ReliableIdentityPolicy(name);capacities[name]=sum(p.numel() for p in net.parameters())
    ratio=max(capacities.values())/min(capacities.values());assert ratio<=1.1,(capacities,ratio)
    torch.manual_seed(20261009)
    network=ReliableIdentityPolicy('multi_question');inputs,targets=batch([r,removed],device='cpu')
    details=network.details(inputs);total,_=objective(details,inputs,targets,'availability_joint');total.backward()
    task_gradient=network.core.task.weight.grad.norm(dim=1).tolist()
    assert task_gradient[1]>0 and task_gradient[2]>0,task_gradient
    try:network(dict(inputs,GT_labels=targets));raise AssertionError('GT input accepted')
    except ValueError:pass
    save(REPORTS/'ACTION_SEMANTICS_TESTS.json',dict(status='PASS',binding=binding(),
        UNKNOWN_zero_cost_not_negative=True,certified_wrong_cost_gradient=True,lawful_capacity_joint=True,
        withholding_input_copy_preserves_source=True,withholding_absence_DEFER_not_NEW=True,
        parameters=capacities,max_min_ratio=ratio,independent_task_gradient_norms=task_gradient,GT_actor_input_rejected=True,
        lifecycle_scope='MATCH DEFER is a stage action; subsequent frozen REACT/START_NEW fallback is explicitly separate and receives no invented labels'))
    print('PHASE14_ACTION_SEMANTICS_PASS',capacities,flush=True)

if __name__=='__main__':main()
