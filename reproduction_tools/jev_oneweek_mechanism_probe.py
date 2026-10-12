"""Functional mechanism probes on real TRAIN inputs, not tracking-gain claims."""
import argparse
import torch
from jev_oneweek_common import *
from jev_oneweek_assess import load_model
from jev_oneweek_tests import fixture
from jev_oneweek_learning import batch,initialize


def main(arm,seed,phase):
    protect();records=fixture();record=next(r for r in records if len(r['refs'])>=4 and len(r['rows'])>=2)
    x,y=batch([record]);model,checkpoint,_=load_model(arm,seed,phase)
    q=len(record['rows']);k=len(record['refs'])
    examples=[]
    for a,b in [(0,2),(1,3),(0,3),(1,2)]:
        plan=[-1]*q;plan[0]=a;plan[1]=b;examples.append(tuple(plan))
    with torch.no_grad():
        details=model.details(x);values=model.score_plans(details,x,0,examples)
        interaction=float(values[0]+values[1]-values[2]-values[3])
        qdifference=float((details['cross_logits'].sigmoid()-details['stability_logits'].sigmoid()).abs().mean())
        original_cross=details['cross_logits'].clone()
        if model.tasks is not None:
            original=model.tasks.weight.detach().clone()
            model.tasks.weight.copy_(original.flip(0));swapped=model.details(x)
            sensitivity=float((swapped['cross_logits']-original_cross).abs().mean())
            model.tasks.weight.copy_(original)
        else:sensitivity=None
    # This equivalence belongs to the new log-odds Binary family, not the
    # historical Phase VI binary controller with a different feature contract.
    threshold,_=initialize('threshold',seed);binary,_=initialize('binary',seed)
    threshold.eval();binary.load_state_dict(threshold.state_dict(),strict=True);binary.eval()
    with torch.no_grad():
        threshold_values=threshold.details(x);binary_values=binary.details(x)
    equivalent=torch.equal(threshold_values['edge_logits'],binary_values['edge_logits'])
    assert equivalent
    result=dict(status='COMPLETE_FUNCTIONAL_PROBE',binding=binding('Real TRAIN input legal-assignment interaction and typed conditioning probes'),
                arm=arm,checkpoint=checkpoint,key=record['key'],legal_assignment_examples=examples,
                learned_joint_interaction=interaction,interaction_tolerance=1e-6,
                observed_nonadditive_function=abs(interaction)>1e-6,
                same_state_two_question_probability_difference=qdifference,
                same_reader_swapped_question_embedding_sensitivity=sensitivity,
                new_binary_pair_threshold_exact_reparameterization=equivalent,
                original_VI_binary_not_claimed_equivalent=True,
                probes_do_not_establish_accuracy_or_JEV_independent_value=True)
    save(REPORTS/f'MECHANISM_{arm}_{phase}_SEED{seed}.json',result)
    print('MECHANISM_PROBE',interaction,qdifference,sensitivity,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',default='jev');p.add_argument('--seed',type=int,default=20261009);p.add_argument('--phase',default='pilot');a=p.parse_args()
    main(a.arm,a.seed,a.phase)
