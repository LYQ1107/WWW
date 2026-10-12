"""Measure new controllers with real images and native parameter storage reads."""
import gc
import argparse
import torch
from jev_oneweek_common import *
from jev_oneweek_online import policy
from jev_oneweek_count_params import ParameterReads


def main(arms,tag):
    protect()
    from jev_phase13_runtime import build_tracker,cache_inputs,run
    from jev_phase14_latency import prune
    from gtr.modeling.jev_oneweek.native import attach_jevtrack
    values,frames,_=cache_inputs(12);cases=[]
    for arm in arms:
        seed_all();actor,checkpoint=policy(arm,20261009,'formal')
        model=build_tracker(12,actor,react_learned=False,live=True)
        removed=prune(model)['removed_total'];attach_jevtrack(model)
        counter=ParameterReads(dict(tracker=model,policy=actor))
        with torch.no_grad(),counter:raw,_=run(model,values,frames,stop=1)
        torch.cuda.synchronize();item=counter.report()
        item.update(arm=arm,checkpoint=checkpoint,removed_GTA_parameters=removed,
                    policy_parameters=sum(p.numel() for p in actor.parameters()),
                    new_module_parameters=sum(p.numel() for name,p in actor.named_parameters() if not name.startswith('identity.')),
                    registered_parameters_by_module=dict(tracker=sum(p.numel() for p in model.parameters()),policy=sum(p.numel() for p in actor.parameters())),
                    net_total_vs_original_GMT=item['total_loaded_parameters']-53515309,
                    no_GT_actor_inputs=True)
        cases.append(item)
        print('NEW_ACTUAL_PARAMETERS',arm,item['total_loaded_parameters'],item['actual_forward_active_parameters'],flush=True)
        del raw,model,actor,counter;gc.collect();torch.cuda.empty_cache()
    target='NEW_MODULE_PARAMETER_READS.json' if tag=='main' else 'NEW_MODULE_PARAMETER_READS_EXTRA.json'
    save(REPORTS/target,dict(status='COMPLETE',binding=binding('New trained models actual real-image native parameter reads'),
         cases=cases,hardware=torch.cuda.get_device_name(),partial_FLOPs_not_claimed_complete=True,
         prefix_scope_not_all_gallery_branches=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arms',nargs='+',default=['jev','mlp','attention','deepsets','fixed_question','cross_only'])
    p.add_argument('--tag',choices=['main','extra'],default='main');a=p.parse_args()
    try:main(a.arms,a.tag)
    except Exception:
        import traceback
        failure('new_parameters',traceback.format_exc());raise
