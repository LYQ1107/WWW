"""Sequential phases within one frozen case, no human-dependent runner steps."""
import argparse
from jev_phase13_learning import *

def main(variant,seed,phase):
    if phase=='validation':
        from evaluate_jev_stage2_online import main as evaluate
        for video in VAL:evaluate(video,variant,seed)
    else:
        from train_jev_stage2_onpolicy import rollout,train
        manifest=rollout(variant,seed)
        if manifest['status']!='PASS':raise RuntimeError('Own-state positive support gate failed, fine-tune is forbidden')
        if not (OUT/'onpolicy_training_v1'/variant/f'seed{seed}'/'RESULT.json').exists():train(variant,seed)
        from evaluate_jev_stage2_online import main as evaluate
        for video in VAL:evaluate(video,variant,seed,'onpolicy')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--variant',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--phase',choices=['validation','onpolicy'],required=True);a=p.parse_args();main(a.variant,a.seed,a.phase)
