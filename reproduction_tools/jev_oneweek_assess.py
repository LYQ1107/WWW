"""Paired current-association probes; never substitute for full tracking."""
import argparse
import torch
from jev_oneweek_common import *
from jev_oneweek_learning import load_records,initialize,assess


def load_model(arm,seed,phase):
    result=read(OUT/'training_v1'/arm/f'seed{seed}'/phase/'RESULT.json')
    checkpoint=result['checkpoint'];assert sha(checkpoint['path'])==checkpoint['SHA256']
    model,initializer=initialize(arm,seed)
    saved=torch.load(checkpoint['path'],map_location='cuda:0')
    missing,unexpected=model.load_state_dict(saved['new_modules'],strict=False)
    assert not unexpected and all(key.startswith('identity.') for key in missing)
    assert initializer==saved['initializer']
    return model.eval(),checkpoint,result


def main(arm,seed,phase):
    protect();train,validation,sources=load_records()
    baseline,_=initialize(arm,seed);before=assess(baseline,validation,128)
    del baseline;torch.cuda.empty_cache()
    model,checkpoint,training=load_model(arm,seed,phase);after=assess(model,validation,128)
    assert before['counts']['rows']==after['counts']['rows']
    assert before['counts']['plans_generated']==after['counts']['plans_generated']
    result=dict(status='COMPLETE_PAIRED_SAME128_PAYLOAD_PROBE',before=before,after=after,
                checkpoint=checkpoint,source_training=ref(OUT/'training_v1'/arm/f'seed{seed}'/phase/'RESULT.json'),
                correction_note='First Tiny report before used64 while after128. Original retained; this explicitly paired recomputation fixes the comparison without retraining.',
                full_native_tracking_metrics=None,independent_generalization=False)
    save(REPORTS/f'PAIRED_AUDIT_{arm}_{phase}_SEED{seed}.json',result)
    print('PAIRED_PROBE',before['counts'],after['counts'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',default='jev');p.add_argument('--phase',default='tiny');p.add_argument('--seed',type=int,default=20261009);a=p.parse_args()
    main(a.arm,a.seed,a.phase)
