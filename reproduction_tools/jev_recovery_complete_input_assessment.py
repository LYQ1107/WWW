"""Isolated inherited assessment with actual recovery batch; retain old metrics."""
import types
import torch
from jev_recovery_common import *
from jev_recovery_learning import load_records, load_model, batch
from jev_oneweek_learning import assess as historical_assess

# The imported decorated historical function resolves batch in its original
# module globals. Clone only the underlying evaluator globals, never mutate
# an inherited module shared by a running original experiment.
raw = historical_assess.__wrapped__
assess = torch.no_grad()(types.FunctionType(raw.__code__,dict(raw.__globals__,batch=batch),
                          raw.__name__,raw.__defaults__,raw.__closure__))


def main():
    protect();torch.set_num_threads(1)
    _,records,sources=load_records(2);rows=[]
    for arm in ['jev','threshold','mlp','attention','f0_mean','mlp_fusion','attention_fusion',
                'fixed_question','cross_only','stability_only','no_interaction','no_shared']:
        p=OUT/'training_R2/XIV'/arm/'seed20261009/pilot/RESULT.json'
        if not p.exists():continue
        actor,checkpoint,training=load_model(arm,20261009,'pilot',2,'XIV')
        final=assess(actor,records,128)
        rows.append(dict(arm=arm,checkpoint=checkpoint,original_result=ref(p),
            original_inherited_assessment=training['final'],complete_recovery_input_assessment=final))
        del actor;torch.cuda.empty_cache()
        save(REPORTS/'COMPLETE_INPUT_ASSESSMENT_AUDIT.json',dict(status='IN_PROGRESS',
            old_results_immutable=True,trained_weights_unchanged=True,new_gradients=0,
            reason='Historical imported assess resolves historical batch: R2 training batch and certified probe are correct, but inherited assessment omits anchor evidence/affinity proposals.',
            sources=sources,rows=rows,independent_validation=False))
        print('COMPLETE_INPUT_ASSESS',arm,final['rank_pair_accuracy'],flush=True)
    report=read(REPORTS/'COMPLETE_INPUT_ASSESSMENT_AUDIT.json');report['status']='COMPLETE_AVAILABLE_PILOT_AUDIT'
    report['missing_arms']=[a for a in ['no_interaction','no_shared'] if not any(r['arm']==a for r in rows)]
    save(REPORTS/'COMPLETE_INPUT_ASSESSMENT_AUDIT.json',report)


if __name__=='__main__':main()
