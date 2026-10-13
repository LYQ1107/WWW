"""Parallel execution of already registered R2 seed jobs; no added budget/DEV."""
import argparse
from jev_recovery_common import *
from jev_recovery_controller import run_job
from jev_recovery_publish import publish


def main(seed,gpu):
    protect();assert seed in [20261009,20261010]
    gate=read(REPORTS/'R2_XIV_MECHANISM_GATE.json');assert gate['formal_eligible']
    arms=['jev','threshold','mlp','attention','f0_mean','mlp_fusion','attention_fusion',
          'fixed_question','cross_only','stability_only','no_interaction','no_shared']
    uuid=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    assert not any(line.startswith(uuid+',') for line in processes.splitlines()),'Prefer idle GPU at worker launch'
    for arm in arms:
        if seed!=20261009:
            run_job(f'R2_XIV_{arm}_pilot_seed{seed}','jev_recovery_train.py',
                ['--arm',arm,'--phase','pilot','--seed',seed,'--revision',2,'--base','XIV'],gpu)
        else:assert read(OUT/'training_R2/XIV'/arm/f'seed{seed}/pilot/RESULT.json')['engineering_training_pass']
        run_job(f'R2_XIV_{arm}_formal_seed{seed}','jev_recovery_train.py',
            ['--arm',arm,'--phase','formal','--seed',seed,'--revision',2,'--base','XIV'],gpu)
        r=read(OUT/'training_R2/XIV'/arm/f'seed{seed}/formal/RESULT.json')
        assert r['total_updates']==4000 and r['actual_new_updates']==3000 and r['engineering_training_pass']
        publish(f'Record canonical R2 {arm} seed{seed} formal4000 on parallel GPU{gpu}')
    save(OUT/'parallel_formal'/f'seed{seed}.json',dict(status='COMPLETE_EXISTING_REGISTERED_SEED',seed=seed,GPU=gpu,
         no_extra_updates=True,no_DEV_evaluation=True,original_progression_reuses_same_locked_jobs=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--gpu',type=int,required=True);a=p.parse_args()
    try:main(a.seed,a.gpu)
    except Exception:
        import traceback
        failure('parallel_formal_seed_'+str(a.seed),traceback.format_exc());raise
