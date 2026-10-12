"""Finite equal-budget queue: last checkpoints, all seeds, no DEV model selection."""
import argparse
import traceback
from jev_oneweek_common import *
from jev_oneweek_controller import run_job
from jev_oneweek_publish import publish


def main(gpu,arms,seeds):
    protect()
    assert read(REPORTS/'PILOT_NATIVE_SUMMARY.json')['status']=='PARTIAL_COMPLETE_FIXED_SCOPE_CONTROLS'
    for arm in arms:
        for seed in seeds:
            for phase in ['pilot','formal']:
                path=OUT/'training_v1'/arm/f'seed{seed}'/phase/'RESULT.json'
                if path.exists():
                    result=read(path);assert result['status']=='COMPLETE'
                    assert result['engineering_training_pass'];continue
                name=f'{phase}_{arm}_seed{seed}'
                run_job(name,[PYTHON,str(ROOT/'reproduction_tools/jev_oneweek_train.py'),
                              '--arm',arm,'--phase',phase,'--seed',str(seed)],gpu)
                assert read(path)['engineering_training_pass']
                publish(f'Record {arm} seed{seed} {phase} equal-budget actual training')
            for video in [17,18,19]:
                path=OUT/'online_v1/formal'/arm/f'seed{seed}'/f'video{video:02d}'/'RESULT.json'
                if path.exists():assert read(path)['status']=='COMPLETE';continue
                run_job(f'online_formal_{arm}_seed{seed}_video{video}',
                    [PYTHON,str(ROOT/'reproduction_tools/jev_oneweek_online.py'),
                     '--arm',arm,'--phase','formal','--seed',str(seed),'--video',str(video)],gpu)
            publish(f'Record all complete native DEV videos for {arm} seed{seed}')
            run_job(f'MATLAB_formal_{arm}_seed{seed}',
                    [PYTHON,str(ROOT/'reproduction_tools/jev_oneweek_matlab.py'),
                     '--arm',arm,'--phase','formal','--seed',str(seed),'--videos','17','18','19'])
            publish(f'Record pooled official MATLAB for {arm} seed{seed}')
    print('FINITE_QUEUE_COMPLETE',gpu,arms,seeds,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--arms',nargs='+',required=True);p.add_argument('--seeds',type=int,nargs='+',default=SEEDS);a=p.parse_args()
    try:main(a.gpu,a.arms,a.seeds)
    except Exception:
        failure('finite_queue',traceback.format_exc())
        publish('Retain finite-queue failure without discarding a seed or changing science protocol')
        raise
