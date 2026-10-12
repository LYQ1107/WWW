"""Complete immutable references and solver/rule controls without new training."""
import argparse
import traceback
from jev_oneweek_common import *
from jev_oneweek_controller import run_job
from jev_oneweek_publish import publish


def main(gpu,arms,seeds):
    protect()
    for arm in arms:
        chosen=[20261009] if arm in ['cosine','XV_V3_reference','base_single','base_repeat','base_rule','edge_single','edge_repeat','edge_rule'] else seeds
        for seed in chosen:
            completed=OUT/'matlab_v1/formal'/arm/f'seed{seed}'/'17_18_19/RESULT.json'
            if completed.exists():
                assert read(completed)['status']=='COMPLETE'
                for video in DEV:
                    assert read(OUT/'online_v1/formal'/arm/f'seed{seed}'/f'video{video:02d}/RESULT.json')['status']=='COMPLETE'
                print('SKIP_COMPLETE_NATIVE_MATLAB_BLOCK',arm,seed,flush=True);continue
            for video in DEV:
                path=OUT/'online_v1/formal'/arm/f'seed{seed}'/f'video{video:02d}'/'RESULT.json'
                if path.exists():assert read(path)['status']=='COMPLETE';continue
                run_job(f'online_formal_{arm}_seed{seed}_video{video}',
                    [PYTHON,str(ROOT/'reproduction_tools/jev_oneweek_online.py'),'--arm',arm,
                     '--phase','formal','--seed',str(seed),'--video',str(video)],gpu)
            publish(f'Record complete native {arm} reference/control seed{seed}')
            run_job(f'MATLAB_formal_{arm}_seed{seed}',
                [PYTHON,str(ROOT/'reproduction_tools/jev_oneweek_matlab.py'),'--arm',arm,
                 '--phase','formal','--seed',str(seed),'--videos','17','18','19'])
            publish(f'Record official complete pooled metrics for {arm} seed{seed}')
    print('CONTROL_QUEUE_COMPLETE',arms,seeds,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--arms',nargs='+',required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=SEEDS);a=p.parse_args()
    try:main(a.gpu,a.arms,a.seeds)
    except Exception:
        failure('control_queue',traceback.format_exc());publish('Retain control-queue failure for engineering diagnosis');raise
