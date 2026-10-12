"""Reuse canonical fuser jobs on an idle project GPU; never add experiments."""
from jev_recovery_common import *
from jev_recovery_controller import run_job
from jev_recovery_publish import publish


if __name__=='__main__':
    protect()
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    gpu=2
    for index in [6,5,2]:
        uuid=subprocess.check_output(['nvidia-smi','-i',str(index),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
        if not any(line.startswith(uuid+',') for line in processes.splitlines()):
            gpu=index;break
    print('CANONICAL_FUSION_ADDITIONAL_QUEUE_GPU',gpu,flush=True)
    for kind in ['independent_readers','mlp','attention']:
        assert read(OUT/'diagnostic_fusion_v2'/kind/'tiny/RESULT.json')['engineering_pass']
        run_job(f'R2_FUSION_{kind}_pilot_v2','jev_recovery_fusion_experiment.py',['--kind',kind,'--phase','pilot'],gpu)
        assert read(OUT/'diagnostic_fusion_v2'/kind/'pilot/RESULT.json')['engineering_pass']
        publish('Complete canonical '+kind+' final-fusion pilot on GPU'+str(gpu)+'; existing queue reuses locked result')
