"""Reuse canonical fuser jobs on currently idle GPU6, never add experiments."""
from jev_recovery_common import *
from jev_recovery_controller import run_job
from jev_recovery_publish import publish


if __name__=='__main__':
    protect()
    uuid=subprocess.check_output(['nvidia-smi','-i','6','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    assert not any(line.startswith(uuid+',') for line in processes.splitlines()),'Prefer idle device at launch'
    for kind in ['independent_readers','mlp','attention']:
        assert read(OUT/'diagnostic_fusion_v2'/kind/'tiny/RESULT.json')['engineering_pass']
        run_job(f'R2_FUSION_{kind}_pilot_v2','jev_recovery_fusion_experiment.py',['--kind',kind,'--phase','pilot'],6)
        assert read(OUT/'diagnostic_fusion_v2'/kind/'pilot/RESULT.json')['engineering_pass']
        publish('Complete canonical '+kind+' final-fusion pilot using free GPU6; existing queue reuses locked result')
