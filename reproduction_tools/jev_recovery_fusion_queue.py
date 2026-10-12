"""Finite additive diagnosis queue; original R2/R3 progression is untouched."""
import argparse
from jev_recovery_common import *
from jev_recovery_controller import run_job
from jev_recovery_publish import publish
from jev_recovery_fusion_experiment import KINDS


def main(gpu):
    protect()
    for phase in ['tiny','pilot']:
        for kind in KINDS:
            run_job(f'R2_FUSION_{kind}_{phase}_v1','jev_recovery_fusion_experiment.py',
                ['--kind',kind,'--phase',phase],gpu)
            result=read(OUT/'diagnostic_fusion'/kind/phase/'RESULT.json')
            assert result['engineering_pass'],'Retain failed engineering result; do not auto-retry'
            publish('Record bounded supplementary '+kind+' '+phase+' on identical frozen JEV frontend')
        values=[read(OUT/'diagnostic_fusion'/k/phase/'RESULT.json') for k in KINDS]
        assert len({v['batch_key_digest'] for v in values})==1
        counts=[v['new_parameters'] for v in values];assert max(counts)/min(counts)<1.01
    for kind in KINDS:
        run_job(f'R2_FUSION_{kind}_certified_v1','jev_recovery_fusion_experiment.py',['--kind',kind,'--mode','probe'],gpu)
        publish('Record exact same-state correction of supplementary '+kind+' final fuser')
    for kind in KINDS:
        run_job(f'R2_FUSION_{kind}_TRAIN16_v1','jev_recovery_fusion_experiment.py',['--kind',kind,'--mode','online'],gpu)
        run_job(f'R2_FUSION_{kind}_identity_v1','jev_recovery_fusion_experiment.py',['--kind',kind,'--mode','identity'],None)
        publish('Record complete TRAIN16 native state and identity of supplementary '+kind+' final fuser')
    save(REPORTS/'DIAGNOSTIC_FUSION_RESULT_INDEX.json',dict(status='COMPLETE_BOUNDED_TRAIN16_DIAGNOSTICS',
        sources=[ref(p) for p in (REPORTS/'diagnostic_fusion').glob('*.json')]+[ref(p) for p in (REPORTS/'diagnostic_fusion_native').glob('*.json')],
        frontend='Common frozen R2 JEV Pilot; same auxiliary predictions for all new final fusers',
        pure_ordinary_architecture_advantage_not_tested_here=True,independent_validation=False,
        original_R2_R3_gates_changed=False,major_revision=2,no_DEV_tuning=True))
    publish('Complete bounded final-fusion learning diagnostic with matched ordinary controls')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,default=2);a=p.parse_args()
    try: main(a.gpu)
    except Exception:
        import traceback
        failure('diagnostic_fusion_queue',traceback.format_exc());raise
