"""Finish preregistered frozen-weight online interventions after small fusers."""
from jev_recovery_common import *
from jev_recovery_controller import run_job
from jev_recovery_publish import publish


def main():
    protect()
    while not (REPORTS/'DIAGNOSTIC_FUSION_RESULT_INDEX.json').exists():
        if unreviewed_failures():raise RuntimeError('Retained diagnostic failure requires review')
        save(OUT/'addendum_online/PROGRESS.json',dict(status='WAITING_MATCHED_FUSER_ONLINE',GPU=2))
        time.sleep(30)
    modes=read(REPORTS/'QUESTION_INTERVENTION_NATIVE_PROTOCOL.json')['modes']
    for mode in modes:
        run_job('R2_XIV_'+mode+'_frozen_native_v1','jev_recovery_question_intervention_online.py',['--mode',mode],2)
        publish('Record complete mutated-state TRAIN16 frozen '+mode+' intervention')
    while not (REPORTS/'R2_XIV_MECHANISM_GATE.json').exists():
        if unreviewed_failures():raise RuntimeError('Retained original failure requires review')
        time.sleep(30)
    run_job('R2_XIV_complete_input_all_pilot_audit_v2','jev_recovery_complete_input_all_pilots.py',gpu=2)
    save(REPORTS/'QUESTION_INTERVENTION_NATIVE_INDEX.json',dict(status='COMPLETE_FROZEN_INTERVENTIONS',
        baseline=ref(REPORTS/'R2_XIV/ONLINE_jev_pilot_SEED20261009_VIDEO16.json'),
        source=[ref(p) for p in (REPORTS/'question_intervention_native').glob('*.json')],
        full_native=True,same_cold_initial_state=True,subsequent_states_policy_dependent=True,
        frozen_weight_interventions_not_retrained_architecture_evidence=True,independent_validation=False))
    publish('Complete additive frozen online intervention evidence and complete-input pilot audit')


if __name__=='__main__':
    try:main()
    except Exception:
        import traceback
        failure('addendum_frozen_online_queue',traceback.format_exc());raise
