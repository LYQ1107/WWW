"""Actual H64 same-prefix native commit test of frozen final-fusion masks."""
from jev_recovery_common import *
from jev_recovery_learning import load_model
from jev_recovery_question_intervention_online import FrozenIntervention


def main():
    protect()
    os.environ['JEV_RECOVERY_REVISION']='2';os.environ['JEV_RECOVERY_BASE']='XIV'
    import jev_recovery_paired_native as paired
    paired.OUT=OUT/'question_intervention_paired'
    paired.REPORTS=REPORTS/'question_intervention_paired'
    def policy(mode,seed,phase):
        actor,cp,_=load_model('jev',seed,phase,2,'XIV')
        return FrozenIntervention(actor,mode).eval(),cp
    paired.policy=policy
    paired.main(['full','cross_representation_only','persistent_representation_only',
                 'mean_representations','no_joint_plan_residual'],20261009,'pilot',16)


if __name__=='__main__':
    try:main()
    except Exception:
        import traceback
        failure('question_paired_native',traceback.format_exc());raise
