"""Explicit legacy-compatible duplicate-GT diagnostic, excluded from strict gates."""
from jev_phase15_common import *
from jev_phase15_queue import pin, run_queue


def main():
    source=pin('source_duplicate_diagnostic_v2')
    jobs=[dict(key=f'{policy}_TRAIN14_duplicate_diagnostic',script='jev_phase15_causal_commitment.py',
        args=['--video','14','--continuation',policy,'--duplicate-gt-diagnostic'],
        result=str(OUT/'causal_commitment_v2_duplicate_diagnostic'/policy/'video14/RESULT.json'))
        for policy in ['pi_multi_frozen20k','pi_fixed_frozen20k']]
    result=run_queue(source,jobs,'duplicate_GT_diagnostic_queue_v2',max_active=2)
    assert not result['failed'],result['failed']


if __name__=='__main__':main()
