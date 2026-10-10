"""One evidence-authorized own-state collection, then the final bounded pilot."""
from jev_phase15_common import *
from jev_phase15_queue import pin,run_queue
from jev_phase15_pilot_queue import main as pilot


def main():
    protect();protocol=read(REPORTS/'RESEARCH_VERSION_3_PROTOCOL.json')
    assert protocol['status']=='FROZEN_BEFORE_TRAINING'
    assert read(REPORTS/'PILOT_RESULTS.json')['full_pilot_qualified'] is False
    source=pin('source_own_state_collection_v3')
    jobs=[dict(key=f'own_pilot_state_TRAIN{v}',script='jev_phase15_collect_pilot_onpolicy.py',args=['--video',str(v)],
        result=str(OUT/'onpolicy_pilot_dataset_v3'/f'video{v:02d}'/'RESULT.json')) for v in TRAIN]
    result=run_queue(source,jobs,'onpolicy_pilot_collection_queue_v3',max_active=4);assert not result['failed'],result
    # Source may be pinned later for training, but policy/data manifests must be
    # exact and remain independent of development evaluation.
    pilot(3)


if __name__=='__main__':main()
