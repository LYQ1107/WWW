"""Bounded legal P2 interventions and natural TRAIN collection, no training."""
from jev_phase15_common import *
from jev_phase15_queue import pin, run_queue


def main():
    protect()
    # Genuine long-training qualification still requires all P1 controls and P2.
    for video in DEV:
        assert read(OUT / 'native_replay_v2/multi_question_seed20261009' / f'video{video:02d}' / 'RESULT.json')['status'] == 'PASS'
    source = pin('source_causal_v1'); jobs = []
    for video in DEV:
        for continuation in ['pi_multi_frozen20k', 'pi_fixed_frozen20k']:
            jobs.append({'key': f'{continuation}_video{video}', 'script': 'jev_phase15_causal_commitment.py',
                'args': ['--video', str(video), '--continuation', continuation],
                'result': str(OUT / 'causal_commitment_v1' / continuation / f'video{video:02d}' / 'RESULT.json')})
    for video in TRAIN:
        jobs.append({'key': f'collect_TRAIN{video}', 'script': 'jev_phase15_collect_train.py',
                     'args': ['--video', str(video)],
                     'result': str(OUT / 'train_commitment_prefixes_v1' / f'video{video:02d}' / 'RESULT.json')})
    result = run_queue(source, jobs, 'causal_and_TRAIN_collection_queue_v1', max_active=4)
    assert not result['failed'], result['failed']
    jobs = []
    for video in TRAIN:
        for continuation in ['pi_multi_frozen20k', 'pi_fixed_frozen20k']:
            jobs.append({'key': f'{continuation}_TRAIN{video}', 'script': 'jev_phase15_causal_commitment.py',
                'args': ['--video', str(video), '--continuation', continuation],
                'result': str(OUT / 'causal_commitment_v1' / continuation / f'video{video:02d}' / 'RESULT.json')})
    result = run_queue(source, jobs, 'TRAIN_causal_commitment_queue_v1', max_active=4)
    assert not result['failed'], result['failed']


if __name__ == '__main__': main()
