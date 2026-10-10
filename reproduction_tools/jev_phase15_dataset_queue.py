"""Freeze a common TRAIN corpus before any Phase XV optimizer update."""
from jev_phase15_common import *
from jev_phase15_queue import pin, run_queue


def main():
    protect(); assert read(REPORTS/'STRUCTURAL_TESTS.json')['status'] == 'PASS'
    for video in TRAIN:
        assert read(OUT/'train_commitment_prefixes_v1'/f'video{video:02d}'/'RESULT.json')['status']=='PASS'
    source=pin('source_dataset_v1')
    jobs=[dict(key=f'commitment_DATASET_TRAIN{video}',script='jev_phase15_collect_dataset.py',
               args=['--video',str(video)],result=str(OUT/'commitment_dataset_v1'/f'video{video:02d}'/'RESULT.json')) for video in TRAIN]
    result=run_queue(source,jobs,'commitment_dataset_queue_v1',max_active=4)
    assert not result['failed'], result['failed']


if __name__=='__main__': main()
