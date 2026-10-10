"""Bounded pinned actors after actual native unconditional parity passes."""
from jev_phase16_common import *
from jev_phase16_queue import pin
from jev_phase16_gpu_queue import run_stack_queue


def main():
    protect();storage_guard()
    assert read(REPORTS/'NATIVE_MEMORY_PARITY.json')['status']=='PASS_EARLIEST_NATIVE_CONTRACT'
    source=pin('source_P2_P3');jobs=[]
    for video in TRAIN:
        for arm in 'ABCD':
            jobs.append(dict(key=f'factorial_{arm}_video{video:02d}',script='jev_phase16_memory_factorial.py',
                args=['--video',str(video),'--arm',arm],result=str(OUT/'P3_factorial'/arm/f'video{video:02d}/RESULT.json')))
    for video in TRAIN+DEV:
        jobs.append(dict(key=f'shadow_video{video:02d}',script='jev_phase16_admission_shadow.py',
            args=['--video',str(video)],result=str(OUT/'P2_shadow'/f'video{video:02d}/RESULT.json')))
    result=run_stack_queue(source,jobs,'P2_P3_queue',max_active=6)
    assert result['status']=='COMPLETE',result['failed']


if __name__=='__main__':main()
