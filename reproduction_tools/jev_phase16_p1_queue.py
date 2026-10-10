"""Pinned read-only reader trial, bounded six-worker GPU scheduling."""
from jev_phase16_common import *
from jev_phase16_queue import pin
from jev_phase16_gpu_queue import run_stack_queue


def main():
    assert read(REPORTS/'PHASE16_P0_GO_NO_GO.json')['P1_frozen_prototype_trial_qualified']
    source=pin('source_P1');jobs=[dict(key=f'P1_video{v}',script='jev_phase16_prototype_audit.py',
        args=['--video',str(v)],result=str(OUT/'P1'/f'video{v:02d}/RESULT.json')) for v in TRAIN+DEV]
    r=run_stack_queue(source,jobs,'P1_queue');assert not r['failed'],r['failed']
    subprocess.run([PYTHON,str(source/'reproduction_tools/jev_phase16_prototype_results.py')],cwd=source,check=True)


if __name__=='__main__':main()
