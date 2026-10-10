"""Actual complete-image memory-only audit after bounded causal queue exits."""
from jev_phase16_common import *
from jev_phase16_queue import pin
from jev_phase16_gpu_queue import run_stack_queue


def main():
    protect();storage_guard()
    descriptor=read(OUT/'source_P8_native_provenance.json')
    source=Path(descriptor['path'])
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()==descriptor['commit']
    assert read(OUT/'P2_P3_queue/RESULT.json')['status']=='COMPLETE'
    jobs=[dict(key=f'live_video{v:02d}',script='jev_phase16_evaluate.py',args=['--video',str(v)],
        result=str(OUT/'P8_memory_only_live'/f'video{v:02d}/RESULT.json')) for v in DEV]
    jobs.append(dict(key='actual_image_latency',script='jev_phase16_latency.py',args=[],result=str(OUT/'P8_latency_memory_only/RESULT.json')))
    result=run_stack_queue(source,jobs,'P8_queue',max_active=4)
    assert result['status']=='COMPLETE',result['failed']
    subprocess.run([PYTHON,str(ROOT/'reproduction_tools/jev_phase16_matlab.py')],cwd=ROOT,check=True)


if __name__=='__main__':main()
