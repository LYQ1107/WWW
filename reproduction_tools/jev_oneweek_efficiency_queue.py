"""Wait for one own GPU queue to finish, then measure full-video FPS serially."""
import argparse
import fcntl
from jev_oneweek_common import *
from jev_oneweek_controller import run_job
from jev_oneweek_publish import publish


def main(gpu,training_log):
    protect();assert gpu==5
    status=OUT/'efficiency_queue/PROGRESS.json'
    while True:
        text=training_log.read_text() if training_log.exists() else ''
        if 'FINITE_QUEUE_COMPLETE' in text:break
        if 'Traceback (most recent call last)' in text:
            raise RuntimeError('Training queue failed; efficiency dependency needs review')
        save(status,dict(status='WAITING_FOR_OWN_FORMAL_GPU5_QUEUE',dependency=str(training_log)))
        time.sleep(20)
    gpu_uuid=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    os.environ['JEV_EXPECTED_GPU_UUID']=gpu_uuid
    for arm in ['parent','cosine','mlp','attention','jev','deepsets','fixed_question','no_shared_recompute','XIII_fixed_reference','XV_V3_reference','original_GMT']:
        folder=OUT/'full_video_efficiency_v1/formal'/arm/'seed20261009/video17'
        if (folder/'RESULT.json').exists():continue
        existing=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True)
        if any(line.strip().startswith(gpu_uuid+',') for line in existing.splitlines()):
            raise RuntimeError('GPU5 acquired by another process; no silent contended FPS measurement')
        save(status,dict(status='MEASURING_COMPLETE_VIDEO_THREE_RESTARTS',arm=arm,GPU=gpu))
        run_job(f'efficiency_{arm}_seed20261009_video17',
                [PYTHON,str(ROOT/'reproduction_tools/jev_oneweek_profile_latency.py'),'--arm',arm,
                 '--phase','formal','--seed','20261009','--video','17'],gpu)
        publish(f'Record three complete real-image same-GPU FPS restarts for {arm}')
        if arm in ['parent','mlp','attention','jev','no_shared_recompute']:
            run_job(f'component_profile_{arm}',
                    [PYTHON,str(ROOT/'reproduction_tools/jev_oneweek_component_profile.py'),'--arm',arm],gpu)
            publish(f'Record separate same-GPU complete-video components for {arm}')
    save(status,dict(status='COMPLETE',GPU=gpu))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--gpu',type=int,default=5)
    parser.add_argument('--training-log',type=Path,default=OUT/'FORMAL_QUEUE_GPU5.log');a=parser.parse_args()
    try:main(a.gpu,a.training_log)
    except Exception:
        import traceback
        failure('efficiency_queue',traceback.format_exc());publish('Retain full-video efficiency queue failure for review');raise
