"""Three fresh full-video real-image restarts on one GPU, without GT/evaluation logs."""
import argparse
import gc
import numpy as np
import torch
from jev_oneweek_common import *
from jev_oneweek_online import policy
from jev_oneweek_count_params import original_model,original_live_payloads


def main(arm,seed,phase,video):
    protect();assert video in DEV
    from jev_phase13_runtime import build_tracker,cache_inputs,run
    from jev_phase14_latency import prune
    from gtr.modeling.jev_oneweek.native import attach_jevtrack
    from gtr.modeling.jev_native_state import fingerprint
    values,frames,_=cache_inputs(video)
    folder=OUT/'full_video_efficiency_v1'/phase/arm/f'seed{seed}'/f'video{video:02d}'
    folder.mkdir(parents=True,exist_ok=False)
    trials=[];digests=[]
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','')
    uuid=os.environ.get('JEV_EXPECTED_GPU_UUID')
    if uuid is None and visible.isdigit():
        uuid=subprocess.check_output(['nvidia-smi','-i',visible,'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    assert uuid,'Physical timing GPU UUID must be established before measurements'
    for repeat in range(3):
        seed_all(seed)
        if arm=='original_GMT':
            actor=None;checkpoint=ref('/data1/liuyeqiang/WWW/outputs/stage2_single_gpu/model_20000.pth')
            model,cfg=original_model();original_live_payloads(model,values,cfg);executor=None
        else:
            actor,checkpoint=policy(arm,seed,phase)
            model=build_tracker(video,actor,react_learned=False,live=True)
            prune(model);executor=attach_jevtrack(model)
        assert model.jev_perception_cache_reader is None
        if executor is not None:assert executor.observer is None and executor.commit_observer is None
        # Warmup is separate from measurement. Seed/reset the complete actual
        # native run afterwards, preserving real cold-start state semantics.
        with torch.no_grad():warm,_=run(model,values,frames,stop=7)
        del warm;seed_all(seed)
        concurrent_before=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True)
        if uuid:
            other=[line for line in concurrent_before.splitlines() if line.strip().startswith(uuid+',') and int(line.split(',')[1])!=os.getpid()]
            assert not other,'Timing GPU has another active process; preserve this attempted run separately'
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        with torch.no_grad():raw,_=run(model,values,frames)
        torch.cuda.synchronize();seconds=time.perf_counter()-start
        latency=[v['total_stage2_ms'] for v in executor.latency] if executor is not None else []
        digest=fingerprint(raw[:2*frames]);digests.append(digest)
        trial=dict(repeat=repeat,full_scene_frames=frames,camera_payloads=2*frames,
                   elapsed_seconds=seconds,full_two_camera_scene_FPS=frames/seconds,
                   camera_payload_FPS=2*frames/seconds,
                   Stage2_p50_p95_ms=list(map(float,np.quantile(latency[16:],[.5,.95]))) if latency else None,
                   Stage2_scope='complete native executor includes history/model/plans/solver/Bank/Gallery' if latency else 'Original GMT native timing requires separate detailed profile; full FPS measured',
                   peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,
                   peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20,
                   current_prediction_sha=digest,concurrent_GPU_processes_before=concurrent_before,
                   concurrent_GPU_processes_after=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True))
        if uuid:
            other=[line for line in trial['concurrent_GPU_processes_after'].splitlines() if line.strip().startswith(uuid+',') and int(line.split(',')[1])!=os.getpid()]
            assert not other,'A new process acquired the GPU during the timing run'
        trials.append(trial);save(folder/'PROGRESS.json',dict(status='FULL_VIDEO_RESTARTS',trials=trials))
        print('FULL_VIDEO_SCENE_FPS',arm,repeat,trial['full_two_camera_scene_FPS'],flush=True)
        del raw,model,actor,executor;gc.collect();torch.cuda.empty_cache()
    assert len(set(digests))==1,'Fresh restart changed committed raw Instance predictions'
    result=dict(status='COMPLETE',binding=binding('Three fresh complete real-image two-camera native runs'),
                arm=arm,seed=seed,phase=phase,video=video,checkpoint=checkpoint,scene_frames=frames,
                trials=trials,mean_scene_FPS=float(np.mean([t['full_two_camera_scene_FPS'] for t in trials])),
                std_scene_FPS=float(np.std([t['full_two_camera_scene_FPS'] for t in trials])),
                three_fresh_full_restart_prediction_parity=True,hardware=torch.cuda.get_device_name(),
                torch=str(torch.__version__),CUDA=torch.version.cuda,GPU_UUID=uuid,threads=1,warmup_scene_frames=8,
                includes_actual_image_IO_backbone_detector_VFCE_history_assignment_Bank_Gallery=True,
                GT_evaluation_and_journals_in_timed_region=False,cached_perception_hits=0,
                same_common_frontend=arm!='original_GMT',detailed_component_breakdown='PENDING_SEPARATE_INSTRUMENTED_PROFILE')
    save(folder/'RESULT.json',result)
    save(REPORTS/f'EFFICIENCY_{arm}_{phase}_SEED{seed}_VIDEO{video:02d}.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',required=True);p.add_argument('--seed',type=int,default=20261009)
    p.add_argument('--phase',default='formal');p.add_argument('--video',type=int,default=17);a=p.parse_args()
    try:main(a.arm,a.seed,a.phase,a.video)
    except Exception:
        import traceback
        failure('full_video_efficiency',traceback.format_exc());raise
