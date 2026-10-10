"""Uninstrumented real images plus complete native Stage2 on shared GPU."""
import argparse
import time
import numpy as np
import torch
from jev_phase15_common import *
from jev_phase15_evaluate import load_policy
from jev_phase13_runtime import build_tracker,cache_inputs,run
from gtr.modeling.jev_phase15.native_commit_adapter import attach
from gtr.modeling.jev_native_state import fingerprint


def quantile(values):
    return dict(zip(['p50','p95','max'],map(float,np.quantile(values,[.5,.95,1])))) if values else None


def main(version=3):
    protect();torch.set_num_threads(1);torch.manual_seed(20261009)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    frames=256;values,total,reader=cache_inputs(17);policy,ck,trained=load_policy('F_full',20261009,version,'pilot')
    model=build_tracker(17,policy,react_learned=False,live=True);executor=attach(model)
    assert model.jev_perception_cache_reader is None and executor.observer is None and executor.commit_observer is None
    out=OUT/f'latency_v{version}';out.mkdir(parents=True,exist_ok=True);assert not (out/'RESULT.json').exists()
    samples=[];original=executor.commit
    def commit(*a,**k):
        torch.cuda.synchronize();start=time.perf_counter();value=original(*a,**k);torch.cuda.synchronize()
        samples.append((time.perf_counter()-start)*1000);return value
    executor.commit=commit;trials=[];digests=[]
    for repeat in range(3):
        with torch.no_grad():run(model,values,total,stop=7)
        samples.clear();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        with torch.no_grad():raw,_=run(model,values,total,stop=frames-1)
        torch.cuda.synchronize();seconds=time.perf_counter()-start
        assert len(samples)==2*frames-1
        digest=fingerprint(raw[:2*frames]);digests.append(digest)
        trial=dict(repeat=repeat,scene_frames=frames,camera_payloads=2*frames,seconds=seconds,
            full_two_camera_scene_FPS=frames/seconds,camera_payload_FPS=2*frames/seconds,
            total_native_Stage2_ms=quantile(samples[16:]),peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20)
        trials.append(trial);save(out/'PROGRESS.json',dict(status='RUNNING',trials=trials))
        print('PHASE15_REAL_IMAGE_RUNTIME',version,trial,flush=True)
        del raw
    assert len(set(digests))==1,'restart changed actual committed prediction fields'
    result=dict(status='COMPLETE',binding=binding(seed=20261009,checkpoints=[ck],dataset=ref(ANNOTATIONS),
        evaluator='CUDA-synchronized real image/native runtime, no GT/evaluator/journals',scope='three restarts of actual live DEV17 first256 scene frames; shared GPU, diagnostic pilot'),
        trained=trained,trials=trials,perception_cache_hits=0,strict_original_Stage1=True,
        includes_image_IO_detector_VFCE_history_policy_assignment_Gallery_Bank_posteriors=True,
        total_native_Stage2_includes_adapter_posterior_write=True,warmup_scene_frames=8,
        three_restart_full_Instance_parity=True,raw_SHA256=digests[0],
        full_video_uninstrumented_FPS=None,scope_not_complete_video=True,
        runtime_thresholds_pass=all(t['total_native_Stage2_ms']['p95']<=10 and t['full_two_camera_scene_FPS']>=25 for t in trials),
        concurrent_GPU_processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True),
        hardware=torch.cuda.get_device_name(),pilot_runtime_is_not_formal_tracking_GO=True)
    save(out/'RESULT.json',result);save(REPORTS/'EFFICIENCY.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--version',type=int,default=3);a=p.parse_args();main(a.version)
