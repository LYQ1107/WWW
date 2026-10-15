"""TRAIN-only fixed-head CPU/CUDA gradient throughput, zero optimizer updates."""
import argparse
import json
from pathlib import Path
import statistics
import time
import torch
from pirga_indexed_store import sha
from pirga_training_cache import NativeTrainingCache
from pirga_fulltrain_dataset import FullTrainDataset,load_supervision
from cv_pil_v2_model import CandidatePredictor,supervised_losses as original_losses
from pirga_device_losses import supervised_losses
from cv_pil_native_handoff import safe_torch_load


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--cache-sha',required=True);p.add_argument('--admission',type=Path,required=True);p.add_argument('--cpu-probe',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    started=time.monotonic();torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    admission=json.loads(a.admission.read_text());dataset=FullTrainDataset(admission['stores']);labels=load_supervision(admission['supervision']['path'],admission['supervision']['sha256'],dataset)
    cpu_probe=json.loads(a.cpu_probe.read_text());keys=[tuple(v['key']) for v in cpu_probe['CPU_forward_backward_seconds_by_query'] if labels[tuple(v['key'])]['certified_target_indices']]
    cache=NativeTrainingCache(a.cache,a.cache_sha,device='cuda');model=CandidatePredictor().cuda()
    cp=Path('/data1/liuyeqiang/WWW_cvpil_v3_runtime_20261015/p3_matched_step300_seed20261015_v1/C3_step300.pth')
    if sha(cp)!='76e66718cde4ebd73d55b06fbecd3bff21c0205335e6dc4c45ef650ac920208c':raise ValueError('Frozen head changed')
    model.load_state_dict(safe_torch_load(cp)['state_dict']);times=[]
    for key in keys:
        b=cache.batch(key);label=labels[key]
        for repeat in range(3):
            torch.cuda.synchronize();t=time.monotonic();s,h=model(b);rank,pred,count=supervised_losses(s,h,b,label);loss=rank+.5*pred
            model.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);torch.cuda.synchronize()
            if not torch.isfinite(loss):raise ValueError('Nonfinite device probe')
            if repeat:times.append(dict(key=list(key),histories=len(b.histories),seconds=time.monotonic()-t))
    r=dict(status='PASS_TRAIN_ONLY_CUDA_DETERMINISTIC_GRADIENT_PROBE',queries=len(keys),repeats=2,optimizer_updates=0,training=False,DEV_read=False,
           deterministic_algorithms=True,device_name=torch.cuda.get_device_name(),median_step_seconds=statistics.median(v['seconds'] for v in times),
           measurements=times,wall_seconds=time.monotonic()-started,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),
           timing_note='Actual rank+prediction forward/backward+clip, no optimizer step; full process wall additionally charged by parent. Not imageFPS or a trained result.')
    if a.out.exists():raise FileExistsError(a.out)
    a.out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='measurements'},indent=2),flush=True)

if __name__=='__main__':main()
