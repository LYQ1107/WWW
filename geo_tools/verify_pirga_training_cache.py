"""Real TRAIN-only tensor/score parity and bounded CPU throughput, no updates."""
import argparse
import json
from pathlib import Path
import time
import torch
from pirga_training_cache import NativeTrainingCache
from pirga_fulltrain_dataset import FullTrainDataset
from cv_pil_v2_model import CandidatePredictor,tensorize
from cv_pil_native_handoff import safe_torch_load
from pirga_indexed_store import sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--cache-sha',required=True);p.add_argument('--admission',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(1);started=time.monotonic()
    if a.out.exists():raise FileExistsError(a.out)
    cache=NativeTrainingCache(a.cache,a.cache_sha)
    if sha(a.admission)!=cache.manifest['source_admission_sha256']:raise ValueError('Admission differs from cache source')
    admission=json.loads(a.admission.read_text());dataset=FullTrainDataset(admission['stores'])
    checkpoint=Path('/data1/liuyeqiang/WWW_cvpil_v3_runtime_20261015/p3_matched_step300_seed20261015_v1/C3_step300.pth')
    if sha(checkpoint)!='76e66718cde4ebd73d55b06fbecd3bff21c0205335e6dc4c45ef650ac920208c':raise ValueError('Frozen nonzero head changed')
    model=CandidatePredictor();model.load_state_dict(safe_torch_load(checkpoint)['state_dict']);model.eval()
    keys=[]
    # Label-free physical/native call indices, all11 scenes and long contexts.
    for scene in sorted(dataset.scenes):
        calls=[c for c in cache.calls if c['scene']==scene]
        selected=calls[:2]+calls[len(calls)//2:len(calls)//2+2]+sorted(calls,key=lambda c:len(c['histories']),reverse=True)[:2]
        keys.extend(tuple(c['keys'][0]) for c in selected if c['keys'])
    keys=list(dict.fromkeys(keys))
    with torch.no_grad():
        for key in keys:
            normal=tensorize(dataset.actor(key));fast=cache.batch(key)
            for name in ('query','histories','owners','age','cross_view','native_scores'):
                if not torch.equal(getattr(normal,name),getattr(fast,name)):raise ValueError('Exact native batch mismatch '+str((key,name)))
            if normal.destination_camera!=fast.destination_camera:raise ValueError('Physical camera mismatch')
            if not torch.equal(model(normal)[0],model(fast)[0]):raise ValueError('Nonzero learned head score differs')
    # Fixed TRAIN input diagnostics only, no optimizer/model update, no DEV read.
    timed=[]
    for key in keys:
        t=time.monotonic();batch=cache.batch(key)
        if not len(batch.native_scores):continue
        scores,_=model(batch);loss=scores.square().mean();model.zero_grad(set_to_none=True);loss.backward()
        timed.append(dict(key=list(key),histories=len(batch.histories),seconds=time.monotonic()-t))
    result=dict(status='PASS_REAL_ALL11_FAST_INPUT_BITWISE_PARITY',tested_queries=len(keys),scenes=sorted(dataset.scenes),batch_fields_bitwise_equal=True,
        frozen_nonzero_head_scores_bitwise_equal=True,training=False,optimizer_updates=0,DEV_read=False,GT_supervision_loaded=False,GPU_wall_hours=0,
        CPU_forward_backward_seconds_by_query=timed,source_cache_sha256=a.cache_sha,admission_sha256=sha(a.admission),wall_seconds=time.monotonic()-started)
    a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='CPU_forward_backward_seconds_by_query'},indent=2),flush=True)

if __name__=='__main__':main()
