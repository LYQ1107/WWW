"""Build exactly the frozen four-control/full12 matrix after native admission.

Preparation reads code/protocol/gates only, never GT or development scores.
The owned batch supervisor separately enforces the registered resource budget.
"""
import argparse
import json
import re
from pathlib import Path
import subprocess
import sys
from common import ROOT,CHECKPOINT,save_json,sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--gate',type=Path,required=True)
    p.add_argument('--source-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--gpus',type=int,nargs='+',required=True);p.add_argument('--attempt',default='v1');a=p.parse_args()
    if not re.fullmatch(r'v[1-9][0-9]*',a.attempt):raise ValueError('Explicit versioned attempt required')
    if a.out.exists():raise FileExistsError(a.out)
    if len(set(a.gpus))!=len(a.gpus) or not 1<=len(a.gpus)<=4 or any(g<0 for g in a.gpus):raise ValueError('At most4 unique authorized GPUs')
    matrix_path=ROOT/'results/CIR_GMT/P2_FULL_ONLINE_MATRIX.json';matrix=json.loads(matrix_path.read_text())
    gate=json.loads(a.gate.read_text());methods=['guard','margin','threshold','bias2']
    videos=[1,2,3,10,11,14,15,16,18,19,22,23]
    if matrix['required_video_ids']!=videos or matrix['new_methods']!=methods:raise ValueError('Frozen finite matrix drift')
    if gate['status']!='PASS_FULL_VIDEO_NATIVE_OFF_QUALIFICATION' or gate['stage']!='P2_FINITE_CONTROLS':raise ValueError('Native admission required')
    if gate['allowed_methods']!=['off',*methods] or gate['allowed_video_ids']!=videos:raise ValueError('Complete control admission missing')
    source=a.source_root.resolve()
    if subprocess.check_output(['git','status','--porcelain'],cwd=source):raise ValueError('Clean frozen actor source required')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
    for name,digest in gate['current_source_sha256'].items():
        if sha256(source/name)!=digest:raise ValueError('Source drift after actual native qualification')
    a.out.mkdir(parents=True);jobs=[];bindings=[]
    for method in methods:
        runs={}
        for video in videos:
            root=a.out/method/f'scene{video}'
            spec=dict(split='train',method=method,video_ids=[video],qualification=str(a.gate.resolve()),out_root=str(root.resolve()))
            path=a.out/f'{method}_scene{video}_spec.json';save_json(path,spec)
            runs[str(video)]=str(root.resolve()/f'video{video}/manifest.json')
            jobs.append(dict(name=f'{method}_video{video}',maximum_seconds=3600,
                log=str(a.out.resolve()/f'logs/{method}_video{video}.log'),
                command=[sys.executable,'-u','test_net.py','--config-file','configs/VISION_test.yaml','--num-gpus','1',
                    '--cir-run-spec',str(path.resolve()),'MODEL.WEIGHTS',str(CHECKPOINT),'OUTPUT_DIR',str(a.out.resolve()/f'setup/{method}_video{video}')]))
            bindings.append(dict(path=str(path.resolve()),sha256=sha256(path)))
        save_json(a.out/f'{method}_runs.json',dict(required_video_ids=videos,runs=runs))
    allocation=dict(stage='P2_COMPLETE_REGISTERED_DEV12_FOUR_CONTROLS_'+a.attempt.upper(),physical_GPUs=a.gpus,
        maximum_GPU_wall_hours=12.,maximum_stage_wall_seconds=21600,source_root=str(source),source_commit=commit,jobs=jobs,
        estimate='P1 two methods all12 took2.95GPUh; four controls about6GPUh before guard overhead;12GPUh registered bound, no silent video omission')
    save_json(a.out/'ALLOCATION.json',allocation)
    result=dict(status='PREPARED_NOT_RUNNING',allocation_sha256=sha256(a.out/'ALLOCATION.json'),
        frozen_matrix_sha256=sha256(matrix_path),native_gate_sha256=sha256(a.gate),specs=bindings,
        source_commit=commit,required_jobs=48,GT_read=False,metrics_read=False,
        resource_requirement='Register stage credit in independent CIR ledger before batch; this file does not grant credit')
    save_json(a.out/'PREPARATION.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
