#!/usr/bin/env python3
"""Run one preregistered TRAIN-only R3 state-fix replay after R2 audits.

No optimizer and no TEST path. Wait for the original diagnostic GPU to be idle;
never signal, stop or restart a healthy worker.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def compare_train_outputs(r2,r3):
    import numpy as np
    from types import SimpleNamespace
    from worldmot.data.visiontrack import VisionTrack
    from worldmot.training.world_object_identity_targets import match_train_identities
    ds=VisionTrack('/data/DATASETS/TRACKING/JDE/VisionTrack',split='train',load_annotations=True)
    total=Counter();scenes={}
    for scene in [4,8,12,16,20,24]:
        counts=Counter();prior=[{},{}]
        with (r2/f'scene_{scene:02d}/tracks.jsonl').open() as left,(r3/f'scene_{scene:02d}/tracks.jsonl').open() as right:
            for pair in ds.synchronized_pairs(scene):
                outputs=[json.loads(next(left)),json.loads(next(right))]
                objects=[{(int(cam),x['detection_id']):x for cam,values in o['tracks_2d_by_camera'].items() for x in values} for o in outputs]
                if outputs[0]['timestamp']!=pair.frame_id or outputs[1]['timestamp']!=pair.frame_id or set(objects[0])!=set(objects[1]):raise ValueError('TRAIN_COMPARISON_CANDIDATES_OR_TIMESTAMPS_DIFFER')
                keys=sorted(objects[0]);obs=[]
                for key in keys:
                    if objects[0][key]['bbox_xywh']!=objects[1][key]['bbox_xywh']:raise ValueError('TRAIN_COMPARISON_BOX_CHANGED')
                    obs.append(SimpleNamespace(key=key,camera_id=key[0],bbox_xywh=np.asarray(objects[0][key]['bbox_xywh'])))
                labels,_=match_train_identities(obs,ds.ground_truth_pair(pair))
                for a,key in enumerate(keys):
                    if labels[a] is None:continue
                    temporal=[]
                    for method in range(2):
                        label_key=(key[0],labels[a]);gid=objects[method][key]['global_id']
                        temporal.append(None if label_key not in prior[method] else gid==prior[method][label_key])
                        prior[method][label_key]=gid
                    if all(x is not None for x in temporal):
                        counts['temporal_events']+=1
                        counts['temporal_R3_corrections']+=int(temporal==[False,True]);counts['temporal_R3_regressions']+=int(temporal==[True,False])
                    for b in range(a+1,len(keys)):
                        if labels[b] is None or key[0]==keys[b][0]:continue
                        same=labels[a]==labels[b];kind='same_person' if same else 'different_person'
                        correct=[(obj[key]['global_id']==obj[keys[b]]['global_id'])==same for obj in objects]
                        counts[kind+'_events']+=1
                        counts[kind+'_R3_corrections']+=int(correct==[False,True]);counts[kind+'_R3_regressions']+=int(correct==[True,False])
                counts['paired_timestamps']+=1
            if left.readline() or right.readline():raise ValueError('TRAIN_COMPARISON_EXTRA_FRAMES')
        total.update(counts);scenes[str(scene)]={'original_GT_legal':scene in [4,8,12],'counts':dict(counts)}
    return {'scope':'All six fixed TRAIN VAL scenes; clean-label diagnostics, not TEST/official metrics',
        'counts':dict(total),'per_scene':scenes,'threshold_or_budget_selection':False}


def main():
    p=argparse.ArgumentParser();p.add_argument('--r2-root',type=Path,required=True);p.add_argument('--r3-root',type=Path,required=True);a=p.parse_args()
    registration=json.loads((a.r3_root/'registration.json').read_text())
    gpu=registration['GPU_uuid'];method='FULL_WORLD_OBJECT_V3'
    r2_audit=a.r2_root/'diagnostics/retained_TRAIN_VAL'/method
    while True:
        ready=(r2_audit/'epoch_04/attribution.json').is_file()
        used=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
        if ready and gpu not in used:break
        time.sleep(10)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,PYTHONPATH='worldmot:worldmot/analysis',PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2')
    out=a.r3_root/'TRAIN_VAL_epoch01'
    cmd=[sys.executable,'-u',str(Path(__file__).with_name('worldobject_v3_spatial_temporal_attribution.py')),
        '--run-root',str(a.r3_root),'--method',method,'--checkpoint',registration['checkpoint'],
        '--output',str(out),'--device','cuda:0']
    with (a.r3_root/'TRAIN_replay.log').open('x') as stream:
        process=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,env=env)
        (a.r3_root/'worker.json').write_text(json.dumps({'pid':process.pid,'time':time.time(),'GPU':gpu,'training':False},indent=2)+'\n')
        code=process.wait()
    if code:raise RuntimeError('R3_TRAIN_ONLY_REPLAY_FAILED_LOG_PRESERVED')
    result=compare_train_outputs(r2_audit/'epoch_01',out)
    result.update({'status':'COMPLETE_TRAIN_ONLY_PAIRED_DIAGNOSTIC','checkpoint_sha256':registration['checkpoint_sha256'],
        'R2_source_sha256':registration['R2_source_sha256'],'R3_source_sha256':registration['R3_source_sha256'],
        'optimizer_created':False,'TEST_read':False,'R3_long_training_started':False})
    (a.r3_root/'TRAIN_only_paired_attribution.json').write_text(json.dumps(result,indent=2)+'\n')
    repo=Path(__file__).resolve().parents[2]
    cmd=['/home/liuyeqiang/anaconda3/bin/python',str(repo/'worldmot/scripts/evaluate_fullvideo.py'),
        '--authorization',str(a.r3_root/'authorization_v3.json'),'--dataset-root','/data/DATASETS/TRACKING/JDE/VisionTrack',
        '--split','train','--audit',str(a.r3_root.parent/'fulltrain_20261022/audit/data_audit.json'),
        '--predictions',str(out),'--output',str(a.r3_root/'official_TRAIN_VAL'),'--scenes','4','8','12','16','20','24',
        '--method-id',method,'--skip-matlab']
    with (a.r3_root/'official_TRAIN_VAL.log').open('x') as log:
        subprocess.run(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            env={**env,'CUDA_VISIBLE_DEVICES':'','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'},check=True)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
