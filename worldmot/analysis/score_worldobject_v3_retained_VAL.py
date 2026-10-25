#!/usr/bin/env python3
"""Score retained-epoch full TRAIN VAL videos with unchanged TrackEval.

Only raw-GT-legal TRAIN scenes 4/8/12 contribute; 16/20/24 remain excluded.
No TEST is read, no budget or checkpoint selection is changed.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-root',required=True,type=Path);a=p.parse_args()
    root=a.run_root/'diagnostics/retained_TRAIN_VAL'
    todo={(m,e) for m in ('FULL_WORLD_OBJECT_V3','HEAD_ONLY_NO_GEOMETRY_V3') for e in range(1,5)}
    repo=Path(__file__).resolve().parents[2]
    while todo:
        for method,epoch in sorted(todo):
            folder=root/method/f'epoch_{epoch:02d}'
            attribution=folder/'attribution.json';output=folder/'official_TRAIN_VAL'
            if not attribution.exists():continue
            if output.exists():
                report=json.loads((output/'metrics.json').read_text())
                if 'TrackEval' not in report:raise ValueError('EXISTING_PARTIAL_VAL_EVALUATION_NOT_RESTARTED')
                todo.remove((method,epoch));continue
            command=['/home/liuyeqiang/anaconda3/bin/python',str(repo/'worldmot/scripts/evaluate_fullvideo.py'),
                '--authorization',str(a.run_root/'authorization_v3.json'),'--dataset-root','/data/DATASETS/TRACKING/JDE/VisionTrack',
                '--split','train','--audit',str(a.run_root.parent/'fulltrain_20261022/audit/data_audit.json'),
                '--predictions',str(folder),'--output',str(output),'--scenes','4','8','12','16','20','24',
                '--method-id',method,'--skip-matlab']
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
            with (folder/'official_TRAIN_VAL.log').open('x') as log:
                subprocess.run(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
            report=json.loads((output/'metrics.json').read_text())
            observed=json.loads(attribution.read_text())
            receipt={'schema':'worldobject_v3_retained_epoch_TRAIN_VAL_TrackEval_v1','status':'COMPLETE_TRACK_EVAL_TRAIN_VAL',
                'TRAIN_VAL_scope':[4,8,12,16,20,24],'legal_scored_scenes':[4,8,12],'raw_GT_blocked_scenes':[16,20,24],
                'method':method,'selection_epoch':epoch,'checkpoint_sha256':observed['checkpoint_sha256'],
                'replay_exact_original_diagnostics':observed['original_epoch_prediction_diagnostics_exactly_reproduced'],
                'metrics':report['TrackEval']['combined'],'TEST_read':False,'budget_reselected':False,
                'MATLAB_requested':False,'original_evaluator_report':str(output/'metrics.json')}
            (folder/'TRAIN_VAL_metrics.json').write_text(json.dumps(receipt,indent=2)+'\n')
            print(json.dumps(receipt),flush=True);todo.remove((method,epoch))
        if todo:time.sleep(10)


if __name__=='__main__':main()
