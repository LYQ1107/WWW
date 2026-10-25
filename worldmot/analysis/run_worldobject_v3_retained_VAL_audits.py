#!/usr/bin/env python3
"""Read-only serial per-epoch audit on one idle authorized V100; no training."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--method',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    records=json.loads((a.run_root/'audit_snapshots/epoch_archive_index.json').read_text())['records']
    records=sorted((x for x in records if x['method']==a.method and x['usable_as_selection_epoch_checkpoint']),key=lambda x:x['selection_epochs'])
    if [x['selection_epochs'] for x in records]!=[1,2,3,4]:raise ValueError('EXACT_FOUR_RETAINED_SELECTION_EPOCHS_REQUIRED')
    root=a.run_root/'diagnostics/retained_TRAIN_VAL'/a.method;root.mkdir(parents=True,exist_ok=False)
    for row in records:
        epoch=row['selection_epochs'];out=root/f'epoch_{epoch:02d}'
        command=[sys.executable,'-u',str(Path(__file__).with_name('worldobject_v3_spatial_temporal_attribution.py')),
            '--run-root',str(a.run_root),'--method',a.method,'--checkpoint',row['path'],'--output',str(out),'--device',a.device]
        with (root/f'epoch_{epoch:02d}.log').open('x') as stream:
            process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
            (root/'worker.json').write_text(json.dumps({'pid':process.pid,'epoch':epoch,'checkpoint_sha256':row['sha256'],
                'started':time.time(),'training':False},indent=2)+'\n')
            code=process.wait()
        if code:raise RuntimeError(f'RETAINED_AUDIT_FAILED: epoch {epoch}, code {code}, logs preserved')
        report=json.loads((out/'attribution.json').read_text())
        # A numerical mismatch is reported explicitly; no replay is rerun with
        # a new threshold or altered state to force agreement.
        print(json.dumps({'method':a.method,'completed_epoch':epoch,
            'original_diagnostics_exact':report['original_epoch_prediction_diagnostics_exactly_reproduced'],
            'spatial_recoveries':report['counts'].get('correct_selected_spatial_conflict_visual_recoveries',0),
            'new_ID_next_frame':report['counts'].get('correct_spatial_recovery_next_frame_new_ID',0)}),flush=True)


if __name__=='__main__':main()
