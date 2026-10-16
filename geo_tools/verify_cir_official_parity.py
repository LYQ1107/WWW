"""Original official sliding versus refactored official versus causal shared core.

Small actual image prefix is deliberately mapped as a whole here, reproducing
the official mapper. This is engineering qualification, not an online metric.
"""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import time
from types import MethodType
import torch
from common import ROOT,load_gmt,map_records,train_data,video_records,seed_all,save_json,sha256,CHECKPOINT
from test_cir_native_commit import original_method
from verify_cir_off_parity import FullScoreTrace,decisions,rng_digest
from gtr.geometry.online_runner import CausalGMT,clone_instances
from gtr.modeling.meta_arch import gtr_rcnn as native
from gtr.modeling.meta_arch.custom_rcnn import CustomRCNN
from gtr.evaluation.mot_evaluation import custom_instances_to_coco_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',type=int,default=4)
    p.add_argument('--groups',type=int,default=40);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.scene not in range(1,25):raise ValueError('TRAIN only')
    if a.out.exists():raise FileExistsError(a.out)
    a.out.mkdir(parents=True)
    cfg,model=load_gmt();stock_min=model.min_track_len;model.min_track_len=0
    rows=video_records(train_data(),a.scene,max_frames=a.groups)
    frame_rows=sorted(rows,key=lambda r:(r['frame_id'],r['view_id']))
    results={};start=time.monotonic()
    for arm in ('ORIGINAL_OFFICIAL','CIR_OFF_OFFICIAL','CIR_KEEP_OFFICIAL','CIR_OFF_CAUSAL'):
        seed_all(20261014);model.cir_controller.enabled=arm=='CIR_KEEP_OFFICIAL'
        model.run_global_tracker_plus=MethodType(original_method() if arm=='ORIGINAL_OFFICIAL' else native.GTRRCNN.run_global_tracker_plus,model)
        directory=a.out/arm;directory.mkdir();trace=FullScoreTrace(model,directory);trace.install()
        predictions=[];t=time.monotonic()
        try:
            if arm=='CIR_OFF_CAUSAL':
                tracker=CausalGMT(model)
                for frame in sorted({r['frame_id'] for r in rows}):
                    group=[r for r in rows if r['frame_id']==frame]
                    output=tracker.step(map_records(cfg,group))
                    for row,item in zip(group,output):
                        predictions.extend(custom_instances_to_coco_json(item['instances'].to('cpu'),row['image_id']))
            else:
                batch=map_records(cfg,rows)
                with torch.inference_mode():output,_=model(batch)
                for row,item in zip(frame_rows,output):
                    predictions.extend(custom_instances_to_coco_json(item['instances'].to('cpu'),row['image_id']))
            torch.cuda.synchronize()
            last_rng=rng_digest()
        finally:trace.close()
        count=Counter(r['track_id'] for r in predictions)
        stock=[r for r in predictions if count[r['track_id']]>=stock_min]
        results[arm]=dict(raw=predictions,stock=stock,rng=last_rng,decisions=decisions(directory),seconds=time.monotonic()-t)
        save_json(directory/'raw_predictions.json',predictions);save_json(directory/'stock_predictions.json',stock)
        print(json.dumps(dict(arm=arm,raw_detections=len(predictions),stock_detections=len(stock),seconds=results[arm]['seconds'])),flush=True)
    comparisons={arm:{k:results[arm][k]==results['ORIGINAL_OFFICIAL'][k] for k in ('raw','stock','rng','decisions')} for arm in results if arm!='ORIGINAL_OFFICIAL'}
    report=dict(status='PASS_REAL_PREFIX_OFFICIAL_CAUSAL' if all(all(v.values()) for v in comparisons.values()) else 'FAIL_NO_ADMISSION',
        scene=a.scene,groups=a.groups,images=len(rows),full_video=len(rows)==len(video_records(train_data(),a.scene)),
        comparisons=comparisons,raw_protocol='MIN_TRACK_LEN=0 after tracking; fixed stock50 reconstructed by identical global occurrence count, same for allarms',
        official_entry='actual model.forward/sliding_inference_GMT, ordinary official whole-prefix mapper; not test_net complete-video result',
        checkpoint_sha256=sha256(CHECKPOINT),GT_in_inference=False,
        source_sha256_at_import=__import__('verify_cir_off_parity').SOURCE_AT_IMPORT,
        this_script_sha256=sha256(__file__),wall_seconds=time.monotonic()-start,
        camera_input_unchanged=True,physical_time='native frame indices; wall delay not inferred')
    save_json(a.out/'RESULT.json',report);print(json.dumps(report),flush=True)
    if report['status'].startswith('FAIL'):raise RuntimeError('Official/causal parity failed')


if __name__=='__main__':main()
