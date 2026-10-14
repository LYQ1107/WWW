"""Actual group input -> identity output latency including current image reading."""
import argparse,time,numpy as np,torch
from common import *

def main():
    p=argparse.ArgumentParser();p.add_argument('--video-id',type=int,default=1);p.add_argument('--max-frames',type=int);p.add_argument('--tag',default='online_B0');p.add_argument('--feature-weight',type=float,default=0.);a=p.parse_args()
    out=RUNTIME/'online'/a.tag
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);sys.path.insert(0,str(RUNTIME/'sources/vggt'))
    opts=dict(ENABLED=a.feature_weight!=0,FEATURE_ENABLED=True,FEATURE_WEIGHT=a.feature_weight,FEATURE_LAYERS=[17],CALIBRATION_PATH=str(ROOT/'manifests/calibration_l17_roi_sym_train_v1.json'),VGGT_CHECKPOINT=str(RUNTIME/'weights/vggt_1b.pt'))
    cfg,model=load_gmt(geo_opts=opts)
    from gtr.geometry.online_runner import CausalGMT
    from gtr.evaluation.mot_evaluation import custom_instances_to_coco_json
    rows=video_records(train_data(),a.video_id,a.max_frames);groups={}
    for row in rows:groups.setdefault(row['frame_id'],[]).append(row)
    seed_all(20261014);tracker=CausalGMT(model);predictions=[];latency=[];start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    for timestamp,records in sorted(groups.items()):
        torch.cuda.synchronize();t=time.perf_counter();mapped=map_records(cfg,records);outputs=tracker.step(mapped)
        # CPU serialization constitutes externally available identity output.
        for record,output in zip(records,outputs):predictions.extend(custom_instances_to_coco_json(output['instances'].to('cpu'),record['image_id']))
        torch.cuda.synchronize();latency.append(time.perf_counter()-t)
    total=time.perf_counter()-start;path=out/'predictions.json';save_json(path,predictions)
    report=provenance();report.update(causal_online=True,full_video=a.max_frames is None,video_id=a.video_id,synchronized_groups=len(groups),images=len(rows),
        protocol='causal GMT initialization/GTA/memory, no future MIN_TRACK_LEN filtering or delay_cls; current/past timestamps only',
        feature_weight=a.feature_weight,seconds=total,camera_image_fps=len(rows)/total,synchronized_group_fps=len(groups)/total,
        input_to_identity_p50=float(np.percentile(latency,50)),input_to_identity_p95=float(np.percentile(latency,95)),latencies_seconds=latency,
        peak_gpu_bytes=torch.cuda.max_memory_allocated(),peak_reserved_gpu_bytes=torch.cuda.max_memory_reserved(),geo_calls=model.geo_runtime.calls,vggt_calls=model.geo_runtime.vggt_calls,
        predictions_sha256=sha256(path),setup_time_included=False,first_geometry_model_load_included=True)
    save_json(out/'manifest.json',report);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
