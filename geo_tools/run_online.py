"""Actual group input -> identity output latency including current image reading."""
import argparse,time,numpy as np,torch
from common import *

def main():
    p=argparse.ArgumentParser();p.add_argument('--video-id',type=int,default=1);p.add_argument('--max-frames',type=int);p.add_argument('--tag',default='online_B0');p.add_argument('--feature-weight',type=float,default=0.)
    p.add_argument('--geometry-weight',type=float,default=0.);p.add_argument('--geometry-method',default='sift')
    p.add_argument('--backbone',choices=['vggt','dinov2','gmt'],default='vggt');p.add_argument('--layer',type=int,default=17);p.add_argument('--projection-dim',type=int,default=-1);p.add_argument('--calibration',default='');p.add_argument('--pooling',default='roi_align')
    p.add_argument('--interval',type=int,default=10);p.add_argument('--policy',choices=['fixed','every','random','motion','identity','joint'],default='')
    p.add_argument('--call-budget',type=int,default=20);p.add_argument('--budget-window',type=int,default=100);p.add_argument('--min-interval',type=int,default=2);p.add_argument('--offset',type=int,default=0);p.add_argument('--scheduler-seed',type=int,default=20261014);a=p.parse_args()
    out=RUNTIME/'online'/a.tag
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);sys.path[:0]=[str(RUNTIME/'sources/vggt'),str(RUNTIME/'sources/dinov2')]
    default_calibration={'vggt':'calibration_l17_roi_sym_train_v1.json','dinov2':'calibration_dinov2_roi_train_v1.json','gmt':'calibration_gmt_reid_absolute_train_v1.json'}[a.backbone]
    opts=dict(ENABLED=a.feature_weight!=0 or a.geometry_weight!=0,FEATURE_ENABLED=True,GEOMETRY_ENABLED=True,FEATURE_WEIGHT=a.feature_weight,GEOMETRY_WEIGHT=a.geometry_weight,
        FEATURE_LAYERS=[a.layer] if a.backbone=='vggt' else [-1],FEATURE_BACKBONE=a.backbone,PROJECTION_DIM=a.projection_dim,POOLING=a.pooling,SYMMETRIZE_VIEW_ORDER=a.backbone=='vggt',
        CALIBRATION_PATH=a.calibration or str(ROOT/'manifests'/default_calibration),VGGT_CHECKPOINT=str(RUNTIME/'weights/vggt_1b.pt'),DINOV2_CHECKPOINT=str(RUNTIME/'weights/dinov2_vitl14_reg4.pth'),
        GEOMETRY_METHOD=a.geometry_method,UPDATE_INTERVAL=a.interval,UPDATE_OFFSET=a.offset,SPARSE_UPDATE_ENABLED=bool(a.policy),UPDATE_POLICY=a.policy or 'fixed',
        COMPUTE_BUDGET=a.call_budget,BUDGET_WINDOW=a.budget_window,MIN_UPDATE_INTERVAL=a.min_interval,SCHEDULER_SEED=a.scheduler_seed)
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
    schedule_path=out/'schedule_trace.json';save_json(schedule_path,model.geo_runtime.schedule_trace)
    report=provenance();report.update(causal_online=True,full_video=a.max_frames is None,video_id=a.video_id,synchronized_groups=len(groups),images=len(rows),
        protocol='causal GMT initialization/GTA/memory, no future MIN_TRACK_LEN filtering or delay_cls; current/past timestamps only',
        feature_weight=a.feature_weight,seconds=total,camera_image_fps=len(rows)/total,synchronized_group_fps=len(groups)/total,
        input_to_identity_p50=float(np.percentile(latency,50)),input_to_identity_p95=float(np.percentile(latency,95)),latencies_seconds=latency,
        peak_gpu_bytes=torch.cuda.max_memory_allocated(),peak_reserved_gpu_bytes=torch.cuda.max_memory_reserved(),geo_calls=model.geo_runtime.calls,vggt_calls=model.geo_runtime.vggt_calls,
        predictions_sha256=sha256(path),setup_time_included=False,first_geometry_model_load_included=True,geo_options=opts,
        dinov2_calls=model.geo_runtime.dinov2_calls,geometry_valid_calls=model.geo_runtime.geometry_valid_calls,geometry_invalid_calls=model.geo_runtime.geometry_invalid_calls,
        geo_failures=model.geo_runtime.failed,motion_failures=model.geo_runtime.motion_failures,geo_seconds=model.geo_runtime.seconds,
        budget_unit=model.geo_runtime.budget_unit,planned_update_cost=model.geo_runtime.update_cost,schedule_trace=str(schedule_path),schedule_trace_sha256=sha256(schedule_path),
        encoder_call_scope='extract invocations including failed invocations; failed model loading has no forward invocation',
        call_budget_scope='rolling cap on planned work, conservative on failed/empty attempts; actual encoder calls reported separately; equal cap does not guarantee equal realized calls')
    save_json(out/'manifest.json',report);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
