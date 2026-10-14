"""Actual group input -> identity output latency including current image reading."""
import argparse,time,numpy as np,torch
from common import *

def main():
    p=argparse.ArgumentParser();p.add_argument('--video-id',type=int,default=1);p.add_argument('--max-frames',type=int);p.add_argument('--tag',default='online_B0');p.add_argument('--feature-weight',type=float,default=0.)
    p.add_argument('--geometry-weight',type=float,default=0.);p.add_argument('--geometry-method',default='sift')
    p.add_argument('--backbone',choices=['vggt','dinov2','gmt'],default='vggt');p.add_argument('--layer',type=int,default=17);p.add_argument('--projection-dim',type=int,default=-1);p.add_argument('--calibration',default='');p.add_argument('--pooling',default='roi_align')
    p.add_argument('--interval',type=int,default=10);p.add_argument('--policy',choices=['fixed','every','random','motion','identity','joint'],default='')
    p.add_argument('--call-budget',type=int,default=20);p.add_argument('--budget-window',type=int,default=100);p.add_argument('--min-interval',type=int,default=2);p.add_argument('--offset',type=int,default=0);p.add_argument('--scheduler-seed',type=int,default=20261014);p.add_argument('--head-path',default='');p.add_argument('--max-residual',type=float,default=1.);p.add_argument('--async-gpu',default='');p.add_argument('--async-timeout',type=float,default=5.);p.add_argument('--total-call-budget',type=int,default=-1);p.add_argument('--validate-reversed-vggt-order',action='store_true');p.add_argument('--reuse-target-features',action='store_true');p.add_argument('--validate-gmt-camera-index',action='store_true');p.add_argument('--trace-association',action='store_true');a=p.parse_args()
    out=RUNTIME/'online'/a.tag
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);sys.path[:0]=[str(RUNTIME/'sources/vggt'),str(RUNTIME/'sources/dinov2')]
    if a.validate_reversed_vggt_order:
        if a.backbone!='vggt' or a.async_gpu:raise ValueError('swap validation requires synchronous VGGT')
        from gtr.geometry.vggt_adapter import FrozenVGGT
        from gtr.geometry.view_order_validation import reverse_and_restore
        original_extract=FrozenVGGT.extract
        def swapped_extract(self,images,boxes,*args,**kwargs):
            return reverse_and_restore(lambda x,b,*ar,**kw:original_extract(self,x,b,*ar,**kw),images,boxes,*args,**kwargs)
        # Local to this validation process; production source/model defaults
        # and GMT input camera indexing remain unchanged.
        FrozenVGGT.extract=swapped_extract
    default_calibration={'vggt':'calibration_l17_roi_sym_train_v1.json','dinov2':'calibration_dinov2_roi_train_v1.json','gmt':'calibration_gmt_reid_absolute_train_v1.json'}[a.backbone]
    opts=dict(ENABLED=a.feature_weight!=0 or a.geometry_weight!=0,FEATURE_ENABLED=True,GEOMETRY_ENABLED=True,FEATURE_WEIGHT=a.feature_weight,GEOMETRY_WEIGHT=a.geometry_weight,
        FEATURE_LAYERS=[a.layer] if a.backbone=='vggt' else [-1],FEATURE_BACKBONE=a.backbone,PROJECTION_DIM=a.projection_dim,POOLING=a.pooling,SYMMETRIZE_VIEW_ORDER=a.backbone=='vggt',
        CALIBRATION_PATH=a.calibration or str(ROOT/'manifests'/default_calibration),VGGT_CHECKPOINT=str(RUNTIME/'weights/vggt_1b.pt'),DINOV2_CHECKPOINT=str(RUNTIME/'weights/dinov2_vitl14_reg4.pth'),
        ASYNC_WORKER=bool(a.async_gpu),ASYNC_GPU=a.async_gpu,ASYNC_TIMEOUT=a.async_timeout,VGGT_SOURCE=str(RUNTIME/'sources/vggt'),HEAD_PATH=a.head_path,MAX_RESIDUAL=a.max_residual,GEOMETRY_METHOD=a.geometry_method,UPDATE_INTERVAL=a.interval,UPDATE_OFFSET=a.offset,SPARSE_UPDATE_ENABLED=bool(a.policy),UPDATE_POLICY=a.policy or 'fixed',
        COMPUTE_BUDGET=a.call_budget,TOTAL_CALL_BUDGET=a.total_call_budget,BUDGET_WINDOW=a.budget_window,MIN_UPDATE_INTERVAL=a.min_interval,SCHEDULER_SEED=a.scheduler_seed,REUSE_TARGET_FEATURES=a.reuse_target_features)
    cfg,model=load_gmt(geo_opts=opts)
    association_trace=None
    if a.trace_association:
        from gtr.geometry.association_trace import AssociationTrace
        association_trace=AssociationTrace(model);association_trace.install()
    camera_hook=None
    if a.validate_gmt_camera_index:
        from gtr.geometry.camera_encoding_validation import install_single_image_camera_control
        camera_hook=install_single_image_camera_control(model)
    from gtr.geometry.online_runner import CausalGMT
    from gtr.evaluation.mot_evaluation import custom_instances_to_coco_json
    rows=video_records(train_data(),a.video_id,a.max_frames);groups={}
    for row in rows:groups.setdefault(row['frame_id'],[]).append(row)
    seed_all(20261014);tracker=CausalGMT(model);predictions=[];latency=[];start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    try:
        model.geo_runtime.prepare_async()
        for timestamp,records in sorted(groups.items()):
            torch.cuda.synchronize();t=time.perf_counter();mapped=map_records(cfg,records);outputs=tracker.step(mapped)
            # CPU serialization constitutes externally available identity output.
            for record,output in zip(records,outputs):predictions.extend(custom_instances_to_coco_json(output['instances'].to('cpu'),record['image_id']))
            torch.cuda.synchronize();latency.append(time.perf_counter()-t)
    finally:model.geo_runtime.close_async()
    total=time.perf_counter()-start;path=out/'predictions.json';save_json(path,predictions)
    schedule_path=out/'schedule_trace.json';save_json(schedule_path,model.geo_runtime.schedule_trace)
    if association_trace is not None:
        association_trace.close();save_json(out/'association_trace.json',association_trace.records)
    async_report=model.geo_runtime.async_report()
    if async_report is not None:save_json(out/'async_trace.json',async_report)
    report=provenance();report.update(causal_online=True,full_video=a.max_frames is None,video_id=a.video_id,synchronized_groups=len(groups),images=len(rows),
        speed_eligible=not (a.validate_reversed_vggt_order or a.validate_gmt_camera_index or a.trace_association),validation_reversed_vggt_order=a.validate_reversed_vggt_order,
        validation_corrected_gmt_camera_index=a.validate_gmt_camera_index,
        camera_order_validation_scope='encoder images/ROI inputs reversed before each call; pooled vectors, patch maps and camera pose rows restored to original physical camera IDs; GMT input/association ordering canonical and unchanged',
        protocol='causal GMT initialization/GTA/memory, no future MIN_TRACK_LEN filtering or delay_cls; current/past timestamps only',
        feature_weight=a.feature_weight,seconds=total,camera_image_fps=len(rows)/total,synchronized_group_fps=len(groups)/total,
        input_to_identity_p50=float(np.percentile(latency,50)),input_to_identity_p95=float(np.percentile(latency,95)),latencies_seconds=latency,
        peak_gpu_bytes=torch.cuda.max_memory_allocated(),peak_reserved_gpu_bytes=torch.cuda.max_memory_reserved(),geo_calls=model.geo_runtime.calls,vggt_calls=model.geo_runtime.vggt_calls,
        predictions_sha256=sha256(path),setup_time_included=False,first_geometry_model_load_included=True,geo_options=opts,
        dinov2_calls=model.geo_runtime.dinov2_calls,geometry_valid_calls=model.geo_runtime.geometry_valid_calls,geometry_invalid_calls=model.geo_runtime.geometry_invalid_calls,
        geo_failures=model.geo_runtime.failed,motion_failures=model.geo_runtime.motion_failures,geo_seconds=model.geo_runtime.seconds,
        budget_unit=model.geo_runtime.budget_unit,planned_update_cost=model.geo_runtime.update_cost,schedule_trace=str(schedule_path),schedule_trace_sha256=sha256(schedule_path),
        async_worker=async_report,async_startup_included=bool(a.async_gpu),async_shutdown_included=bool(a.async_gpu),
        total_vggt_calls=model.geo_runtime.vggt_calls+(async_report['encoder_calls'] if async_report else 0),
        encoder_call_scope='extract invocations including failed invocations; failed model loading has no forward invocation',
        target_feature_cache=model.geo_runtime.target_cache_report(),
        association_trace=(dict(path=str(out/'association_trace.json'),sha256=sha256(out/'association_trace.json'),scope='read-only same-live-history counterfactual GTA/Hungarian diagnostic; original baseline run comparison is separate; tensor synchronization means speed-ineligible') if association_trace is not None else None),
        call_budget_scope='rolling cap on planned work, conservative on failed/empty attempts; actual encoder calls reported separately; equal cap does not guarantee equal realized calls')
    if a.head_path:report.update(head_checkpoint_sha256=sha256(a.head_path),head_projection_dimensions=model.geo_runtime.head_checkpoint['settings']['output_dim']*(2 if a.backbone=='vggt' else 1))
    save_json(out/'manifest.json',report);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
