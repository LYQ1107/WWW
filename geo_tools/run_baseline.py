"""Fresh full-video original GMT, fixed correct frame/view mapping, no perception cache."""
import argparse, time, numpy as np, torch
from common import *

def main():
    p=argparse.ArgumentParser(); p.add_argument('--video-ids', type=int, nargs='+', default=[1]); p.add_argument('--max-frames', type=int); p.add_argument('--tag', default='B0'); p.add_argument('--geo-enabled',action='store_true'); p.add_argument('--geometry-weight',type=float,default=0.); p.add_argument('--feature-weight',type=float,default=0.); p.add_argument('--interval',type=int,default=10); p.add_argument('--calibration',default=''); p.add_argument('--layer',type=int,default=23); args=p.parse_args()
    out=RUNTIME/'baseline'/args.tag
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    sys.path.insert(0,str(RUNTIME/'sources/vggt'))
    geo_opts=dict(ENABLED=args.geo_enabled,GEOMETRY_ENABLED=True,FEATURE_ENABLED=True,GEOMETRY_WEIGHT=args.geometry_weight,FEATURE_WEIGHT=args.feature_weight,UPDATE_INTERVAL=args.interval,CALIBRATION_PATH=args.calibration,FEATURE_LAYERS=[args.layer],VGGT_CHECKPOINT=str(RUNTIME/'weights/vggt_1b.pt'))
    cfg, model=load_gmt(geo_opts=geo_opts); data=train_data(); reports=[]
    from gtr.evaluation.mot_evaluation import align_visiontrack_inputs_to_outputs, custom_instances_to_coco_json
    for vid in args.video_ids:
        seed_all(20261014); torch.cuda.reset_peak_memory_stats(); start=time.perf_counter()
        records=video_records(data,vid,args.max_frames); t=time.perf_counter(); batch=map_records(cfg,records); read_seconds=time.perf_counter()-t
        phase={'detector':0.,'association':0.}; latency=[]
        original_inference=model.inference; original_asso=model.get_asso
        def infer(*a,**kw):
            torch.cuda.synchronize(); t=time.perf_counter(); r=original_inference(*a,**kw); torch.cuda.synchronize(); dt=time.perf_counter()-t
            phase['detector']+=dt; latency.append(dt); return r
        def asso(*a,**kw):
            torch.cuda.synchronize(); t=time.perf_counter(); r=original_asso(*a,**kw); torch.cuda.synchronize(); phase['association']+=time.perf_counter()-t; return r
        model.inference=infer; model.get_asso=asso
        torch.cuda.synchronize(); t=time.perf_counter()
        with torch.inference_mode(): outputs,views=model(batch)
        torch.cuda.synchronize(); inference_seconds=time.perf_counter()-t
        model.inference=original_inference; model.get_asso=original_asso
        rows=[]
        for record, output in zip(align_visiontrack_inputs_to_outputs('VISION_train',records,outputs),outputs):
            rows.extend(custom_instances_to_coco_json(output['instances'].to('cpu'),record['image_id']))
        path=out/f'video_{vid:02d}.json'; save_json(path,rows)
        report=provenance(); report.update(video_id=vid, frames=len(records), synchronized_groups=len(records)//views,
            full_video=args.max_frames is None, causal_online=False, read_preprocess_seconds=read_seconds,
            inference_seconds=inference_seconds, total_seconds=time.perf_counter()-start, phase_seconds=phase,
            peak_gpu_bytes=torch.cuda.max_memory_allocated(), prediction=str(path),prediction_sha256=sha256(path),geo_options=geo_opts,geo_calls=model.geo_runtime.calls,geo_failures=model.geo_runtime.failed,geo_seconds=model.geo_runtime.seconds,
            detection_latency_p50=float(np.percentile(latency,50)), detection_latency_p95=float(np.percentile(latency,95)),
            latency_scope='single-image detector only; NOT end-to-end identity latency')
        report['camera_image_fps']=len(records)/report['total_seconds']; report['synchronized_group_fps']=(len(records)//views)/report['total_seconds']
        save_json(out/f'video_{vid:02d}_manifest.json',report); reports.append(report); print(json.dumps(report),flush=True)
    save_json(out/'manifest.json',reports)
if __name__=='__main__': main()
