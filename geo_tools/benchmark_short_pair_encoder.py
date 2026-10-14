"""Pinned VGGT/FastVGGT microbenchmark; never an end-to-end GMT speed test."""
import argparse
import time
import traceback
import numpy as np
import torch
from common import *


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--variant',choices=['original','fast'],required=True)
    parser.add_argument('--tag',required=True)
    parser.add_argument('--repeats',type=int,default=5)
    args=parser.parse_args()
    if args.repeats<2:raise ValueError('at least two measured repetitions')
    out=RUNTIME/'encoder_microbench'/args.tag/args.variant
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    source=RUNTIME/'sources'/('vggt' if args.variant=='original' else 'FastVGGT')
    sys.path.insert(0,str(source))
    report=provenance();report.update(variant=args.variant,status='RUNNING',used_cache=True,
        cache_scope='frozen TRAIN detector ROI boxes only; images read and encoder run for every invocation; no dense token cache',
        scope='three fixed TRAIN synchronized two-camera pairs, both view orders; encoder/ROI/optional-camera microbenchmark, not complete GMT FPS',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip(),
        source_python_sha256={str(p.relative_to(source)):sha256(p) for directory in [source/'vggt',source/'merging'] if directory.exists() for p in directory.rglob('*.py')},
        source_dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=source,text=True)),
        precision='outer FP16 autocast on V100; original unmodified Fast source retains its own hard-coded BF16 conversions',
        interface_bridge='enable all heads for strict official checkpoint loading, then disable unused dense/track heads; Fast public patch-dimension setter; map Fast four retained outputs to original indices4/11/17/23',
        weights=str(RUNTIME/'weights/vggt_1b.pt'),weights_sha256=sha256(RUNTIME/'weights/vggt_1b.pt'),rows=[],actual_encoder_calls=0)
    save_json(out/'manifest.json',report)
    setup_start=time.perf_counter()
    try:
        from vggt.models.vggt import VGGT
        from vggt.utils.load_fn import load_and_preprocess_images
        from gtr.geometry.roi_feature_extractor import PadMapping,patch_map,pool_roi,compress
        from feature_metrics import evaluate_scores
        seed_all(20261014)
        model=VGGT(enable_camera=True,enable_point=True,enable_depth=True,enable_track=True)
        model.load_state_dict(torch.load(RUNTIME/'weights/vggt_1b.pt',map_location='cpu'),strict=True)
        model.point_head=None;model.depth_head=None;model.track_head=None
        model=model.to('cuda').eval().requires_grad_(False)
        report['static_model_setup_seconds']=time.perf_counter()-setup_start
        samples_path=RUNTIME/'samples/screen_train_v1/samples.json'
        samples=json.loads(samples_path.read_text())
        if samples['provenance'].get('split')!='train':raise ValueError('TRAIN fixtures required')
        chosen=[next(sample for sample in samples['samples'] if sample['video_id']==video and (video!=8 or sample['frame_id']==262)) for video in [5,8,12]]
        if {s['video_id'] for s in chosen}!={5,8,12}:raise ValueError('fixed TRAIN scene fixture changed')
        report['sample_manifest_sha256']=sha256(samples_path)
        report['samples']=[dict(video_id=s['video_id'],frame_id=s['frame_id'],path=s['path'],sha256=sha256(s['path'])) for s in chosen]
        torch.cuda.reset_peak_memory_stats();metric_items=[]
        def encode(images,boxes):
            if args.variant=='fast':model.update_patch_dimensions(images.shape[-1]//14,images.shape[-2]//14)
            report['actual_encoder_calls']+=1
            with torch.autocast('cuda',dtype=torch.float16):
                tokens,start=model.aggregator(images[None].cuda())
            retained={layer:tokens[index] for index,layer in enumerate([4,11,17,23])} if args.variant=='fast' else {layer:tokens[layer] for layer in [4,11,17,23]}
            if any(value is None for value in retained.values()):raise ValueError('missing retained layer')
            vectors=[compress(pool_roi(patch_map(retained[17][0,view],start,images.shape[-2:]),box.cuda(),'roi_align')[0],0).cpu() for view,box in enumerate(boxes)]
            # Same camera-head FP32 path as our actual feature+camera adapter.
            with torch.autocast('cuda',enabled=False):pose=model.camera_head(tokens)[-1].float().cpu()
            if not all(torch.isfinite(v).all() for v in vectors) or not torch.isfinite(pose).all():raise FloatingPointError('nonfinite encoder result')
            return vectors,pose,dict(output_list_length=len(tokens),patch_start_idx=start,retained_shapes={str(layer):list(value.shape) for layer,value in retained.items()})
        for sample in chosen:
            arrays=np.load(sample['path']);paths=[view['path'] for view in sample['views']]
            boxes=[PadMapping(view['width'],view['height']).boxes(torch.from_numpy(arrays[f'boxes_{index}'])) for index,view in enumerate(sample['views'])]
            timings=[]
            for repeat in range(args.repeats+1):
                torch.cuda.synchronize();started=time.perf_counter()
                images=load_and_preprocess_images(paths,mode='pad')
                forward,pose,info=encode(images,boxes)
                reverse,_,_=encode(images.flip(0),boxes[::-1])
                vectors=[torch.cat([forward[v],reverse[1-v]],dim=1)/2**.5 for v in range(2)]
                torch.cuda.synchronize();seconds=time.perf_counter()-started
                if repeat:timings.append(seconds)
            artifact=out/Path(sample['path']).name
            np.savez_compressed(artifact,view0=vectors[0].numpy().astype(np.float16),view1=vectors[1].numpy().astype(np.float16),pose=pose.numpy())
            metric_items.append(dict(score=(vectors[1]@vectors[0].T).numpy(),ids1=arrays['labels_0'],ids2=arrays['labels_1'],base=arrays['gmt_logits']))
            row=dict(video_id=sample['video_id'],frame_id=sample['frame_id'],measured_pair_seconds=timings,
                     median_two_order_seconds=float(np.median(timings)),p95_two_order_seconds=float(np.percentile(timings,95)),
                     warmup_two_order_calls=2,measured_two_order_calls=2*args.repeats,read_and_preprocess_included=True,
                     artifact=str(artifact),artifact_sha256=sha256(artifact),**info)
            report['rows'].append(row);save_json(out/'manifest.json',report);print(json.dumps(row),flush=True)
        report.update(status='PASS',feature_proxy_metrics=evaluate_scores(metric_items),peak_gpu_bytes=torch.cuda.max_memory_allocated(),peak_reserved_gpu_bytes=torch.cuda.max_memory_reserved(),
                      claim_limitations='static model setup reported separately; fixed cached detector boxes omit GMT detection/GTA; warm microbenchmark speed cannot be relabeled end-to-end throughput or tracking accuracy')
    except Exception as error:
        report.update(status='INCOMPATIBLE_OR_FAILED',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
    save_json(out/'manifest.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
