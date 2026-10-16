"""Complete actual-image GMT forward, isolated CIR namespace, raw + stock outputs.

The original official forward/sliding core executes. Images are mapped one
synchronized group at a time, avoiding a duplicate complete-video RAM cache.
"""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import resource
import subprocess
import time
from types import MethodType
import torch
from common import ROOT,DATA,CHECKPOINT,load_gmt,map_records,seed_all,sha256,save_json,train_data,video_records
from gtr.evaluation.mot_evaluation import custom_instances_to_coco_json
from test_cir_native_commit import original_method


class CurrentImages(list):
    """Metadata on integer access/iteration; true image tensors on inference slice.

    The stock forward only requests image slices for its current camera; final
    postprocessing requires original dimensions, never old image tensors.
    """
    def __init__(self,cfg,records):
        super().__init__([dict(r) for r in records])
        self.cfg=cfg;self.groups=defaultdict(list);self.current=None;self.mapped={}
        for r in records:self.groups[r['frame_id']].append(r)
        self.processed=[];self.inference_images=set()

    def __getitem__(self,key):
        if not isinstance(key,slice):return super().__getitem__(key)
        records=super().__getitem__(key)
        if not records:return []
        frame=records[0]['frame_id']
        if any(r['frame_id']!=frame for r in records):raise ValueError('Official core requested future/mixed-time image slice')
        if self.current!=frame:
            if self.current is not None and frame<=self.current:raise ValueError('Noncausal image read')
            rows=map_records(self.cfg,self.groups[frame]);self.mapped={r['image_id']:r for r in rows};self.current=frame
        for r in records:
            image=r['image_id']
            if image in self.inference_images:raise ValueError('Repeated actual image inference')
            self.inference_images.add(image);self.processed.append(image)
        return [self.mapped[r['image_id']] for r in records]


def validate_request(a):
    if a.video not in range(1,25):raise ValueError('TRAIN only; TEST requires separate final frozen lock')
    if a.out.exists():raise FileExistsError(a.out)
    gate=json.loads(a.qualification.read_text())
    if gate['status']!='PASS_FULL_VIDEO_NATIVE_OFF_QUALIFICATION':raise ValueError('Complete original/OFF qualification required')
    if a.method not in gate.get('allowed_methods',('original','off','keep')) or a.video not in gate.get('allowed_video_ids',range(1,25)):
        raise ValueError('This policy/video has not passed the current qualification gate')
    for name,digest in gate['current_source_sha256'].items():
        if sha256(ROOT/name)!=digest:raise ValueError('Native source changed after OFF qualification: '+name)
    return gate


def run_loaded(a,cfg,model):
    validate_request(a)
    cfg=cfg.clone();cfg.defrost()
    cfg.MODEL.CIR.ENABLED=a.method in ('keep','c1');cfg.MODEL.CIR.MODE=a.method if cfg.MODEL.CIR.ENABLED else 'off'
    from gtr.modeling.identity_revision.controller import IdentityRevisionController
    model.cir_controller=IdentityRevisionController(cfg.MODEL.CIR)
    from gtr.modeling.meta_arch.gtr_rcnn import GTRRCNN
    model.run_global_tracker_plus=MethodType(original_method() if a.method=='original' else GTRRCNN.run_global_tracker_plus,model)
    if cfg.MODEL.GEO.ENABLED or model.geo_runtime.active:raise ValueError('CIR must run pure frozen GMT')
    model.eval().requires_grad_(False)
    # raw0 and stock50 are both retained; filter is only after full tracking.
    stock_min=model.min_track_len;model.min_track_len=0
    records=video_records(train_data(),a.video);batch=CurrentImages(cfg,records)
    a.out.mkdir(parents=True);(a.out/'config.yaml').write_text(cfg.dump())
    source={str(p.relative_to(ROOT)):sha256(p) for p in (ROOT/'gtr/modeling/meta_arch/gtr_rcnn.py',
        ROOT/'gtr/config.py',ROOT/'gtr/geometry/online_runner.py',ROOT/'gtr/data/datasets/mot.py',
        ROOT/'test_net.py',ROOT/'geo_tools/run_cir_online.py',ROOT/'geo_tools/cir_official_driver.py',
        ROOT/'geo_tools/cir_configuration.py',ROOT/'geo_tools/common.py',ROOT/'geo_tools/test_cir_native_commit.py',
        *sorted((ROOT/'gtr/modeling/identity_revision').glob('*.py')))}
    from cir_configuration import configuration_hashes,reference_configuration_hash
    checkpoint_sha256=sha256(CHECKPOINT)
    config_hashes=configuration_hashes(cfg,checkpoint_sha256)
    if config_hashes['frozen_GMT_config_sha256']!=reference_configuration_hash(checkpoint_sha256):
        raise ValueError('CIR comparison may not change GMT inference settings outside the CIR namespace')
    seed_all(20261014);torch.cuda.reset_peak_memory_stats()
    manifest=dict(status='RUNNING',split='train',video_id=a.video,method=a.method,
        seed=20261014,full_video=False,GT_in_inference=False,checkpoint_sha256=sha256(CHECKPOINT),
        config_sha256=sha256(a.out/'config.yaml'),source_sha256=source,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_tree_dirty_at_video_start=bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT)),
        entrypoint=getattr(a,'entrypoint','geo_tools/run_cir_online.py'),
        qualification=dict(path=str(a.qualification),sha256=sha256(a.qualification)),
        inference_protocol='actual model.forward/sliding_inference_GMT with synchronized current image mapper; raw0 + stock50',
        parameter_count=sum(p.numel() for p in model.parameters()),new_learnable_parameters=0)
    manifest.update(config_hashes)
    manifest['hardware']=dict(CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),
        GPU_name=torch.cuda.get_device_name(),GPU_total_bytes=torch.cuda.get_device_properties(0).total_memory,
        torch=torch.__version__,torch_CUDA=torch.version.cuda)
    save_json(a.out/'manifest.json',manifest)
    begin=time.perf_counter()
    try:
        with torch.inference_mode():
            output,views=model(batch)
            # Filtering depends only on whole-video ID occurrence counts and
            # commutes with box scaling; use the actual native filter once.
            model.min_track_len=stock_min
            stock=model._remove_short_track([item['instances'] for item in output]) if stock_min>0 else [item['instances'] for item in output]
        torch.cuda.synchronize();seconds=time.perf_counter()-begin
        frame_rows=sorted(records,key=lambda r:(r['frame_id'],r['view_id']))
        if len(output)!=len(records) or set(batch.processed)!={r['image_id'] for r in records}:raise ValueError('Incomplete real image/track output coverage')
        files={}
        for kind,instances in (('raw',[item['instances'] for item in output]),('stock',stock)):
            predictions=[]
            for row,inst in zip(frame_rows,instances):
                predictions.extend(custom_instances_to_coco_json(inst.to('cpu'),row['image_id']))
            path=a.out/f'{kind}_predictions.json';save_json(path,predictions)
            files[kind]=dict(path=str(path.resolve()),sha256=sha256(path),detections=len(predictions))
        manifest.update(status='COMPLETE',full_video=True,processed_image_ids=batch.processed,predictions=files,
            full_image_seconds=seconds,full_image_FPS=len(records)/seconds,
            peak_GPU_bytes=torch.cuda.max_memory_allocated(),peak_process_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            timing_scope='image read/map + frozen detector/descriptor/GTA + all native state/Bank + CIR + postprocess and stock filtering; excludes model init, GT evaluation and output serialization; no feature cache',
            stock_min_track_len=stock_min,stock_filtering='unchanged model._remove_short_track on final Instances; count across original full multiview video',
            perception_cache_used=False,VGGT_calls=model.geo_runtime.vggt_calls,DINO_calls=model.geo_runtime.dinov2_calls,
            image_sha256={str(r['image_id']):sha256(r['file_name']) for r in records})
        import numpy as np
        controller=model.cir_controller
        manifest['CIR_stats']=dict(calls=controller.calls,score_changed_calls=controller.score_changed_calls,
            extra_joint_solves=controller.extra_joint_solves,changed_primary_rows=controller.changed_primary_rows,
            changed_original_accept_rows=controller.changed_original_accept_rows,fallback_reasons=dict(controller.fallbacks),
            controller_seconds=sum(controller.intervention_latencies),
            p50_ms=float(np.quantile(controller.intervention_latencies,.5)*1000) if controller.intervention_latencies else None,
            p95_ms=float(np.quantile(controller.intervention_latencies,.95)*1000) if controller.intervention_latencies else None,
            correctness='not labelled here; changed native primary rows are not certified GT repairs',
            Bank_and_initialization='original, no C1 score modification in these stages')
    except BaseException as error:
        manifest.update(status='FAILED_NO_ADMISSION',failure=repr(error),processed_image_ids=batch.processed)
        raise
    finally:save_json(a.out/'manifest.json',manifest)
    print(json.dumps(dict(status=manifest['status'],method=a.method,video=a.video,images=len(records),FPS=manifest['full_image_FPS'])),flush=True)
    return manifest




def main():
    p=argparse.ArgumentParser();p.add_argument('--video',type=int,required=True)
    p.add_argument('--method',choices=('original','off','keep','c1'),required=True)
    p.add_argument('--qualification',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    validate_request(a)
    cfg,model=load_gmt()
    run_loaded(a,cfg,model)


if __name__=='__main__':main()
