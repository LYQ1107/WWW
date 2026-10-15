"""Single-current-observation local magnification, bounded immutable tokens."""
import argparse
import time
import numpy as np
import torch
from PIL import Image
from common import *
from extract_trackvggt_tokens import NEW,OLD,save_bounded_npz
from trackvggt_crop_data import crop_rectangle,crop_image,crop_target_pool


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--backbone',choices=['vggt','dinov2'],required=True)
    p.add_argument('--probe',action='store_true');a=p.parse_args();plan=json.loads(a.plan.read_text())
    if plan['status']!='LOCKED_OBSERVATION_ONLY_CROP_COMPARISON':raise ValueError('unlocked crop scope')
    cohort=json.loads(Path(plan['base_cohort']).read_text())
    if sha256(plan['base_cohort'])!=plan['base_cohort_sha256']:raise ValueError('changed base cohort')
    keys={tuple(k) for k in plan['fit_keys']+plan['validation_keys']};items=[r for r in cohort['fit']+cohort['validation'] if tuple(r['key']) in keys]
    if len(items)!=len(keys):raise ValueError('crop input selection mismatch')
    payloads={};source_artifacts={str(a.plan):sha256(a.plan),plan['base_cohort']:sha256(plan['base_cohort'])};queries=[]
    for item in sorted(items,key=lambda r:r['key']):
        path=item['event_file']
        if path not in payloads:
            if sha256(path)!=cohort['source_artifacts'][path]:raise ValueError('native crop source changed')
            payloads[path]=torch.load(path,map_location='cpu');source_artifacts[path]=sha256(path)
        payload=payloads[path];event=payload['events'][item['event_index']]
        if event['key']!=item['key']:raise ValueError('crop native index drift')
        query=payload['observations'][tuple(event['query_source'])];queries.append((item,query))
    if a.probe:
        # Fixed physical-key-first held-out source, without GT/teacher filtering.
        queries=[next(pair for pair in queries if pair[0]['scope']=='validation')]
    else:
        gate=NEW/'interfaces'/f'{a.backbone}_crop224_precheck_v1.json';proof=json.loads(gate.read_text())
        if proof['status']!='PASS_REAL_LOCAL_CROP_REPEAT' or proof['plan_sha256']!=sha256(a.plan):raise ValueError('real crop interface gate required')
    out=NEW/'tokens'/f'{a.backbone}_crop224_v1'
    if not a.probe:
        if out.exists():raise FileExistsError(out)
        out.mkdir(parents=True)
    sys.path[:0]=[str(OLD/'sources/vggt'),str(OLD/'sources/dinov2')]
    from gtr.geometry.trackvggt_features import FrozenTrackVGGT
    from gtr.geometry.track_dinov2_features import FrozenTrackDINO
    checkpoint=OLD/'weights'/('vggt_1b.pt' if a.backbone=='vggt' else 'dinov2_vitl14_reg4.pth');seed_all(20261014)
    started=time.perf_counter();encoder=(FrozenTrackVGGT if a.backbone=='vggt' else FrozenTrackDINO)(checkpoint);torch.cuda.synchronize();setup=time.perf_counter()-started
    started=time.perf_counter();rows=[];torch.cuda.reset_peak_memory_stats()
    for item,query in queries:
        if time.perf_counter()-started+setup>plan['maximum_new_frozen_encoder_GPU_wall_hours']*3600/2:raise ValueError('bounded crop encoder budget exceeded')
        rect=crop_rectangle(query['box'],query['width'],query['height'],plan['context_expansion'])
        if rect is None:raise ValueError('native-box crop misses image content; preserve failure and define neutral handling before full extraction')
        path=Path(query['path']);source_artifacts[str(path)]=sha256(path)
        with Image.open(path) as image:images,mapping=crop_image(image,rect,plan['resolution'])
        layers,attention,metadata=encoder.extract(images);arrays,counts=crop_target_pool(layers,attention,query,rect,plan['resolution'])
        if not counts['raw_patch_centers']:raise ValueError('actual crop still has no native-box center')
        if a.probe:
            repeat_layers,repeat_attention,_=encoder.extract(images);repeat,_=crop_target_pool(repeat_layers,repeat_attention,query,rect,plan['resolution'])
            if any(not np.array_equal(value,repeat[name]) for name,value in arrays.items()):raise ValueError('real crop encoder repeat differs')
            proof=dict(status='PASS_REAL_LOCAL_CROP_REPEAT',backbone=a.backbone,plan_sha256=sha256(a.plan),key=item['key'],crop_rect=rect,
                input_hw=list(images.shape[-2:]),input_observations=1,physical_camera_id=query['camera'],not_an_additional_physical_camera=True,
                arrays_bitwise_repeat=True,counts=counts,metadata=metadata,frozen=True,checkpoint_sha256=sha256(checkpoint),
                setup_seconds=setup,probe_seconds=time.perf_counter()-started,provenance=provenance())
            save_json(NEW/'interfaces'/f'{a.backbone}_crop224_precheck_v1.json',proof);print(json.dumps({k:v for k,v in proof.items() if k!='provenance'}));return
        target=out/('_'.join(map(str,item['key']))+'.npz');save_bounded_npz(target,arrays)
        rows.append(dict(key=item['key'],scope=item['scope'],file=str(target),sha256=sha256(target),crop_rect=rect,original_box=query['box'],
            original_image_id=query['image_id'],physical_camera_id=query['camera'],frame=query['frame'],resolution=plan['resolution'],
            local_magnification_only=True,input_view_count=1,metadata=metadata,**counts))
        del layers,attention
        if len(rows)%50==0:print(json.dumps(dict(backbone=a.backbone,targets=len(rows),encoder_calls=encoder.calls)),flush=True)
    torch.cuda.synchronize();report=dict(status='COMPLETE_BOUNDED_REAL_CROP_POOL',backbone=a.backbone,rows=rows,events=len(rows),plan_sha256=sha256(a.plan),
        setup_seconds=setup,extraction_seconds=time.perf_counter()-started,encoder_calls=encoder.calls,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        backbone_checkpoint_sha256=sha256(checkpoint),source_artifacts=source_artifacts,token_bytes=sum(Path(row['file']).stat().st_size for row in rows),
        no_dense_maps_or_images_saved=True,not_physical_camera_or_geometry=True,frozen=True,provenance=provenance())
    save_json(out/'manifest.json',report);print(json.dumps({k:v for k,v in report.items() if k not in ('rows','source_artifacts','provenance')}))


if __name__=='__main__':main()
