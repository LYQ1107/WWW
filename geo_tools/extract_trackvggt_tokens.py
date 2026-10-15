"""Bounded target token pools on real event images; never a deployment cache."""
import argparse,collections,fcntl,time
import numpy as np
import torch
from common import *

NEW=Path('/data1/liuyeqiang/WWW_trackvggt_runtime')
OLD=Path('/data1/liuyeqiang/WWW_vggt_geo_runtime')


def save_bounded_npz(path,arrays,cap=2*1024**3):
    path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix('.part')
    with temporary.open('xb') as stream:np.savez_compressed(stream,**arrays)
    with (NEW/'tokens/.write.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        total=sum(p.stat().st_size for p in (NEW/'tokens').rglob('*.npz'))
        if total+temporary.stat().st_size>cap:
            temporary.unlink();raise ValueError('shared2GiB teacher compact token budget exhausted')
        if path.exists():raise FileExistsError(path)
        temporary.replace(path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--backbone',choices=['vggt','dinov2'],required=True)
    p.add_argument('--cohort',default=str(NEW/'cohorts/hard_events_expanded_v2/cohort.json'))
    p.add_argument('--scene',type=int,required=True);p.add_argument('--tag',required=True);p.add_argument('--limit-events',type=int)
    a=p.parse_args();out=NEW/'tokens'/a.tag
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);cohort=json.loads(Path(a.cohort).read_text())
    if cohort['status']!='LOCKED_SCENE_DISJOINT_REAL_EVENT_COHORT':raise ValueError('unlocked event cohort')
    items=sorted([row for row in cohort['fit']+cohort['validation'] if row['scene']==a.scene],key=lambda row:row['key'])
    if a.limit_events is not None:items=items[:a.limit_events]
    if not items:raise ValueError('no requested physical queries')
    sys.path[:0]=[str(OLD/'sources/vggt'),str(OLD/'sources/dinov2')]
    from vggt.utils.load_fn import load_and_preprocess_images
    from gtr.geometry.trackvggt_features import FrozenTrackVGGT,LAYERS
    from gtr.geometry.track_dinov2_features import FrozenTrackDINO
    from gtr.geometry.roi_feature_extractor import PadMapping,patch_map,pool_roi
    from gtr.geometry.identity_guided_sampler import patch_regions,greedy_2d,normalize_attention
    checkpoint=OLD/'weights'/('vggt_1b.pt' if a.backbone=='vggt' else 'dinov2_vitl14_reg4.pth')
    started=time.perf_counter();encoder=(FrozenTrackVGGT if a.backbone=='vggt' else FrozenTrackDINO)(checkpoint)
    torch.cuda.synchronize();setup_seconds=time.perf_counter()-started
    data=train_data();records=video_records(data,a.scene);by_frame=collections.defaultdict(list)
    for row in records:by_frame[row['frame_id']].append(row)
    payloads={};by_time=collections.defaultdict(list);sources={str(Path(a.cohort)):sha256(a.cohort)}
    for item in items:
        path=Path(item['event_file'])
        if str(path) not in payloads:
            payloads[str(path)]=torch.load(path,map_location='cpu');sources[str(path)]=sha256(path)
        payload=payloads[str(path)];event=payload['events'][item['event_index']]
        if event['key']!=item['key']:raise ValueError('event index drift')
        query=payload['observations'][tuple(event['query_source'])]
        by_time[query['frame']].append((item,event,query))
    rows=[];coverage=collections.Counter();started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    for frame,events in sorted(by_time.items()):
        current=sorted(by_frame[frame],key=lambda row:row['view_id'])
        if len(current)!=2:raise ValueError('expected synchronized physical two-camera group')
        images=load_and_preprocess_images([row['file_name'] for row in current],mode='pad')[None]
        layers,attention,meta=encoder.extract(images)
        for record in current:
            if record['file_name'] not in sources:sources[record['file_name']]=sha256(record['file_name'])
        for item,event,query in events:
            view=next(i for i,row in enumerate(current) if row['view_id']==query['camera'])
            mapping=PadMapping(query['width'],query['height']);device=layers[4].device
            xy,inside,context,valid=patch_regions(mapping,query['box'],device=device)
            saliency=attention[0,view].float()
            normalized=torch.zeros_like(saliency);region=inside|context
            normalized[region]=normalize_attention(saliency[region])
            pool=greedy_2d(normalized+inside.float(),xy,inside,context,64,context_cap=32)
            pool_inside=inside[pool];original_box=torch.tensor([query['box']],device=device)
            roi=[]
            for layer in LAYERS:
                fm=patch_map(layers[layer][0,view],0,(518,518))
                value,_=pool_roi(fm,mapping.boxes(original_box),'roi_align');roi.append(value[0].cpu().numpy().astype(np.float16))
            arrays={f'layer_{layer}':layers[layer][0,view,pool].float().cpu().numpy().astype(np.float16) for layer in LAYERS}
            arrays.update(xy=xy[pool].cpu().numpy(),inside=pool_inside.cpu().numpy(),attention=saliency[pool].cpu().numpy(),
                patch_indices=pool.cpu().numpy(),roi=np.stack(roi))
            name='_'.join(map(str,item['key']))+'.npz';path=out/name;save_bounded_npz(path,arrays)
            count=int(inside.sum());bucket='0' if count==0 else '1-3' if count<4 else '4-15' if count<16 else '16+'
            coverage[bucket]+=1
            rows.append(dict(key=item['key'],scope=item['scope'],file=str(path),sha256=sha256(path),event_file=item['event_file'],event_index=item['event_index'],
                image_id=query['image_id'],frame_id=query['frame'],physical_camera_id=query['camera'],box=query['box'],
                history_snapshot_id=dict(query_key=item['key'],candidate_track_ids=event['candidate_tracks'],source_event_file_sha256=sources[item['event_file']]),
                raw_patch_centers=count,patch_bin=bucket,valid_pool_tokens=len(pool),all_selected_inside_content=bool(valid[pool].all()),
                input_image_ids=[row['image_id'] for row in current],input_physical_cameras=[row['view_id'] for row in current],
                layer_ids=list(LAYERS),channels=int(layers[4].shape[-1]),metadata=meta,small_target_neutral_fallback=count==0))
        del layers,attention
        if len(rows)%25==0:print(json.dumps(dict(scene=a.scene,backbone=a.backbone,events=len(rows),encoder_calls=encoder.calls)),flush=True)
    torch.cuda.synchronize();seconds=time.perf_counter()-started
    report=dict(status='COMPLETE_BOUNDED_REAL_TARGET_TOKEN_POOL',backbone=a.backbone,scene=a.scene,rows=rows,events=len(rows),
        limited_probe=a.limit_events is not None,encoder_calls=encoder.calls,patch_bins=dict(coverage),frozen=True,
        backbone_checkpoint=str(checkpoint),backbone_checkpoint_sha256=sha256(checkpoint),gmt_checkpoint_sha256=sha256(CHECKPOINT),
        teacher_head_checkpoint_sha256=None,teacher_head_scope='pre-training frozen token pool, no teacher head yet; later relation targets must bind trained head SHA',
        ag_qd_version={str(path.relative_to(ROOT)):sha256(path) for path in [ROOT/'gtr/geometry/identity_guided_sampler.py',ROOT/'gtr/geometry/track_query_encoder.py',ROOT/'gtr/geometry/track_qd_teacher.py',ROOT/'gtr/geometry/track_conditioned_see_query.py']},
        preprocessing='official RGB pad518, rounded patch14 shorter side, WHITE padding, normalization once inside corresponding frozen backbone; pad centers excluded',
        prepool='at most64 deterministic attention/spatial box-context patches, context cap32; AG16/24/32 learned selection occurs after TRAIN relevance fitting',
        no_dense_maps_or_full_attention_saved=True,setup_seconds=setup_seconds,extraction_seconds=seconds,
        timing_scope='offline actual frozen encoder/read/preprocessing/selected-token export; not online deployment FPS; VGGT/DINO encoder load separately charged',
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),source_artifacts=sources,
        token_bytes=sum(Path(row['file']).stat().st_size for row in rows),provenance=provenance())
    save_json(out/'manifest.json',report)
    print(json.dumps({key:report[key] for key in ['status','backbone','scene','events','encoder_calls','patch_bins','token_bytes','extraction_seconds']}),flush=True)


if __name__=='__main__':main()
