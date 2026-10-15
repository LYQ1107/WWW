"""Only missing V3/new-cycle target pools; all completed V2 pools are reused."""
import argparse
import collections
import time
import numpy as np
import torch
from common import *
from extract_trackvggt_tokens import save_bounded_npz,NEW,OLD
from trackvggt_token_pool import full_image_target_pool


def target_requests(cohort,scene):
    new_keys={tuple(k) for k in cohort['primary_new_target_keys']};requests={}
    for item in cohort['fit']:
        key=tuple(item['key'])
        if item['scene']==scene and key in new_keys:requests[key]=dict(key=item['key'],source=item['key'][1:],event_file=item['event_file'],event_index=item['event_index'],roles=['new_V3_fit'],parents=[item['key']])
    for row in cohort['auxiliary_target_index']:
        if row['scene']!=scene:continue
        key=tuple(row['key'])
        if key in requests:raise ValueError('auxiliary should alias an existing primary physical key')
        requests[key]=dict(key=row['key'],source=row['source'],event_file=row['event_file'],event_index=None,roles=['cycle_B_auxiliary'],parents=row['parents'])
    if {tuple(k) for k in cohort['reused_V2_target_keys']}&set(requests):raise ValueError('completed V2 target requested again')
    return sorted(requests.values(),key=lambda r:r['key'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--backbone',choices=['vggt','dinov2'],required=True);p.add_argument('--scene',type=int,required=True)
    p.add_argument('--cohort',default=str(NEW/'cohorts/hard_events_expanded_v3/cohort.json'));p.add_argument('--tag',required=True);p.add_argument('--limit-targets',type=int)
    a=p.parse_args();cohort_path=Path(a.cohort);cohort=json.loads(cohort_path.read_text())
    if cohort['status']!='LOCKED_SCENE_DISJOINT_REAL_EVENT_COHORT' or cohort.get('cohort_version')!=3:raise ValueError('locked V3 native cohort required')
    if cohort.get('interface_probe') and a.limit_targets is None:raise ValueError('interface probe requires an explicit finite target limit')
    requests=target_requests(cohort,a.scene)
    if a.limit_targets is not None:requests=requests[:a.limit_targets]
    if not requests:raise ValueError('no missing physical target for this scene')
    out=NEW/'tokens'/a.tag
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);sources={str(cohort_path):sha256(cohort_path)};payloads={};by_time=collections.defaultdict(list)
    for request in requests:
        path=request['event_file']
        if path not in payloads:
            if sha256(path)!=cohort['source_artifacts'][path]:raise ValueError('native source changed')
            payloads[path]=torch.load(path,map_location='cpu');sources[path]=sha256(path)
        payload=payloads[path];source=tuple(request['source']);query=payload['observations'][source]
        if request['event_index'] is not None and payload['events'][request['event_index']]['query_source']!=list(source):raise ValueError('primary event index drift')
        by_time[query['frame']].append((request,query))
    sys.path[:0]=[str(OLD/'sources/vggt'),str(OLD/'sources/dinov2')]
    from vggt.utils.load_fn import load_and_preprocess_images
    from gtr.geometry.trackvggt_features import FrozenTrackVGGT,LAYERS
    from gtr.geometry.track_dinov2_features import FrozenTrackDINO
    checkpoint=OLD/'weights'/('vggt_1b.pt' if a.backbone=='vggt' else 'dinov2_vitl14_reg4.pth')
    seed_all(20261014);started=time.perf_counter();encoder=(FrozenTrackVGGT if a.backbone=='vggt' else FrozenTrackDINO)(checkpoint)
    torch.cuda.synchronize();setup=time.perf_counter()-started
    data=train_data();by_frame=collections.defaultdict(list)
    for image in video_records(data,a.scene):by_frame[image['frame_id']].append(image)
    rows=[];bins=collections.Counter();started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    for frame,items in sorted(by_time.items()):
        current=sorted(by_frame[frame],key=lambda r:r['view_id'])
        images=load_and_preprocess_images([r['file_name'] for r in current],mode='pad')[None]
        layers,attention,metadata=encoder.extract(images)
        for image in current:
            if image['file_name'] not in sources:sources[image['file_name']]=sha256(image['file_name'])
        for request,query in items:
            view=next(i for i,image in enumerate(current) if image['view_id']==query['camera'])
            arrays,counts=full_image_target_pool(layers,attention,view,query);name='_'.join(map(str,request['key']))+'.npz';path=out/name
            save_bounded_npz(path,arrays);bins[counts['patch_bin']]+=1
            binding=dict(query_key=request['key'],roles=request['roles'],parents=request['parents'],source_event_file_sha256=sources[request['event_file']])
            if request['event_index'] is not None:binding['candidate_track_ids']=payloads[request['event_file']]['events'][request['event_index']]['candidate_tracks']
            rows.append(dict(key=request['key'],scope='fit',file=str(path),sha256=sha256(path),event_file=request['event_file'],event_index=request['event_index'],
                image_id=query['image_id'],frame_id=query['frame'],physical_camera_id=query['camera'],box=query['box'],history_snapshot_id=binding,
                **counts,input_image_ids=[r['image_id'] for r in current],input_physical_cameras=[r['view_id'] for r in current],layer_ids=list(LAYERS),channels=int(layers[4].shape[-1]),
                metadata=metadata,small_target_neutral_fallback=counts['raw_patch_centers']==0,roles=request['roles']))
        del layers,attention
        if len(rows)%25==0:print(json.dumps(dict(scene=a.scene,backbone=a.backbone,targets=len(rows),encoder_calls=encoder.calls)),flush=True)
    torch.cuda.synchronize();seconds=time.perf_counter()-started
    report=dict(status='COMPLETE_BOUNDED_REAL_TARGET_TOKEN_POOL',backbone=a.backbone,scene=a.scene,rows=rows,events=len(rows),limited_probe=a.limit_targets is not None,
        encoder_calls=encoder.calls,patch_bins=dict(bins),frozen=True,backbone_checkpoint=str(checkpoint),backbone_checkpoint_sha256=sha256(checkpoint),gmt_checkpoint_sha256=sha256(CHECKPOINT),
        teacher_head_checkpoint_sha256=None,teacher_head_scope='Pretraining frozen target pool; no fitted teacher head yet',
        no_dense_maps_or_full_attention_saved=True,cohort_sha256=sha256(cohort_path),source_artifacts=sources,token_bytes=sum(Path(r['file']).stat().st_size for r in rows),
        setup_seconds=setup,extraction_seconds=seconds,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        preprocessing='Identical V2 physical two-view RGB WHITE-pad518/patch14, normalization once, verified math backend. Cycle B targets use actual detector boxes in the physical full views.',
        prepool='Identical V2 union of pure-box spatial64 and attention/spatial64(context cap32), at most128 sorted original indices. No history/GT enters prepool.',
        reused_V2_targets_not_reencoded=True,timing_scope='Offline frozen extraction/read/pool/export; setup separately measured, not deployment FPS',
        ag_qd_version={str(path.relative_to(ROOT)):sha256(path) for path in [ROOT/'gtr/geometry/identity_guided_sampler.py',ROOT/'gtr/geometry/track_query_encoder.py',ROOT/'gtr/geometry/track_qd_teacher.py',ROOT/'gtr/geometry/track_conditioned_see_query.py']},
        provenance=provenance())
    save_json(out/'manifest.json',report)
    print(json.dumps({k:report[k] for k in ['status','backbone','scene','events','encoder_calls','patch_bins','token_bytes','extraction_seconds']}),flush=True)


if __name__=='__main__':main()
