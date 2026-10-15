"""Bounded TRAIN-only target vectors and actual causal proposal logits.

The live adapter returns its input objects unchanged. Identity supervision is
attached only after inference and a separate OFF prefix-equivalence check.
"""
import argparse,collections,time
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from common import *
from evaluate_live_retrieval import iou_matrix


def observation_key(instance,index):
    return (int(instance.geo_frame[index]),int(instance.geo_camera[index]),
            *instance.geo_xyxy[index].detach().cpu().tolist())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--video-id',type=int,choices=[5,8,12],required=True)
    parser.add_argument('--tag',required=True);args=parser.parse_args()
    split=json.loads((ROOT/'manifests/geo_split_v1.json').read_text())
    if args.video_id not in split['association_train_video_ids']:raise ValueError('association TRAIN only')
    out=RUNTIME/'history_train'/args.tag
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);sys.path[:0]=[str(RUNTIME/'sources/vggt'),str(RUNTIME/'sources/dinov2')]
    from gtr.geometry.runtime import GeometryRuntime
    from gtr.geometry.online_runner import CausalGMT,clone_instances
    from gtr.evaluation.mot_evaluation import custom_instances_to_coco_json
    cfg,model=load_gmt();base_geo=cfg.MODEL.GEO.clone()
    def extractor(kind):
        geo=base_geo.clone()
        options=dict(ENABLED=True,FEATURE_ENABLED=True,GEOMETRY_ENABLED=False,FEATURE_WEIGHT=2.,GEOMETRY_WEIGHT=0.,
            FEATURE_BACKBONE=kind,FEATURE_LAYERS=[17] if kind=='vggt' else [-1],
            PROJECTION_DIM=0 if kind=='vggt' else 256,POOLING='roi_align',SYMMETRIZE_VIEW_ORDER=kind=='vggt',
            HEAD_PATH='',SPARSE_UPDATE_ENABLED=True,UPDATE_POLICY='fixed',UPDATE_INTERVAL=5,MIN_UPDATE_INTERVAL=1,
            COMPUTE_BUDGET=200,BUDGET_WINDOW=100,TOTAL_CALL_BUDGET=-1,
            CALIBRATION_PATH=str(ROOT/'manifests'/('calibration_native_l17_absolute_train_v1.json' if kind=='vggt' else 'calibration_dinov2_roi_train_v1.json')),
            VGGT_CHECKPOINT=str(RUNTIME/'weights/vggt_1b.pt'),DINOV2_CHECKPOINT=str(RUNTIME/'weights/dinov2_vitl14_reg4.pth'))
        for key,value in options.items():setattr(geo,key,value)
        return GeometryRuntime(geo)
    vg,dino=extractor('vggt'),extractor('dinov2');model.geo_runtime=vg
    original_observe=vg.observe;metadata=[];vectors={name:[] for name in ('gmt','vggt','dinov2')};lookup={};events=[]
    def observe(instances,records,frame):
        original_observe(instances,records,frame)
        # Both encoders preserve GMT RNG via GeometryRuntime.observe. The
        # independent DINO operates on clones, never overwrites tracking data.
        clones=[clone_instances(inst) for inst in instances];dino.observe(clones,records,frame)
        for instance,independent,record in zip(instances,clones,records):
            for index in range(len(instance)):
                if not bool(instance.geo_valid[index]) or not bool(independent.geo_valid[index]):continue
                native=instance.reid_features[index].detach().float()
                if not torch.isfinite(native).all() or native.norm()==0:continue
                key=observation_key(instance,index)
                if key in lookup:raise ValueError('ambiguous duplicate observation key')
                lookup[key]=len(metadata)
                metadata.append(dict(frame=frame,raw_frame=int(record['frame_id']),camera=int(record['view_id']),
                    image_id=int(record['image_id']),bbox_xyxy=instance.geo_xyxy[index].detach().cpu().tolist()))
                for name,value in [('gmt',native),('vggt',instance.geo_vector[index]),('dinov2',independent.geo_vector[index])]:
                    vectors[name].append(value.detach().cpu().to(torch.float16).numpy())
    def neutral(outputs,instances,k):
        query=instances[k];refs=[inst for index,inst in enumerate(instances) if index!=k]
        q=[lookup.get(observation_key(query,i),-1) for i in range(len(query))]
        r=[lookup.get(observation_key(inst,i),-1) for inst in refs for i in range(len(inst))]
        if any(i>=0 for i in q) and any(i>=0 for i in r):
            logits=outputs[0].detach().float().cpu()
            if tuple(logits.shape)!=(len(q),len(r)):raise ValueError('proposal-logit layout mismatch')
            events.append(dict(query=q,reference=r,logits=logits))
        return outputs
    vg.observe=observe;vg.adapt=neutral
    data=train_data();records=video_records(data,args.video_id,200);groups=collections.defaultdict(list)
    for record in records:groups[record['frame_id']].append(record)
    seed_all(20261014);tracker=CausalGMT(model);predictions=[];started=time.perf_counter()
    for _,group in sorted(groups.items()):
        outputs=tracker.step(map_records(cfg,group))
        for record,output in zip(group,outputs):predictions.extend(custom_instances_to_coco_json(output['instances'].to('cpu'),record['image_id']))
        if tracker.frame%50==0:print(json.dumps(dict(video=args.video_id,groups=tracker.frame,observations=len(metadata),events=len(events))),flush=True)
    torch.cuda.synchronize();collection_seconds=time.perf_counter()-started
    # This second real pass consumes no foundation features or GT labels.
    model.geo_runtime=GeometryRuntime(base_geo);seed_all(20261014);baseline=CausalGMT(model);baseline_predictions=[]
    prefix_ids=set()
    for frame,group in sorted(groups.items()):
        if frame>40:break
        prefix_ids.update(record['image_id'] for record in group)
        outputs=baseline.step(map_records(cfg,group))
        for record,output in zip(group,outputs):baseline_predictions.extend(custom_instances_to_coco_json(output['instances'].to('cpu'),record['image_id']))
    exact=baseline_predictions==[row for row in predictions if row['image_id'] in prefix_ids]
    save_json(out/'predictions.json',predictions)
    if not exact:raise ValueError('neutral TRAIN collector changed causal GMT prefix')
    # Raw GT enters only here, after both inference passes have finished.
    scene=next(video['file_name'] for video in data['videos'] if video['id']==args.video_id)
    truth=collections.defaultdict(list);gt_hashes={}
    for camera in sorted({record['view_id'] for record in records}):
        path=DATA/'train'/f'{scene}_View{camera}'/'gt/gt.txt';gt_hashes[str(path)]=sha256(path);seen=set()
        for line in path.read_text().splitlines():
            values=list(map(float,line.split(',')));frame,identity=map(int,values[:2])
            if (frame,identity) in seen:raise ValueError('ambiguous raw TRAIN identity')
            seen.add((frame,identity))
            if values[6]>0 and values[7]==1:truth[camera,frame].append((identity,values[2:6]))
    labels=np.full(len(metadata),-1,dtype=np.int64);by_image=collections.defaultdict(list)
    for index,row in enumerate(metadata):by_image[row['image_id']].append(index)
    for indices in by_image.values():
        first=metadata[indices[0]];gt=truth[first['camera'],first['raw_frame']]
        if not gt:continue
        boxes=np.asarray([metadata[i]['bbox_xyxy'] for i in indices]);boxes[:,2:]-=boxes[:,:2]
        overlap=iou_matrix(boxes,[value[1] for value in gt]);ri,ci=linear_sum_assignment(np.where(overlap>=.5,1-overlap,1e6))
        for i,j in zip(ri,ci):
            if overlap[i,j]>=.5:labels[indices[i]]=gt[j][0]
    if not metadata or not events:raise ValueError('no finite history observations')
    np.savez_compressed(out/'targets.npz',**{name:np.stack(values) for name,values in vectors.items()},labels=labels)
    torch.save(events,out/'events.pt');save_json(out/'observations.json',metadata)
    save_json(out/'manifest.json',dict(provenance=provenance(),video_id=args.video_id,source_split='association_train',max_frames=200,
        interval=5,max_history_age=20,causal_online=True,full_video=False,used_cache=False,neutral_prefix40_exact=exact,
        observations=len(metadata),events=len(events),images=len(records),collection_seconds=collection_seconds,
        vggt_calls=vg.vggt_calls,dinov2_calls=dino.dinov2_calls,encoder_failures=vg.failed+dino.failed,
        raw_gt_sha256=gt_hashes,fit_video_ids=[5,8],validation_video_ids=[12],
        scope='fresh frozen target-only vectors at fixed keyframes; unchanged causal GMT actual proposal logits; raw GT one-to-one IoU>=0.5 labels attached after inference; no dense token maps, DEV or TEST',
        speed_eligible=False,timing_scope='offline TRAIN collection and neutral-prefix diagnostic, not inference throughput',
        artifacts={p.name:sha256(p) for p in out.iterdir() if p.is_file()}))
    print(json.dumps(dict(video=args.video_id,observations=len(metadata),events=len(events),neutral_prefix40_exact=exact)))


if __name__=='__main__':main()
