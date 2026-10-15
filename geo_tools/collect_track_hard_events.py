"""Read-only real causal GMT histories; labels are attached after raw parity.

No foundation encoder, GT-built history, future image or modified association.
Actual solver provenance uses the existing DecisionTrace implementation.
"""
import argparse,collections,gzip,time
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from common import *
from evaluate_live_retrieval import iou_matrix
from audit_decision_trace import label_run

NEW=Path('/data1/liuyeqiang/WWW_trackvggt_runtime')


def pure_reference_identity(values):
    known={value for value in values if value is not None}
    return next(iter(known)) if len(known)==1 and all(value is not None for value in values) else None


def choose_events(events,limit,fit):
    def order(event):
        # TRAIN fit may use error labels; held-out selection never does.
        error=event['top8_positive'] and not event['native_top1_correct']
        return (0 if fit and error else 1,event['relative_margin'],event['key'])
    return sorted(events,key=order)[:limit]


def attach_labels(data,scene,images,detectors):
    name=next(v['file_name'] for v in data['videos'] if v['id']==scene)
    lookup={(row['view_id'],row['frame_id']):row['image_id'] for row in images}
    truth=collections.defaultdict(list);hashes={};labels={}
    for camera in sorted({row['view_id'] for row in images}):
        path=DATA/'train'/f'{name}_View{camera}'/'gt/gt.txt';hashes[str(path)]=sha256(path);seen=set()
        for line in path.read_text().splitlines():
            value=list(map(float,line.split(',')));frame,identity=map(int,value[:2])
            if (frame,identity) in seen:raise ValueError('conflicting original GT; do not repair')
            seen.add((frame,identity))
            if value[6]>0 and value[7]==1 and (camera,frame) in lookup:
                truth[lookup[camera,frame]].append((identity,value[2:6]))
    by_image=collections.defaultdict(list)
    for key,record in detectors.items():by_image[key[0]].append((key,record['detection']))
    for image,rows in by_image.items():
        for key,_ in rows:labels[key]=None
        target=truth[image]
        if not target:continue
        overlap=iou_matrix([row['bbox'] for _,row in rows],[row[1] for row in target])
        ri,ci=linear_sum_assignment(np.where(overlap>=.5,1-overlap,1e6))
        for i,j in zip(ri,ci):
            if overlap[i,j]>=.5:labels[rows[i][0]]=target[j][0]
    return labels,hashes


def main():
    p=argparse.ArgumentParser();p.add_argument('--video-id',type=int,required=True);p.add_argument('--tag',required=True)
    p.add_argument('--max-frames',type=int,default=400);p.add_argument('--limit',type=int,default=500)
    p.add_argument('--scope',choices=['fit','validation','strict_dev'],required=True);a=p.parse_args()
    spec=json.loads((ROOT/'manifests/trackvggt_design_v1.json').read_text())
    allowed={'fit':spec['association_fit']+spec['bounded_expansion_train'],'validation':spec['association_validation'],'strict_dev':spec['strict_dev5']}
    if a.video_id not in allowed[a.scope] or a.max_frames<40 or a.limit<1:raise ValueError('invalid scene/sampling contract')
    out=NEW/'events'/a.tag
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    from gtr.geometry.decision_trace import DecisionTrace
    from gtr.geometry.online_runner import CausalGMT
    from gtr.evaluation.mot_evaluation import custom_instances_to_coco_json
    cfg,model=load_gmt(seed=spec['inference_seed']);data=train_data();images=video_records(data,a.video_id,a.max_frames)
    records_by_image={row['image_id']:row for row in images};groups=collections.defaultdict(list)
    for row in images:groups[row['frame_id']].append(row)
    detectors={};observations={};calls=[];dims=set()
    trace=DecisionTrace(model,out);original_write=trace.write
    def write(stream,record):
        original_write(stream,record)
        if stream is trace.detectors:detectors[tuple(record['source'])]=record
    trace.write=write;trace.install();original_inference=model.inference;original_association=model.get_asso
    def inference(*args,**kwargs):
        result=original_inference(*args,**kwargs)
        for instance in result:
            dims.add(int(instance.reid_features.shape[-1]))
            for index,source in enumerate(instance.decision_trace_source.detach().cpu().tolist()):
                key=tuple(source);record=records_by_image[key[0]];public=detectors[key]['detection'];box=public['bbox']
                observations[key]=dict(source=list(key),image_id=key[0],camera=int(record['view_id']),frame=int(record['frame_id']),
                    path=record['file_name'],width=record['width'],height=record['height'],
                    box=[box[0],box[1],box[0]+box[2],box[1]+box[3]],feature=instance.reid_features[index].detach().cpu().to(torch.float16).clone())
        return result
    def association(instances,k,**kwargs):
        result=original_association(instances,k,**kwargs)
        if kwargs.get('geo_context',True) is False:return result
        query=instances[k];refs=[instance for i,instance in enumerate(instances) if i!=k]
        if not len(query) or not refs or not sum(len(r) for r in refs):return result
        logits=result[0][0]
        sources=[tuple(source) for instance in refs for source in instance.decision_trace_source.detach().cpu().tolist()]
        ids=torch.cat([instance.track_ids for instance in refs]);unique=ids.unique(sorted=True)
        if tuple(logits.shape)!=(len(query),len(ids)):raise ValueError('proposal logit layout mismatch')
        probabilities=torch.cat(model.roi_heads._activate_asso(logits.split([len(r) for r in refs],dim=1)),dim=1)
        scores=probabilities@(ids[:,None]==unique[None,:]).float()
        queries=[tuple(source) for source in query.decision_trace_source.detach().cpu().tolist()]
        id_list=list(map(int,ids.detach().cpu().tolist()));track_list=list(map(int,unique.detach().cpu().tolist()))
        histories={}
        for track in track_list:
            all_sources=[source for source,identity in zip(sources,id_list) if identity==track]
            recent=sorted(all_sources,key=lambda key:(observations[key]['frame'],observations[key]['camera'],key))[-4:]
            histories[track]=dict(sources=[list(key) for key in recent],observed_length=int(tracker.counts.get(track,len(all_sources))))
        calls.append(dict(queries=[list(key) for key in queries],references=[dict(source=list(key),track=int(identity)) for key,identity in zip(sources,id_list)],
            track_ids=track_list,track_scores=scores.detach().float().cpu(),logits=logits.detach().float().cpu(),histories=histories))
        return result
    model.inference=inference;model.get_asso=association
    seed_all(spec['inference_seed']);tracker=CausalGMT(model);predictions=[];started=time.perf_counter()
    try:
        for timestamp,group in sorted(groups.items()):
            outputs=tracker.step(map_records(cfg,group))
            for record,output in zip(group,outputs):predictions.extend(custom_instances_to_coco_json(output['instances'].to('cpu'),record['image_id']))
            if tracker.frame%50==0:print(json.dumps(dict(scene=a.video_id,groups=tracker.frame,observations=len(observations),calls=len(calls))),flush=True)
    finally:
        model.get_asso=original_association;trace.close()
    torch.cuda.synchronize();collection_seconds=time.perf_counter()-started
    save_json(out/'predictions.json',predictions)
    seed_all(spec['inference_seed']);baseline=CausalGMT(model);prefix_predictions=[];prefix_images=set()
    for group in list(dict(sorted(groups.items())).values())[:40]:
        outputs=baseline.step(map_records(cfg,group))
        for record,output in zip(group,outputs):
            prefix_images.add(record['image_id']);prefix_predictions.extend(custom_instances_to_coco_json(output['instances'].to('cpu'),record['image_id']))
    exact=prefix_predictions==[row for row in predictions if row['image_id'] in prefix_images]
    if not exact:raise ValueError('read-only collection changed actual OFF raw prefix')
    # No raw GT has been read before this point. Annotation records were only
    # used for image metadata by the unchanged GMT data mapper.
    labels,gt_hashes=attach_labels(data,a.video_id,images,detectors)
    outcomes,solver_stats=label_run(out,dict(video_id=a.video_id),images)
    candidates=[];coverage=collections.Counter()
    for call_index,call in enumerate(calls):
        ref_truth=collections.defaultdict(list)
        for ref in call['references']:ref_truth[ref['track']].append(labels[tuple(ref['source'])])
        pure={track:pure_reference_identity(values) for track,values in ref_truth.items()}
        for row,qsource in enumerate(call['queries']):
            key=tuple(qsource);query=observations[key];truth=labels[key]
            score=call['track_scores'][row];order=torch.argsort(score,descending=True,stable=True);top=order[:8].tolist()
            selected=[call['track_ids'][index] for index in top]
            cross=any(observations[tuple(source)]['camera']!=query['camera'] for track in selected for source in call['histories'][track]['sources'])
            coverage['queries_total']+=1
            if not cross:coverage['no_cross_camera_top8_history']+=1;continue
            all_positive=[track for track,value in pure.items() if truth is not None and value==truth]
            positives=[track for track in selected if track in all_positive]
            unknown=[track for track in selected if pure[track] is None]
            known_query=truth is not None;has_all=bool(all_positive);has_top=bool(positives)
            first=selected[0];correct=known_query and pure[first]==truth
            margin=float((score[order[0]]-score[order[1]])/(score[order[0]].abs()+1e-8)) if len(order)>1 else 1.
            coverage['cross_camera_queries']+=1;coverage['known_query']+=known_query;coverage['correct_candidate_anywhere']+=has_all
            coverage['correct_candidate_in_top8']+=has_top;coverage['positive_excluded_by_top8']+=has_all and not has_top
            coverage['top8_correctable_native_top1_errors']+=has_top and not correct
            coverage['unknown_query']+=not known_query;coverage['query_with_ambiguous_top8_track']+=bool(unknown)
            coverage['known_query_no_known_positive_in_history']+=known_query and not has_all
            candidates.append(dict(key=[a.video_id,*key],call_index=call_index,query_row=row,query_source=qsource,
                candidate_tracks=selected,candidate_track_indices=top,native_track_scores=score[top].tolist(),relative_margin=margin,
                query_gt_identity=truth,positive_tracks=positives,all_history_positive_tracks=all_positive,ambiguous_tracks=unknown,
                top8_positive=has_top,all_history_positive=has_all,native_top1_correct=bool(correct),
                loss_eligible=known_query and (has_top or (not has_all and not any(value is None for value in pure.values()))),
                known_unmatched_within_all_recorded_history=known_query and not has_all and not any(value is None for value in pure.values()),
                actual_primary_state=outcomes[key]['state'],actual_solver_scope='primary before memory; own frozen OFF history',
                selection_scope='GT-labeled hard errors first then native margin' if a.scope=='fit' else 'native margin only; no GT selection'))
    selected=choose_events(candidates,a.limit,a.scope=='fit')
    selected_calls={row['call_index'] for row in selected};needed=set()
    for event in selected:
        call=calls[event['call_index']];needed.add(tuple(event['query_source']))
        for track in event['candidate_tracks']:needed.update(tuple(source) for source in call['histories'][track]['sources'])
    payload=dict(events=selected,observations={key:observations[key] for key in needed},calls={i:calls[i] for i in selected_calls},feature_dim=next(iter(dims)))
    if len(dims)!=1:raise ValueError('inconsistent observed native feature dimensions')
    torch.save(payload,out/'events.pt')
    save_json(out/'event_index.json',[{key:value for key,value in event.items()} for event in selected])
    save_json(out/'manifest.json',dict(status='PASS_READONLY_EVENTS_POST_PARITY_LABELS',video_id=a.video_id,scope=a.scope,max_frames=a.max_frames,
        full_video=len(images)==len(video_records(data,a.video_id)),inference_seed=spec['inference_seed'],images=len(images),groups=len(groups),
        native_reid_dim=next(iter(dims)),all_candidate_coverage=dict(coverage),selected_events=len(selected),
        selected_known_query=sum(row['query_gt_identity'] is not None for row in selected),selected_top8_positive=sum(row['top8_positive'] for row in selected),
        selected_native_top1_errors_with_top8_positive=sum(row['top8_positive'] and not row['native_top1_correct'] for row in selected),
        selected_loss_eligible=sum(row['loss_eligible'] for row in selected),selected_actual_primary_states=dict(collections.Counter(row['actual_primary_state'] for row in selected)),
        solver_decision_counts=solver_stats,prefix40_raw_prediction_exact=exact,raw_gt_sha256=gt_hashes,collection_seconds=collection_seconds,
        vggt_calls=0,dinov2_calls=0,speed_eligible=False,gt_query_features=False,future_query_features=False,train_validation_scene_separation=True,
        selection_uses_gt=a.scope=='fit',selection_policy=selected[0]['selection_scope'] if selected else 'empty',
        artifacts={path.name:sha256(path) for path in out.iterdir() if path.is_file()},provenance=provenance(spec['inference_seed'])))
    print(json.dumps(dict(scene=a.video_id,selected=len(selected),coverage=dict(coverage),solver=solver_stats,prefix40_exact=exact)),flush=True)


if __name__=='__main__':main()
