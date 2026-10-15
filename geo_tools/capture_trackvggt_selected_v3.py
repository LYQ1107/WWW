"""Construct missing selected V3 native features, never rescan for statistics.

The full audit intentionally retained no native feature archive. Continuous GMT
is reused to construct only the locked new events, with full prediction and
selected actual-solver equality to the sealed audit before attaching labels.
An evicted observation window bounds extra feature memory; only selected query
and latest-four candidate histories survive in the final payload.
"""
import argparse
import collections
import gzip
import time
import torch
from common import *
from audit_decision_trace import records
from audit_full_train_identity_events import original_truth, label_detectors
from collect_track_hard_events import pure_reference_identity

NEW=Path('/data1/liuyeqiang/WWW_trackvggt_runtime')


class ObservationWindow:
    def __init__(self):
        self.values={};self.peak=0

    def add(self,key,value):
        if key in self.values:raise ValueError('duplicate physical observation')
        self.values[key]=value;self.peak=max(self.peak,len(self.values))

    def evict(self,minimum_frame):
        for key in [key for key,value in self.values.items() if value['frame']<minimum_frame]:del self.values[key]


def selected_call(scores,logits,queries,sources,ids,tracks,block_sizes,window,counts,wanted):
    rows=[i for i,key in enumerate(queries) if key in wanted]
    if not rows:return None,[]
    order=torch.argsort(scores[rows],dim=1,descending=True,stable=True)
    top=order[:,:8].cpu().tolist();all_tracks=tracks.cpu().tolist();ref_ids=ids.cpu().tolist()
    selected_tracks={all_tracks[i] for indices in top for i in indices};histories={}
    for track in selected_tracks:
        history=[source for source,identity in zip(sources,ref_ids) if identity==track]
        recent=sorted(history,key=lambda key:(window.values[key]['frame'],window.values[key]['camera'],key))[-4:]
        histories[track]=dict(sources=[list(key) for key in recent],observed_length=int(counts.get(track,len(history))))
    call=dict(queries=[list(queries[i]) for i in rows],references=[dict(source=list(source),track=int(track)) for source,track in zip(sources,ref_ids)],
        reference_cameras=[window.values[key]['camera'] for key in sources],reference_times=[window.values[key]['frame'] for key in sources],
        reference_finite=[bool(torch.isfinite(window.values[key]['feature']).all() and window.values[key]['feature'].float().norm()>0) for key in sources],
        track_ids=list(map(int,all_tracks)),track_scores=scores[rows].detach().float().cpu().clone(),logits=logits[rows].detach().float().cpu().clone(),
        histories=histories,reference_block_sizes=block_sizes)
    events=[]
    for row,(original_row,indices) in enumerate(zip(rows,top)):
        score=scores[original_row];ranking=order[row]
        margin=float((score[ranking[0]]-score[ranking[1]])/(score[ranking[0]].abs()+1e-8)) if len(ranking)>1 else 1.
        events.append(dict(query_source=list(queries[original_row]),query_row=row,original_query_row=original_row,
            candidate_tracks=[int(all_tracks[i]) for i in indices],candidate_track_indices=indices,
            native_track_scores=score[indices].detach().float().cpu().tolist(),relative_margin=margin))
    return call,events


def retain_observations(window,call,event,kept):
    needed={tuple(event['query_source'])}
    for track in event['candidate_tracks']:needed.update(tuple(s) for s in call['histories'][track]['sources'])
    query_frame=window.values[tuple(event['query_source'])]['frame']
    for key in needed:
        if window.values[key]['frame']>query_frame:raise ValueError('future native history')
        kept[key]=window.values[key]


def main():
    p=argparse.ArgumentParser();p.add_argument('--video-id',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=['full','precheck-v2','precheck-v3'],default='full');a=p.parse_args()
    split=json.loads((ROOT/'manifests/geo_split_v1.json').read_text())
    if a.video_id not in set(split['association_train_video_ids'])-{12}:raise ValueError('new fit events only; validation12 is reused unchanged')
    baseline=NEW/'full_train_audit_v1'/f'scene{a.video_id}_full_v1';bm=json.loads((baseline/'manifest.json').read_text())
    if bm['status']!='COMPLETE_FULL_CAUSAL_AUDIT':raise ValueError('sealed complete native audit required')
    v3path=ROOT/'manifests/trackvggt_expanded_training_v3.json';v3=json.loads(v3path.read_text());v2path=ROOT/'results/trackvggt_expanded_cohort_v2.json'
    if v3['status']!='LOCKED_TRAIN_EVENT_INDEX_REQUIRES_SELECTED_FEATURE_CAPTURE':raise ValueError('unlocked index')
    v2=json.loads(v2path.read_text());probe_payloads={};probe_events={}
    if a.mode=='precheck-v2':
        for item in v2['fit']:
            if item['scene']!=a.video_id:continue
            path=item['event_file']
            if path not in probe_payloads:probe_payloads[path]=torch.load(path,map_location='cpu')
            payload=probe_payloads[path];e=payload['events'][item['event_index']];q=payload['observations'][tuple(e['query_source'])]
            if q['frame']<=40:probe_events[tuple(e['key'])]=(payload,e)
        wanted={key[1:] for key in probe_events}
    else:
        wanted={tuple(r['key'][1:]) for r in v3['additional_fit'] if r['scene']==a.video_id and (a.mode=='full' or r['frame']<=40)}
    if not wanted:raise ValueError('no locked requested query')
    if a.output.exists():raise FileExistsError(a.output)
    out=a.output;out.mkdir(parents=True);data=train_data();full_images=video_records(data,a.video_id)
    images=full_images if a.mode=='full' else video_records(data,a.video_id,40);by_image={i['image_id']:i for i in images};groups=collections.defaultdict(list)
    for image in images:groups[image['frame_id']].append(image)
    prov=provenance()
    for key in ['checkpoint_sha256','config_sha256','annotation_sha256','seed']:
        if prov[key]!=bm['provenance'][key]:raise ValueError('frozen source signature differs: '+key)
    for name in ['predictions.json','actual_decisions.jsonl.gz','detector_provenance.jsonl.gz','events.jsonl.gz']:
        if sha256(baseline/name)!=bm['artifacts'][name]:raise ValueError('sealed audit artifact changed')
    initial=dict(status='RUNNING_NEW_SELECTED_NATIVE_FEATURE_CONSTRUCTION',mode=a.mode,video_id=a.video_id,selected_queries=len(wanted),
        reference_manifest=str(baseline/'manifest.json'),reference_manifest_sha256=sha256(baseline/'manifest.json'),V3_index_sha256=sha256(v3path),
        full_video=a.mode=='full',causal_online=True,continuous_from_group1=True,teacher_calls=0,provenance=prov)
    save_json(out/'manifest.json',initial)
    from gtr.geometry.decision_trace import DecisionTrace
    from gtr.geometry.online_runner import CausalGMT
    from gtr.evaluation.mot_evaluation import custom_instances_to_coco_json
    cfg,model=load_gmt();window=ObservationWindow();pending_detectors={};kept={};calls={};captured=[];primary_counter=0;dimensions=set()
    trace=DecisionTrace(model,out);write_original=trace.write
    def write(stream,record):
        if stream is trace.detectors:
            pending_detectors[tuple(record['source'])]=record['detection']
        elif record['stage']!='run_memory_tracker' and any(tuple(q) in wanted for q in record['queries']):write_original(stream,record)
    trace.write=write;trace.install();infer_original=model.inference;asso_original=model.get_asso
    def infer(*args,**kwargs):
        result=infer_original(*args,**kwargs)
        for inst in result:
            dimensions.add(inst.reid_features.shape[-1])
            for index,source in enumerate(inst.decision_trace_source.cpu().tolist()):
                key=tuple(source);image=by_image[key[0]];public=pending_detectors.pop(key);box=public['bbox']
                window.add(key,dict(source=list(key),image_id=key[0],camera=image['view_id'],frame=image['frame_id'],path=image['file_name'],width=image['width'],height=image['height'],
                    box=[box[0],box[1],box[0]+box[2],box[1]+box[3]],feature=inst.reid_features[index].detach().cpu().to(torch.float16).clone()))
        return result
    def association(instances,k,**kwargs):
        nonlocal primary_counter
        result=asso_original(instances,k,**kwargs)
        if kwargs.get('geo_context',True) is False:return result
        index=primary_counter;primary_counter+=1;query=instances[k];queries=[tuple(q) for q in query.decision_trace_source.cpu().tolist()]
        if not any(q in wanted for q in queries):return result
        refs=[x for j,x in enumerate(instances) if j!=k];logits=result[0][0]
        sources=[tuple(s) for x in refs for s in x.decision_trace_source.cpu().tolist()]
        ids=torch.cat([x.track_ids for x in refs]);tracks=ids.unique(sorted=True)
        if tuple(logits.shape)!=(len(query),len(ids)):raise ValueError('actual logit layout changed')
        probabilities=torch.cat(model.roi_heads._activate_asso(logits.split([len(x) for x in refs],dim=1)),dim=1)
        scores=probabilities@(ids[:,None]==tracks[None,:]).float()
        call,events=selected_call(scores,logits,queries,sources,ids,tracks,[len(x) for x in refs],window,tracker.counts,wanted)
        if not events:return result
        calls[index]=call
        for event in events:
            event.update(key=[a.video_id,*event['query_source']],call_index=index)
            retain_observations(window,call,event,kept)
            captured.append(event)
        return result
    model.inference=infer;model.get_asso=association;seed_all(20261014);tracker=CausalGMT(model);predictions=[];started=time.perf_counter()
    try:
        for frame,group in sorted(groups.items()):
            outputs=tracker.step(map_records(cfg,group))
            for record,output in zip(group,outputs):predictions.extend(custom_instances_to_coco_json(output['instances'].to('cpu'),record['image_id']))
            window.evict(frame-model.test_len)
            if frame%100==0:print(json.dumps(dict(scene=a.video_id,groups=frame,selected=len(captured),window_observations=len(window.values),retained_observations=len(kept))),flush=True)
    finally:
        model.get_asso=asso_original;trace.close()
    torch.cuda.synchronize();seconds=time.perf_counter()-started
    (out/'actual_decisions.jsonl.gz').rename(out/'selected_actual_decisions.jsonl.gz')
    if list(records(out/'detector_provenance.jsonl.gz')):raise ValueError('unexpected unbounded detector archive')
    (out/'detector_provenance.jsonl.gz').unlink() # own empty temporary stream only
    save_json(out/'predictions.json',predictions)
    image_ids=set(by_image);expected=[r for r in json.loads((baseline/'predictions.json').read_text()) if r['image_id'] in image_ids]
    if predictions!=expected:raise ValueError('native feature capture changed sealed causal predictions; labels prohibited')
    actual=iter(records(out/'selected_actual_decisions.jsonl.gz'));selected_decisions=0
    for record in records(baseline/'actual_decisions.jsonl.gz'):
        if record['stage']=='run_memory_tracker' or not any(tuple(q) in wanted for q in record['queries']):continue
        if next(actual)!=record:raise ValueError('selected actual solver/history changed; labels prohibited')
        selected_decisions+=1
    try:next(actual);raise ValueError('extra selected solver call')
    except StopIteration:pass
    if {tuple(e['query_source']) for e in captured}!=wanted or len(captured)!=len(wanted):raise ValueError('missing or duplicate locked query')
    if dimensions!={1152} or pending_detectors:raise ValueError('native feature layout/proposal provenance changed')
    # Only now read labels. GT never selected an online candidate or history.
    truth,blocked,gt_hashes,gt_diagnostics=original_truth(data,a.video_id,full_images)
    labels,_,_,_=label_detectors(baseline,truth,blocked)
    audited={tuple(e['key']):e for e in records(baseline/'events.jsonl.gz') if tuple(e['key'][1:]) in wanted}
    for event in captured:
        call=calls[event['call_index']];target=audited[tuple(event['key'])];identity=labels[tuple(event['query_source'])];values=collections.defaultdict(list)
        for ref in call['references']:values[ref['track']].append(labels[tuple(ref['source'])])
        pure={t:pure_reference_identity(v) for t,v in values.items()};full_pos=[t for t,v in pure.items() if identity is not None and v==identity]
        positive=[t for t in event['candidate_tracks'] if t in full_pos];unknown=[t for t in event['candidate_tracks'] if pure[t] is None]
        local_null=identity is not None and not full_pos and all(v is not None for v in pure.values())
        if identity!=target['query_gt_identity'] or event['candidate_tracks']!=target['top8_tracks'] or set(full_pos)!=set(target['positive_tracks']):raise ValueError('locked purity/candidate labels differ')
        if local_null!=target['genuine_NULL_within_legal_candidates']:raise ValueError('NULL mask changed')
        event.update(query_gt_identity=identity,positive_tracks=positive,all_history_positive_tracks=full_pos,ambiguous_tracks=unknown,
            top8_positive=bool(positive),all_history_positive=bool(full_pos),native_top1_correct=target['ranking_state']=='correct_link',native_top1_state=target['ranking_state'],
            observed_positive_tracks_including_impure=target['observed_positive_including_impure'],loss_eligible=bool(positive) or local_null,
            known_unmatched_within_all_recorded_history=local_null,actual_primary_state=target['actual_primary_state'],actual_solver_scope='sealed unchanged actual primary solver before memory',
            selection_scope='locked V3 TRAIN physical query index; no new mining or validation resampling')
    probe_count=0
    for event in captured:
        if tuple(event['key']) not in probe_events:continue
        old,old_event=probe_events[tuple(event['key'])];old_call=old['calls'][old_event['call_index']];call=calls[event['call_index']]
        assert event['candidate_tracks']==old_event['candidate_tracks']
        assert torch.equal(call['logits'][event['query_row']],old_call['logits'][old_event['query_row']])
        assert torch.equal(call['track_scores'][event['query_row']],old_call['track_scores'][old_event['query_row']])
        for track in event['candidate_tracks']:assert call['histories'][track]==old_call['histories'][track]
        needed={tuple(event['query_source'])}|{tuple(s) for t in event['candidate_tracks'] for s in call['histories'][t]['sources']}
        for key in needed:assert torch.equal(kept[key]['feature'],old['observations'][key]['feature']) and kept[key]['box']==old['observations'][key]['box']
        probe_count+=1
    captured.sort(key=lambda e:e['key']);torch.save(dict(events=captured,observations=kept,calls=calls,feature_dim=1152),out/'events.pt')
    save_json(out/'event_index.json',captured)
    initial.update(status='COMPLETE_LOCKED_V3_SELECTED_NATIVE_FEATURES' if a.mode=='full' else 'PASS_SELECTED_FEATURE_PRECHECK',groups=len(groups),images=len(images),
        predictions_equal_sealed_native=True,full_prediction_file_bytes_exact=sha256(out/'predictions.json')==bm['predictions_sha256'] if a.mode=='full' else 'PREFIX_ONLY',
        selected_actual_decision_calls=selected_decisions,selected_actual_decisions_exact=True,selected_events=len(captured),native_feature_dim=1152,retained_observations=len(kept),
        extra_observation_window_peak=window.peak,extra_window_horizon_groups_after_group=model.test_len+1,extra_window_transient_horizon_groups_at_most=model.test_len+2,full_native_vector_archive=False,only_selected_query_rows_of_matrices=True,
        gt_labels_after_prediction_and_solver_parity=True,raw_gt_sha256=gt_hashes,GT_unreliable_frame_policy=gt_diagnostics,v2_control_events_exact=probe_count,
        future_history=False,GT_query_features=False,collection_seconds=seconds,speed_eligible=False,
        artifacts={p.name:sha256(p) for p in out.iterdir() if p.is_file() and p.name!='manifest.json'})
    save_json(out/'manifest.json',initial)
    print(json.dumps({k:initial[k] for k in ['status','video_id','groups','selected_events','retained_observations','extra_observation_window_peak','v2_control_events_exact','selected_actual_decisions_exact','collection_seconds']}),flush=True)


if __name__=='__main__':main()
