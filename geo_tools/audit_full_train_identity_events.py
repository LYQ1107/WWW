"""Continuous real GMT audit. No encoder calls or persistent feature/matrix cache.

DecisionTrace/CausalGMT and the collector's per-block activation are reused.
Only compact rank/source indices are streamed during inference. Original GT is
opened for labeling only after full inference and an independent OFF40 gate.
"""
import argparse
import collections
import gzip
import hashlib
import heapq
import itertools
import time
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from common import *
from audit_decision_trace import records, decision_state
from collect_track_hard_events import pure_reference_identity
from census_trackvggt_train import patch_count, patch_bin
from evaluate_live_retrieval import iou_matrix


def original_truth(data, scene, images):
    """No repair: whole duplicate/invalid camera-frames are unreliable."""
    name = next(v['file_name'] for v in data['videos'] if v['id']==scene)
    meta = {(i['view_id'], i['frame_id']):i['image_id'] for i in images}
    prior = {(r['video_id'],r['view_id']):r for r in json.loads((ROOT/'results/train_gt_quality_audit.json').read_text())['rows']}
    truth = collections.defaultdict(list); blocked = set(); hashes = {}; diagnostics = {}
    for camera in sorted({i['view_id'] for i in images}):
        path = DATA/'train'/f'{name}_View{camera}'/'gt/gt.txt'; digest = sha256(path)
        if digest != prior[scene,camera]['raw_sha256']:
            raise ValueError('raw GT hash changed; prior audit cannot be reused')
        hashes[str(path)] = digest; seen = set(); conflicts = set(); invalid = set()
        for line in path.read_text().splitlines():
            v = list(map(float,line.split(',')))
            if len(v)<8 or not np.isfinite(v).all():
                raise ValueError('unlocatable malformed GT; alignment unverified')
            frame, identity = int(v[0]), int(v[1])
            # Locate conflict frames only in files already known to have duplicates.
            if prior[scene,camera]['duplicate_identity_rows']:
                if (frame,identity) in seen:conflicts.add(frame)
                seen.add((frame,identity))
            if v[4]<=0 or v[5]<=0 or frame<1 or identity<0:
                invalid.add(frame); continue
            if v[6]>0 and v[7]==1 and (camera,frame) in meta:
                truth[meta[camera,frame]].append((identity,v[2:6]))
        blocked.update(meta[camera,f] for f in conflicts|invalid if (camera,f) in meta)
        diagnostics[str(camera)] = dict(duplicate_frame_count=len(conflicts), invalid_frame_count=len(invalid), duplicate_frames=sorted(conflicts), invalid_frames=sorted(invalid))
    return truth, blocked, hashes, diagnostics


def label_detectors(out, truth, blocked):
    labels = {}; reasons = {}; boxes = {}; matched_gt = set(); n = 0
    # DecisionTrace writes each image contiguously, including zero detections
    # indirectly via the independently verified image scope.
    for image, iterator in itertools.groupby(records(out/'detector_provenance.jsonl.gz'),key=lambda r:r['source'][0]):
        rows = list(iterator); targets = truth[image]
        for r in rows:
            key = tuple(r['source'])
            if key in labels:raise ValueError('duplicate physical proposal')
            labels[key] = None; boxes[key] = r['detection']['bbox']
            reasons[key] = 'unreliable_raw_GT_frame' if image in blocked else 'no_one_to_one_IoU_match'
        if image in blocked or not targets or not rows:continue
        overlap = iou_matrix([r['detection']['bbox'] for r in rows],[t[1] for t in targets])
        ri, ci = linear_sum_assignment(np.where(overlap>=.5,1-overlap,1e6))
        for i,j in zip(ri,ci):
            if overlap[i,j]>=.5:
                key=tuple(rows[i]['source']);labels[key]=targets[j][0]; reasons[key]='known_one_to_one_IoU_ge_0.5'; matched_gt.add((image,int(j)))
        n += len(targets)
    eligible_gt = sum(len(v) for image,v in truth.items() if image not in blocked)
    return labels,reasons,boxes,dict(reliable_GT_target_rows=eligible_gt, matched_GT_target_rows=len(matched_gt), unmatched_GT_target_rows=eligible_gt-len(matched_gt),
        unmatched_GT_scope='No detector proposal matched at one-to-one IoU>=.5, not identity-query failures; includes confidence/localization misses. Conflict frames excluded and reported.')


def annotate_query(truth, pure, ref_truth, ranked, selected):
    positives = [track for track,value in pure.items() if truth is not None and value==truth]
    observed = [track for track,values in ref_truth.items() if truth is not None and truth in values]
    rank_state = 'no_candidate' if not ranked else 'unknown_query' if truth is None else 'ambiguous_reference' if pure[ranked[0]] is None else 'correct_link' if pure[ranked[0]]==truth else 'wrong_link'
    state = decision_state(truth,ref_truth,selected)
    wrong = state in ('wrong_link','missed_known_positive')
    types = []
    if positives and rank_state=='wrong_link':types.append('A')
    if positives and selected not in positives:types.append('B')
    if positives and not set(positives)&set(ranked[:8]):types.append('C')
    if truth is not None and not positives:types.append('D')
    if any(value is None for value in pure.values()):types.append('E')
    if truth is None or state in ('ambiguous_reference','unknown_candidate_coverage'):types.append('F')
    return dict(positive_tracks=positives, observed_positive_including_impure=observed, ranking_state=rank_state, actual_primary_state=state,
        types=types, known_wrong=wrong, correctable_known_wrong=wrong and bool(positives), genuine_NULL_within_legal_candidates=truth is not None and not observed and all(v is not None for v in pure.values()))


class BoundedExamples:
    def __init__(self, limit):
        self.quotas = dict(zip(['hard','ambiguous','moderate','easy','missing'],[int(limit*.4),int(limit*.15),int(limit*.2),int(limit*.15),limit-int(limit*.4)-int(limit*.15)-int(limit*.2)-int(limit*.15)]))
        self.heaps = collections.defaultdict(list)

    def add(self,row):
        category = 'hard' if row['correctable_known_wrong'] or 'A' in row['types'] else 'ambiguous' if 'E' in row['types'] or 'F' in row['types'] else 'missing' if 'D' in row['types'] else 'moderate' if row['relative_margin']<.2 else 'easy'
        size = self.quotas[category]
        if not size:return
        priority = int(hashlib.sha256(json.dumps(row['key']).encode()).hexdigest(),16)
        item=(-priority,tuple(row['key']),row); heap=self.heaps[category]
        if len(heap)<size:heapq.heappush(heap,item)
        elif item[:2]>heap[0][:2]:heapq.heapreplace(heap,item)

    def rows(self):
        return [dict(**row, retention_category=category) for category,heap in self.heaps.items() for _,_,row in sorted(heap,reverse=True)]


def summarize(out,data,scene,images,max_retained):
    truth,blocked,gt_hashes,gt_diagnostics = original_truth(data,scene,images)
    labels,reasons,boxes,detector_stats = label_detectors(out,truth,blocked)
    metadata = {i['image_id']:i for i in images}; counters=collections.Counter(); states=collections.Counter(); subtypes=collections.Counter()
    types=collections.Counter(); sizes=collections.defaultdict(collections.Counter); by_identity=collections.defaultdict(collections.Counter)
    retained=BoundedExamples(max_retained); events=0; memory=collections.Counter(); ranks=iter(records(out/'rankings.jsonl.gz'))
    # Past-only output anchors: add this group only after all its queries.
    output_history=collections.defaultdict(set); current_frame=None; pending=[]; final_states=collections.Counter()
    def flush_pending():
        for track,value in pending:output_history[track].add(value)
        pending.clear()
    with gzip.open(out/'events.jsonl.gz','xt') as stream:
        primary_index=0
        for decision in records(out/'actual_decisions.jsonl.gz'):
            if decision['stage']=='run_memory_tracker':
                memory['solver_calls']+=1;memory['queries']+=len(decision['queries']);memory['accepted']+=sum(t>=0 for t in decision['accepted_track_ids'])
                continue # averaged descriptor source tags cannot prove GT purity
            ranking=next(ranks)
            if ranking['queries']!=decision['queries']:raise ValueError('primary rank/solver order mismatch')
            ref_truth=collections.defaultdict(list); ref_cameras=collections.defaultdict(set)
            for ref in decision['references']:
                key=tuple(ref['source']);ref_truth[ref['track']].append(labels[key]);ref_cameras[ref['track']].add(metadata[key[0]]['view_id'])
            pure={track:pure_reference_identity(v) for track,v in ref_truth.items()}
            assigned={r['row']:r for r in decision['hungarian_pairs']}
            for qi,source in enumerate(decision['queries']):
                key=tuple(source);q=metadata[key[0]];identity=labels[key];selected=decision['accepted_track_ids'][qi];final=decision['final_track_ids'][qi]
                frame=q['frame_id']
                if frame!=current_frame:
                    flush_pending();current_frame=frame
                if any(metadata[ref['source'][0]]['frame_id']>frame for ref in decision['references']):raise ValueError('future reference in native legal history')
                ranked=ranking['ranked_tracks'][qi];annotation=annotate_query(identity,pure,ref_truth,ranked,selected);positives=annotation['positive_tracks']
                top8=ranked[:8];known=identity is not None;has_pos=bool(positives);wrong=annotation['known_wrong'];counter=counters
                counter['N_total']+=1;counter['N_known']+=known;counter['N_wrong']+=wrong;counter['N_wrong_eligible']+=wrong
                counter['N_rank_correctable']+='A' in annotation['types'];counter['N_assignment_correctable']+='B' in annotation['types']
                counter['N_missing_candidate']+='D' in annotation['types'];counter['N_impure']+='E' in annotation['types'];counter['N_unknown']+='F' in annotation['types']
                counter['N_pure_positive_full']+=has_pos;counter['N_correctable_known_wrong']+=annotation['correctable_known_wrong']
                counter['N_top8_correctable_known_wrong']+=annotation['correctable_known_wrong'] and bool(set(positives)&set(top8))
                counter['N_known_ranking_wrong']+=annotation['ranking_state']=='wrong_link'
                counter['N_known_primary_wrong_link']+=annotation['actual_primary_state']=='wrong_link';counter['N_known_primary_missed_positive']+=annotation['actual_primary_state']=='missed_known_positive'
                counter['N_current_unknown']+=not known;counter['N_genuine_NULL_within_legal_candidates']+=annotation['genuine_NULL_within_legal_candidates']
                for k in [4,8,16]:counter[f'N_top{k}']+=bool(set(positives)&set(ranked[:k]))
                positive_cross=[t for t in positives if any(c!=q['view_id'] for c in ref_cameras[t])]
                positive_same=[t for t in positives if q['view_id'] in ref_cameras[t]]
                counter['N_cross_camera']+=has_pos and bool(positive_cross) and not positive_same
                counter['N_same_camera']+=bool(positive_same);counter['N_both_camera_positive']+=bool(positive_cross and positive_same)
                counter['N_cross_camera_candidate_available']+=any(c!=q['view_id'] for v in ref_cameras.values() for c in v)
                counter['N_memory_or_newID_changed_primary']+=final!=selected;counter['N_primary_accepted']+=selected>=0
                counter['N_hungarian_assigned']+=qi in assigned;counter['N_threshold_rejected']+=qi in assigned and not assigned[qi]['accepted']
                counter['N_hungarian_unassigned']+=qi not in assigned
                raw=ranking['raw_top_track'][qi]
                rawstate='no_candidate' if raw is None else 'unknown_query' if not known else 'ambiguous_reference' if pure[raw] is None else 'correct_link' if pure[raw]==identity else 'wrong_link'
                for prefix,state in [('raw',rawstate),('GTA',annotation['ranking_state'])]:
                    counter[f'N_{prefix}_rank_known_evaluable']+=state in ('correct_link','wrong_link')
                    counter[f'N_{prefix}_rank_known_correct']+=state=='correct_link'
                evidence=[]
                if 'B' in annotation['types']:
                    if qi not in assigned:evidence.append('Hungarian_unassigned_row')
                    if qi in assigned and not assigned[qi]['accepted']:evidence.append('selected_pair_below_actual_threshold')
                    if set(positives)&set(ranking['threshold_pass_tracks'][qi]):evidence.append('pure_positive_passes_threshold_but_solver_does_not_accept_it')
                    if annotation['ranking_state']=='correct_link':evidence.append('pure_positive_row_top1_not_accepted_Hungarian_competition')
                    if ranking['null_dominates_all_blocks'][qi]:evidence.append('NULL_logit_dominates_all_reference_logits')
                    pair=assigned.get(qi)
                    if pair and not pair['accepted'] and pair['score']>ranking['base_threshold'] and pair['threshold']>ranking['base_threshold']:evidence.append('multiplicity_threshold_changes_selected_pair_acceptance')
                    if not evidence:evidence.append('score_or_competition_cause_unresolved')
                subtypes.update(evidence);types.update(annotation['types']);states[annotation['actual_primary_state']]+=1
                box=boxes[key];coverage=patch_count(box,q['width'],q['height']);size=patch_bin(coverage)
                sizes[size].update(dict(queries=1,known=int(known),known_wrong=int(wrong),correctable_known_wrong=int(annotation['correctable_known_wrong']),pure_positive_full=int(has_pos),top8_positive=int(bool(set(positives)&set(top8))),GTA_top1_correct=int(annotation['ranking_state']=='correct_link'),GTA_top1_evaluable=int(annotation['ranking_state'] in ('correct_link','wrong_link'))))
                if known:by_identity[identity].update(dict(queries=1,known_wrong=int(wrong),correctable=int(annotation['correctable_known_wrong']),rank_error=int('A' in annotation['types'])))
                anchor=output_history.get(final,set())
                finalstate='unknown_query' if not known else 'no_previous_group_output_anchor' if not anchor else 'ambiguous_output_anchor' if None in anchor or len(anchor)!=1 else 'consistent_previous_output_identity' if identity in anchor else 'conflicting_previous_output_identity'
                final_states[finalstate]+=1;pending.append((final,identity))
                margin=ranking['relative_margin'][qi]
                row=dict(key=[scene,*key],frame=frame,camera=q['view_id'],query_gt_identity=identity,query_label_reason=reasons[key],box=box,patch_centers=coverage,patch_bin=size,
                    **annotation,top4_tracks=ranked[:4],top8_tracks=top8,top16_tracks=ranked[:16],candidate_count=len(ranked),
                    raw_ranking_state=rawstate,accepted_primary_track=selected,final_output_track=final,final_output_anchor_state=finalstate,
                    solver_pair=assigned.get(qi),failure_evidence=evidence,relative_margin=margin,positive_cross_only=bool(positive_cross) and not positive_same,
                    pure_positive_top8=bool(set(positives)&set(top8)),source_primary_call=primary_index)
                # All actual queries get a lightweight index, not just retained examples.
                stream.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n');retained.add(row);events+=1
            primary_index+=1
        try:next(ranks);raise ValueError('unconsumed primary ranking')
        except StopIteration:pass
    expected=['N_total','N_known','N_wrong','N_wrong_eligible','N_rank_correctable','N_assignment_correctable','N_missing_candidate','N_impure','N_unknown','N_pure_positive_full','N_correctable_known_wrong','N_top8_correctable_known_wrong','N_top4','N_top8','N_top16','N_cross_camera','N_same_camera']
    for key in expected:counters[key]+=0
    rates={f'CandidateCoverage@{k}':dict(numerator=counters[f'N_top{k}'],denominator=counters['N_pure_positive_full'],value=counters[f'N_top{k}']/counters['N_pure_positive_full'] if counters['N_pure_positive_full'] else None) for k in [4,8,16]}
    rates['CorrectableErrorRate']=dict(numerator=counters['N_correctable_known_wrong'],denominator=counters['N_wrong_eligible'],value=counters['N_correctable_known_wrong']/counters['N_wrong_eligible'] if counters['N_wrong_eligible'] else None)
    save_json(out/'retained_examples.json',retained.rows())
    return dict(counts=dict(counters),primary_states=dict(states),event_types_overlapping=dict(types),assignment_failure_evidence_overlapping=dict(subtypes),rates=rates,
        patch_bins={k:dict(v) for k,v in sizes.items()},identity_statistics={str(k):dict(v) for k,v in by_identity.items()},memory_decisions=dict(memory),
        final_output_previous_group_anchor_states=dict(final_states),final_output_GT_numeric_ID_correctness='NOT_DEFINED_PREDICTED_IDS_ARE_PERMUTATION_INVARIANT',
        final_output_identity_metric='NOT_MEASURED_BY_THIS_QUERY_AUDIT',raw_gt_sha256=gt_hashes,GT_unreliable_frames=sorted(blocked),GT_frame_diagnostics=gt_diagnostics,
        detector_coverage=detector_stats,retained_events=len(retained.rows()),definitions=dict(
            N_wrong='Accepted pure known different identity OR rejected despite a pure positive; excludes unknown/impure selected references and unproven new-target rejections.',
            N_unknown='Unknown current GT OR selected impure reference OR rejected with uncertain history and no pure positive; overlaps N_impure, not every query with an impure alternative.',
            N_cross_camera='Pure positive exists only through other-camera references; N_same_camera has at least one pure same-camera positive; both-camera positive is a separate subset.',
            type_D='No pure positive in current complete legal primary history; includes correct observation trapped in impure track; causes not inferred.',
            purity='All source observations of that track in the actual primary candidate context must have the same known GT; not global future purity.',
            raw_ranking='Maximum raw reference-proposal logit mapped to its predicted track; GTA ranking uses actual per-frame NULL softmax then track sum.',
            NULL='No positive observation anywhere in reliable pure candidate references is known NULL only within this legal candidate set, not certified new identity.',
            thresholds='Actual strict > and reference multiplicity boundary; multiplicity is not total lifetime track length. No other minimum-track-length filtering in this causal runner.',
            final_anchor='Past completed synchronized groups, actual primary-query outputs only (initial largest-view identity initialization is not a primary query); same predicted final ID. Diagnostic consistency, not solver purity or CVIDF1.',
            overlap='A–F, ranking failures, assignment failures, unknown/impure and candidate coverage overlap. Do not sum as disjoint events.',
            offline_headroom='GT-assisted fixed native-history query diagnostic, not an online oracle or HOTA/CVIDF1 bound.'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--video-id',type=int,required=True);p.add_argument('--full-video',action='store_true')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,default=20261014);p.add_argument('--summary-only',action='store_true')
    p.add_argument('--max-retained-events',type=int,default=1000);p.add_argument('--precheck-groups',type=int,default=40)
    a=p.parse_args();split=json.loads((ROOT/'manifests/geo_split_v1.json').read_text())
    if a.video_id not in split['association_train_video_ids'] or a.seed!=20261014 or not 0<=a.max_retained_events<=2000:
        raise ValueError('fixed association-TRAIN audit scope/seed/retention required')
    if not a.full_video and a.precheck_groups!=40:raise ValueError('finite precheck is exactly40 groups')
    out=a.output
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    started=time.perf_counter();data=train_data();full_images=video_records(data,a.video_id)
    images=full_images if a.full_video else video_records(data,a.video_id,a.precheck_groups)
    groups=collections.defaultdict(list)
    for image in images:groups[image['frame_id']].append(image)
    if min(groups)!=1 or sorted(groups)!=list(range(1,len(groups)+1)):raise ValueError('nonconsecutive or unverified full frame scope')
    initial=dict(status='RUNNING_GT_NOT_READ',video_id=a.video_id,full_video=a.full_video,groups_planned=len(groups),images_planned=len(images),seed=a.seed,
                 summary_only=a.summary_only,max_retained_events=a.max_retained_events,causal_online=True,continuous_from_first_frame=True,provenance=provenance(a.seed))
    save_json(out/'manifest.json',initial)
    from gtr.geometry.decision_trace import DecisionTrace
    from gtr.geometry.online_runner import CausalGMT
    from gtr.evaluation.mot_evaluation import custom_instances_to_coco_json
    cfg,model=load_gmt(seed=a.seed);trace=DecisionTrace(model,out);trace.install();original_write=trace.write;original_association=model.get_asso
    ranking_stream=gzip.open(out/'rankings.jsonl.gz','xt');pending=[];primary_calls=0
    def association(instances,k,**kwargs):
        result=original_association(instances,k,**kwargs)
        if kwargs.get('geo_context',True) is False:return result
        query=instances[k];refs=[x for j,x in enumerate(instances) if j!=k];logits=result[0][0]
        ids=torch.cat([x.track_ids for x in refs]) if refs else torch.empty(0,dtype=torch.long,device=logits.device)
        unique=ids.unique(sorted=True)
        if tuple(logits.shape)!=(len(query),len(ids)):raise ValueError('real primary shape mismatch')
        if refs:
            probabilities=torch.cat(model.roi_heads._activate_asso(logits.split([len(x) for x in refs],dim=1)),dim=1)
            membership=(ids[:,None]==unique[None,:]).float();scores=probabilities@membership
        else:scores=logits.new_zeros((len(query),0));membership=logits.new_zeros((0,0))
        order=torch.argsort(scores,dim=1,descending=True,stable=True)
        base=float(model.overlap_thresh);bound=base if model.not_mult_thresh else base*membership.sum(0)
        rankids=unique[order].detach().cpu().tolist();raw=[];margin=[];passed=[];null=[]
        for qi in range(len(query)):
            raw.append(int(ids[logits[qi].argmax()]) if len(ids) else None)
            top=order[qi]
            margin.append(float((scores[qi,top[0]]-scores[qi,top[1]])/(scores[qi,top[0]].abs()+1e-8)) if len(top)>1 else 1.)
            passed.append(unique[scores[qi]>bound].detach().cpu().tolist())
            null.append(bool((logits[qi]<=0).all()) if len(ids) else False)
        pending.append(dict(queries=query.decision_trace_source.detach().cpu().tolist(),ranked_tracks=rankids,raw_top_track=raw,relative_margin=margin,
            threshold_pass_tracks=passed,null_dominates_all_blocks=null,base_threshold=base))
        if len(pending)!=1:raise ValueError('unexpected overlapping primary get_asso calls')
        return result
    def write(stream,record):
        nonlocal primary_calls
        original_write(stream,record)
        if stream is trace.decisions and record['stage']!='run_memory_tracker':
            ranking=pending.pop()
            if ranking['queries']!=record['queries']:raise ValueError('raw observation/solver mismatch')
            ranking_stream.write(json.dumps(ranking,separators=(',',':'),allow_nan=False)+'\n');primary_calls+=1
    trace.write=write;model.get_asso=association;seed_all(a.seed);tracker=CausalGMT(model);predictions=[];inference_started=time.perf_counter()
    try:
        for frame,group in sorted(groups.items()):
            outputs=tracker.step(map_records(cfg,group))
            for record,output in zip(group,outputs):predictions.extend(custom_instances_to_coco_json(output['instances'].to('cpu'),record['image_id']))
            if tracker.frame%100==0:print(json.dumps(dict(scene=a.video_id,groups=tracker.frame,total_groups=len(groups),primary_calls=primary_calls)),flush=True)
    finally:
        model.get_asso=original_association;trace.close();ranking_stream.close()
    if pending:raise ValueError('unconsumed primary audit call')
    torch.cuda.synchronize();inference_seconds=time.perf_counter()-inference_started
    save_json(out/'predictions.json',predictions)
    # Fresh standalone OFF run after removing every hook, before opening any GT.
    seed_all(a.seed);baseline=CausalGMT(model);prefix=[];prefix_images=set()
    for group in list(dict(sorted(groups.items())).values())[:40]:
        outputs=baseline.step(map_records(cfg,group))
        for record,output in zip(group,outputs):
            prefix_images.add(record['image_id']);prefix.extend(custom_instances_to_coco_json(output['instances'].to('cpu'),record['image_id']))
    if prefix!=[r for r in predictions if r['image_id'] in prefix_images]:raise ValueError('read-only OFF40 raw prediction parity failed; GT labeling prohibited')
    existing={4:'fit4_prefix400_known_v2',5:'fit5_prefix400_v1',6:'fit6_prefix400_known_v2',7:'fit7_prefix400_known_v2',8:'fit8_prefix400_v1',12:'validation12_prefix400_v1'}
    old_parity='NOT_APPLICABLE'
    if a.video_id in existing:
        folder=Path('/data1/liuyeqiang/WWW_trackvggt_runtime/events')/existing[a.video_id]
        m=json.loads((folder/'manifest.json').read_text());old=folder/'predictions.json'
        if sha256(old)!=m['artifacts']['predictions.json']:raise ValueError('old prefix prediction changed')
        image_ids={r['image_id'] for r in full_images if r['frame_id']<=min(400,len(groups))}
        prior_rows=[r for r in json.loads(old.read_text()) if r['image_id'] in image_ids]
        if prior_rows!=[r for r in predictions if r['image_id'] in image_ids]:raise ValueError('existing400 prefix mismatch; GT labeling prohibited')
        old_parity=dict(exact=True,groups=min(400,len(groups)),reference=str(old),sha256=sha256(old))
    initial.update(status='INFERENCE_PARITY_VERIFIED_PENDING_POST_LABELS',groups=len(groups),images=len(images),inference_seconds=inference_seconds,prefix40_raw_prediction_exact=True,existing_prefix_parity=old_parity)
    save_json(out/'manifest.json',initial)
    summary=summarize(out,data,a.video_id,images,a.max_retained_events)
    save_json(out/'summary.json',summary)
    initial.update(status='COMPLETE_FULL_CAUSAL_AUDIT' if a.full_video else 'COMPLETE_PRECHECK_ONLY',full_video=len(images)==len(full_images) and a.full_video,
        wall_seconds=time.perf_counter()-started,raw_GT_labeling_after_full_inference_and_OFF_parity=True,GT_features_to_tracker=False,future_query_features=False,
        vggt_calls=0,dinov2_calls=0,speed_eligible=False,persistent_native_feature_vectors=0,persistent_dense_candidate_score_matrices=0,
        predictions_sha256=sha256(out/'predictions.json'),artifacts={p.name:sha256(p) for p in out.iterdir() if p.is_file() and p.name!='manifest.json'})
    save_json(out/'manifest.json',initial)
    print(json.dumps(dict(scene=a.video_id,status=initial['status'],groups=len(groups),counts=summary['counts'],rates=summary['rates'])),flush=True)


if __name__=='__main__':main()
