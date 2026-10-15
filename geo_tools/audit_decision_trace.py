"""Label actual recorded assignments after standalone prediction gates pass."""
import argparse
import collections
import gzip
import numpy as np
from scipy.optimize import linear_sum_assignment
from common import *
from evaluate_live_retrieval import iou_matrix
from audit_suite_settings import signature


def records(path):
    with gzip.open(path,'rt') as stream:
        for line in stream:yield json.loads(line)


def verify_run(run,reference):
    manifest=json.loads((run/'manifest.json').read_text());trace=json.loads((run/'decision_trace_manifest.json').read_text())
    ref=json.loads((reference.parent/'manifest.json').read_text())
    if not manifest['causal_online'] or trace['status']!='COMPLETE' or manifest['speed_eligible']:
        raise ValueError('complete speed-ineligible causal trace required')
    for name,digest in trace['artifacts'].items():
        if sha256(run/name)!=digest:raise ValueError('trace artifact changed')
    if sha256(run/'predictions.json')!=manifest['predictions_sha256'] or sha256(reference)!=ref['predictions_sha256']:
        raise ValueError('prediction changed')
    if signature(manifest)[0]!=signature(ref)[0]:raise ValueError('tracking configuration changed')
    if manifest['geo_failures']:raise ValueError('enhancement failed during trace')
    actual=json.loads((run/'predictions.json').read_text());ids=set(trace['image_ids'])
    # Include empty images in the gate via detector inference provenance count.
    data=train_data();images=video_records(data,manifest['video_id'])
    if not manifest['full_video']:
        maximum=max(image['frame_id'] for image in images if image['image_id'] in ids)
        images=[image for image in images if image['frame_id']<=maximum]
    if ids!={image['image_id'] for image in images}:raise ValueError('missing/unexpected observed image IDs')
    if len(images)!=manifest['images'] or trace['images']!=manifest['images']:
        raise ValueError('trace image scope mismatch')
    if actual!=[row for row in json.loads(reference.read_text()) if row['image_id'] in ids]:
        raise ValueError('actual tracing changed standalone predictions')
    return manifest,images


def decision_state(query_truth,references,selected):
    if query_truth is None:return 'unknown_query'
    labels={};uncertain=False
    for track,values in references.items():
        known={value for value in values if value is not None}
        pure=len(known)==1 and all(value is not None for value in values)
        labels[track]=next(iter(known)) if pure else None
        uncertain|=not pure
    if selected>=0:
        if labels[selected] is None:return 'ambiguous_reference'
        return 'correct_link' if labels[selected]==query_truth else 'wrong_link'
    if query_truth in labels.values():return 'missed_known_positive'
    return 'unknown_candidate_coverage' if uncertain else 'correct_reject_within_recorded_candidates'


def label_run(run,manifest,images):
    data=train_data();scene=manifest['video_id'];name=next(v['file_name'] for v in data['videos'] if v['id']==scene)
    by_view_frame={(row['view_id'],row['frame_id']):row['image_id'] for row in images}
    truth=collections.defaultdict(list);hashes={}
    for camera in sorted({row['view_id'] for row in images}):
        path=DATA/'train'/f'{name}_View{camera}'/'gt/gt.txt';hashes[str(path)]=sha256(path);seen=set()
        for line in path.read_text().splitlines():
            values=list(map(float,line.split(',')));frame,identity=map(int,values[:2])
            if (frame,identity) in seen:raise ValueError('duplicate raw GT')
            seen.add((frame,identity))
            if values[6]>0 and values[7]==1 and (camera,frame) in by_view_frame:
                truth[by_view_frame[camera,frame]].append((identity,values[2:6]))
    detections=collections.defaultdict(list);labels={}
    for row in records(run/'detector_provenance.jsonl.gz'):
        key=tuple(row['source'])
        if key in labels:raise ValueError('duplicate proposal provenance')
        labels[key]=None;detections[key[0]].append(row)
    for image,rows in detections.items():
        target=truth[image]
        if not target:continue
        overlaps=iou_matrix([row['detection']['bbox'] for row in rows],[entry[1] for entry in target])
        ri,ci=linear_sum_assignment(np.where(overlaps>=.5,1-overlaps,1e6))
        for i,j in zip(ri,ci):
            if overlaps[i,j]>=.5:labels[tuple(rows[i]['source'])]=target[j][0]
    outcomes={};counts=collections.Counter();bank_calls=0;accepted=0
    for record in records(run/'actual_decisions.jsonl.gz'):
        if record['stage']=='run_memory_tracker':
            bank_calls+=1;continue # averaged descriptor provenance alone cannot prove GT purity
        references=collections.defaultdict(list)
        for ref in record['references']:references[ref['track']].append(labels.get(tuple(ref['source'])))
        for source,selected,final in zip(record['queries'],record['accepted_track_ids'],record['final_track_ids']):
            key=tuple(source)
            if key in outcomes:raise ValueError('proposal assigned twice by primary tracker')
            state=decision_state(labels.get(key),references,selected);counts[state]+=1
            accepted+=selected>=0
            outcomes[key]=dict(state=state,accepted=selected>=0,final_equals_primary=final==selected,stage=record['stage'])
    return outcomes,dict(counts= dict(counts),primary_query_assignments=len(outcomes),accepted_primary_assignments=accepted,
                         memory_solver_calls_excluded_from_gt_correctness=bank_calls,raw_gt_sha256=hashes)


def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline-run',type=Path,required=True);p.add_argument('--method-run',type=Path,required=True)
    p.add_argument('--baseline-reference',type=Path,required=True);p.add_argument('--method-reference',type=Path,required=True);p.add_argument('--report-output',type=Path,required=True);a=p.parse_args()
    runs=[a.baseline_run,a.method_run];references=[a.baseline_reference,a.method_reference]
    verified=[verify_run(run,ref) for run,ref in zip(runs,references)]
    if [image['image_id'] for image in verified[0][1]]!=[image['image_id'] for image in verified[1][1]]:raise ValueError('different image scopes')
    # Only after both standalone output/configuration gates: read raw GT.
    labelled=[label_run(run,*pair) for run,pair in zip(runs,verified)]
    left,right=[pair[0] for pair in labelled];common=left.keys()&right.keys();transitions=collections.Counter()
    good={'correct_link','correct_reject_within_recorded_candidates'};bad={'wrong_link','missed_known_positive'};fixes=harms=unknown=0
    for key in common:
        b,m=left[key]['state'],right[key]['state'];transitions[b+' -> '+m]+=1
        fixes+=b in bad and m in good;harms+=b in good and m in bad
        unknown+=b not in good|bad or m not in good|bad
    save_json(a.report_output,dict(status='PASS_ACTUAL_SOLVER_DIAGNOSTIC',video_id=verified[0][0]['video_id'],full_video=all(pair[0]['full_video'] for pair in verified),full_strict_suite=False,
        runs=[dict(run=str(run),**labelled[i][1],manifest_sha256=sha256(run/'manifest.json'),trace_manifest_sha256=sha256(run/'decision_trace_manifest.json')) for i,run in enumerate(runs)],
        common_primary_queries=len(common),baseline_only_queries=len(left.keys()-right.keys()),method_only_queries=len(right.keys()-left.keys()),
        baseline_wrong_method_correct=fixes,baseline_correct_method_wrong=harms,unknown_either=unknown,transition_counts=dict(transitions),
        reference_artifacts={str(path):sha256(path) for path in references},
        scope='actual Hungarian solver calls and strict threshold before bank fallback/new-ID allocation; candidate truth uses only exact current solver reference proposals, one-to-one IoU>=.5 labels after both prediction gates',
        limitations='known-pure reference tracks only; mixed/unmatched references and unmatched queries stay unknown; no-match correctness is conditional on recorded candidates, not global candidate recall; averaged-descriptor memory decisions excluded; no claim of full tracking correctness or all strict scenes from this scene1 diagnostic; separate runs have their own histories',provenance=provenance()))


if __name__=='__main__':main()
