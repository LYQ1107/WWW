"""Post-inference offline labels, all-query coverage and immutable handoff.

Reuses sealed query certifications, never re-mines the 12-scene audit. Only the
initial largest-camera births (which have no association queries) need the
original one-frame detector/GT certification. No GT enters the input consumer.
"""
import argparse
import collections
import gzip
import torch
from common import *
from validate_dev12_native_inputs import validate, rows, file_sha
from dev12_native_schema import INTERFACE, digest_json

BASE=Path('/data1/liuyeqiang/WWW_trackvggt_runtime')
AUDIT=BASE/'full_train_audit_v1/scene12_full_v1'


def write_rows(path, values):
    with gzip.open(path,'xt') as stream:
        for value in values:stream.write(json.dumps(value,separators=(',',':'),allow_nan=False)+'\n')


def attach_labels(root, queries, calls, observations):
    audited={tuple(row['key']):row for row in rows(AUDIT/'events.jsonl.gz')}
    if set(audited)!=set(queries):raise ValueError('full original native query keys differ')
    labels={tuple(row['key'][1:]):row['query_gt_identity'] for row in audited.values()}
    reasons={tuple(row['key'][1:]):row['query_label_reason'] for row in audited.values()}
    initial=set(observations)-set(labels)
    if any(observations[key]['frame']!=1 for key in initial):raise ValueError('unexpected non-query observation beyond initial births')
    # Original raw GT only now, after complete inference and independent input
    # validation. All regular query labels are reused unchanged by physical key.
    from audit_full_train_identity_events import original_truth
    from evaluate_live_retrieval import iou_matrix
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    truth,blocked,gt_hashes,diagnostics=original_truth(train_data(),12,video_records(train_data(),12))
    for image in sorted({key[0] for key in initial}):
        keys=sorted(key for key in initial if key[0]==image);targets=truth[image]
        for key in keys:labels[key]=None;reasons[key]='unknown_initial_birth'
        if image in blocked or not targets:continue
        overlap=iou_matrix([observations[key]['bbox_xywh'] for key in keys],[t[1] for t in targets])
        ri,ci=linear_sum_assignment(np.where(overlap>=.5,1-overlap,1e6))
        for i,j in zip(ri,ci):
            if overlap[i,j]>=.5:labels[keys[i]]=targets[j][0];reasons[keys[i]]='known_initial_birth_one_to_one_IoU_ge_0.5'
    destination=root/'offline_labels';destination.mkdir()
    write_rows(destination/'observations.jsonl.gz',[dict(source=list(key),gt_identity=labels[key],reason=reasons[key]) for key in sorted(labels)])
    result=[];errors=[];state_counts=collections.Counter();memory_by_key={}
    for call in calls.values():
        if call['stage']=='run_memory_tracker':
            for source,identity in zip(call['queries'],call['decision']['accepted_track_ids']):memory_by_key[tuple(source)]=identity
    for key,query in queries.items():
        event=audited[key];call=calls[query['call_id']];qi=query['query_row'];references=collections.defaultdict(list)
        for ref in call['references']:references[ref['track']].append(labels[tuple(ref['source'])])
        pure={track:values[0] if values and values[0] is not None and all(v==values[0] for v in values) else None for track,values in references.items()}
        identity=event['query_gt_identity'];positive=[track for track,value in pure.items() if identity is not None and value==identity]
        if set(positive)!=set(event['positive_tracks']):raise ValueError('reused original full-history certification differs')
        impure=[track for track,values in references.items() if len({v for v in values if v is not None})>1]
        ambiguous=[track for track,value in pure.items() if value is None]
        aliases=collections.defaultdict(list)
        for track,value in pure.items():
            if value is not None:aliases[value].append(track)
        primary=query['primary_accepted_track'];final=query['final_track'];bank=primary<0 and memory_by_key.get(key[1:],-1)==final and final>=0
        newborn=primary<0 and not bank
        if newborn and final<=call['id_count_before']:raise ValueError('unexplained final ID after primary rejection')
        tags=[]
        state=event['actual_primary_state']
        mapping=dict(correct_link='CORRECT_LINK',wrong_link='WRONG_LINK',missed_known_positive='MISSED_KNOWN_POSITIVE',
            correct_reject_within_recorded_candidates='CORRECT_REJECT',ambiguous_reference='AMBIGUOUS_REFERENCE',unknown_query='UNKNOWN_QUERY',unknown_candidate_coverage='UNKNOWN_CANDIDATE_COVERAGE')
        tags.append(mapping[state])
        if identity is not None and not positive:tags.append('CANDIDATE_MISSING')
        if impure:tags.append('IMPURE_HISTORY')
        if any(len(v)>1 for v in aliases.values()):tags.append('ALIASED_IDENTITY')
        if bank:tags.append('BANK_RESTORED')
        if newborn:tags.append('NEWBORN')
        threshold_positive=[track for track in positive if bool(call['track_scores'][qi,call['track_ids'].index(track)]>call['thresholds'][call['track_ids'].index(track)])]
        row=dict(key=list(key),call_id=call['call_id'],query_row=qi,gt_identity=identity,primary_state=mapping[state],tags=tags,
            positive_tracks=positive,ambiguous_tracks=ambiguous,impure_tracks=impure,pure_track_gt={str(t):v for t,v in pure.items()},
            aliased_pure_identities={str(v):tracks for v,tracks in aliases.items() if len(tracks)>1},
            original_top8_tracks=event['top8_tracks'],pure_positive_in_top8=bool(set(positive)&set(event['top8_tracks'])),
            threshold_passing_pure_positive_tracks=threshold_positive,primary_accepted_track=primary,final_track=final,
            bank_solver_called=key[1:] in memory_by_key,bank_restored=bank,newborn=newborn,
            final_output_anchor_state=event['final_output_anchor_state'],known_original_wrong=event['known_wrong'],
            candidate_missing_scope='no certified pure positive in actual primary candidates; unknown/impure alternatives are not certified negatives',
            final_ID_note='Bank/newborn are actual native actions; numeric Global ID alone does not certify final GT identity correctness')
        result.append(row);state_counts.update(tags)
        if event['known_wrong']:
            errors.append(dict(**row,frame=event['frame'],camera=event['camera'],full_input_complete=True,all_legal_history_complete=True,
                candidate_space_reachable=bool(positive),top8_reachable=bool(set(positive)&set(event['top8_tracks'])),
                current_native_positive_threshold_passes=bool(threshold_positive),input_gap_reason=None if positive else 'NO_CERTIFIED_PURE_POSITIVE_IN_NATIVE_CANDIDATES',
                source_audit_primary_index=event['source_primary_call']))
    if len(errors)!=26:raise ValueError('frozen26 coverage mismatch')
    write_rows(destination/'queries.jsonl.gz',result)
    known_people=sorted({r['gt_identity'] for r in errors if r['gt_identity'] is not None})
    coverage=dict(status='PASS_ALL_NATIVE_INPUTS_PRESENT',all_development_queries=len(queries),original_error_events=26,
        errors_matched_to_inputs=len(errors),errors_with_full_candidate_and_history_inputs=len(errors),
        errors_with_legal_certified_pure_positive=sum(r['candidate_space_reachable'] for r in errors),
        errors_with_Top8_pure_positive=sum(r['top8_reachable'] for r in errors),
        errors_pure_positive_outside_Top8=sum(r['candidate_space_reachable'] and not r['top8_reachable'] for r in errors),
        errors_no_certified_pure_positive=sum(not r['candidate_space_reachable'] for r in errors),
        errors_with_current_threshold_passing_pure_positive=sum(r['current_native_positive_threshold_passes'] for r in errors),
        wrong_links=sum(r['primary_state']=='WRONG_LINK' for r in errors),missed_known_positive=sum(r['primary_state']=='MISSED_KNOWN_POSITIVE' for r in errors),
        errors_bank_restored=sum(r['bank_restored'] for r in errors),errors_newborn=sum(r['newborn'] for r in errors),
        errors_final_track_differs_from_primary=sum(r['final_track']!=r['primary_accepted_track'] for r in errors),
        contributing_gt_identities=known_people,contributing_identity_count=len(known_people),
        errors_by_camera=dict(collections.Counter(str(r['camera']) for r in errors)),
        errors_by_100_group_interval=dict(collections.Counter(f"{((r['frame']-1)//100)*100+1}-{((r['frame']-1)//100+1)*100}" for r in errors)),
        errors_by_identity=dict(collections.Counter(str(r['gt_identity']) for r in errors)),
        all_query_label_tags=dict(state_counts),error_events=errors,
        labels_origin='All14557 primary-query certifications reused by exact physical key from sealed audit; only initial-camera non-query births newly matched against unchanged original GT after inference.',
        original_query_certification_sha256=sha256(AUDIT/'events.jsonl.gz'),raw_gt_sha256=gt_hashes,GT_quality_policy=diagnostics,
        initial_births_not_primary_queries=len(initial),all_unknown_and_impure_references_preserved=True,algorithm_corrected_errors='NOT_MEASURED')
    save_json(root/'coverage_audit.json',coverage)
    return coverage


def main():
    p=argparse.ArgumentParser();p.add_argument('--release',type=Path,required=True);a=p.parse_args();root=a.release
    manifest=json.loads((root/'manifest.json').read_text())
    if manifest['status']!='CAPTURE_COMPLETE_PARITY_PASS_UNLABELED' or manifest['groups']!=1200:raise ValueError('full unsealed native capture required')
    save_json(root/'capture_manifest.json',manifest)
    save_json(root/'precheck_gate.json',json.loads((BASE/'dev12_native_precheck_v1/gate.json').read_text()))
    save_json(root/'hardware_snapshot_during_capture.json',json.loads((BASE/'dev12_native_precheck_v1/hardware_snapshot_during_capture_v1.json').read_text()))
    audit=json.loads((AUDIT/'manifest.json').read_text())
    for name,digest in audit['artifacts'].items():
        if sha256(AUDIT/name)!=digest:raise ValueError('original audit changed')
    for field in ('checkpoint_sha256','config_sha256','annotation_sha256','seed'):
        if manifest['provenance'][field]!=audit['provenance'][field]:raise ValueError('native source changed')
    consumer=validate(root);save_json(root/'input_consumer_audit.json',consumer)
    queries={tuple(r['key']):r for r in rows(root/'native_queries.jsonl.gz')};calls={};observations={}
    for path in sorted((root/'native_calls').glob('calls_*.pt')):
        calls.update({c['call_id']:c for c in torch.load(path,map_location='cpu')})
    for path in sorted((root/'native_observations').glob('observations_*.pt')):observations.update(torch.load(path,map_location='cpu'))
    coverage=attach_labels(root,queries,calls,observations)
    # Independent final label join needs no original GT files and never changes
    # input construction: prove exact key/row/final-ID alignment of each label.
    seen=set()
    for label in rows(root/'offline_labels/queries.jsonl.gz'):
        key=tuple(label['key']);query=queries[key]
        if key in seen or any(label[k]!=query[k] for k in ('call_id','query_row','final_track')):raise ValueError('offline consumer label join failed')
        seen.add(key)
    if seen!=set(queries):raise ValueError('offline label coverage incomplete')
    save_json(root/'consumer_audit.json',dict(**consumer,offline_label_join_exact=True,offline_label_queries=len(seen),
        inputs_constructed_before_labels=True,original_video_and_GT_files_needed_by_consumer=False))
    baseline_decisions=list(rows(AUDIT/'actual_decisions.jsonl.gz'));actual_decisions=list(rows(root/'actual_decisions.jsonl.gz'))
    if actual_decisions!=baseline_decisions or sha256(root/'predictions.json')!=audit['predictions_sha256']:raise ValueError('full native parity failed')
    save_json(root/'parity_audit.json',dict(status='PASS',scope_sha256=manifest['scope_sha256'],full_groups=1200,full_images=2400,
        prediction_file_bytes_exact=True,every_actual_primary_and_Bank_decision_exact=True,actual_solver_calls=len(actual_decisions),
        full_Hungarian_replay_exact=consumer['every_joint_Hungarian_exact'],full_final_commit_linkage_exact=True,
        all1152_features_native_FP32=True,precheck_gate_sha256=sha256(BASE/'dev12_native_precheck_v1/gate.json'),
        original_audit_manifest=str(AUDIT/'manifest.json'),original_audit_manifest_sha256=sha256(AUDIT/'manifest.json'),
        qualification='Original audit did not save internal state fingerprints; full parity proves every recorded solver/threshold/Bank/final-ID decision and prediction. Export-off/on internal state fingerprints additionally match on the real40 precheck.'))
    ledger=json.loads((BASE/'dev12_native_precheck_v1/resource_ledger.json').read_text())
    if any(r['status']!='PASS' for r in ledger['runs']):raise ValueError('unresolved failed native stage')
    save_json(root/'resource_ledger.json',ledger)
    scope=Path(manifest['scope']);scopevalue=json.loads(scope.read_text())
    save_json(root/'frozen_scope.json',scopevalue)
    save_json(root/'source_provenance.json',dict(native_capture=manifest['provenance'],offline_certification=provenance(),
        frozen_scope_sha256=sha256(scope),original_audit_manifest_sha256=sha256(AUDIT/'manifest.json'),
        independent_consumer_source_sha256=sha256(ROOT/'geo_tools/validate_dev12_native_inputs.py'),
        capture_precision='native torch.float32 copied losslessly; no FP16 equivalence claim',
        immutable_source_worktree=manifest['provenance']['source_worktree']))
    total=sum(f.stat().st_size for f in root.rglob('*') if f.is_file())
    if total>scopevalue['maximum_output_bytes'] or ledger['total_gpu_wall_hours']>4:raise ValueError('independent release budget exceeded')
    manifest.update(status='SEALED_ASSET_READY',engineering_status='DEV12_NATIVE_ASSET_HANDOFF_COMPLETE',
        research_status='NO_VGGT_OR_CV_PIL_SCIENTIFIC_SUCCESS_CLAIM',offline_labels_separate=True,read_only_consumer_passed=True,
        actor_GT_read=False,offline_GT_read=True,GT_label_files_present=True,
        independent_gpu_wall_hours=ledger['total_gpu_wall_hours'],artifact_bytes_before_final_seal=total,
        sealed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        artifacts={str(f.relative_to(root)):sha256(f) for f in root.rglob('*') if f.is_file() and f.name not in ('manifest.json','artifact_sha256.json')})
    manifest.pop('GT_read',None)
    save_json(root/'manifest.json',manifest)
    hashes={str(f.relative_to(root)):sha256(f) for f in root.rglob('*') if f.is_file() and f.name!='artifact_sha256.json'}
    save_json(root/'artifact_sha256.json',dict(interface_version=INTERFACE,status='SEALED_ASSET_READY',artifacts=hashes,
        note='Root artifact_sha256.json is the external trust anchor; its own SHA is published in Git. It hashes manifest.json and every other artifact.'))
    # Reverify the exact sealed bytes before applying read-only permissions.
    for name,digest in hashes.items():
        if sha256(root/name)!=digest:raise ValueError('final sealed artifact changed')
    for path in root.rglob('*'):
        if path.is_file():path.chmod(0o444)
    for path in sorted([p for p in root.rglob('*') if p.is_dir()],reverse=True):path.chmod(0o555)
    root.chmod(0o555)
    print(json.dumps(dict(status='DEV12_NATIVE_ASSET_HANDOFF_COMPLETE',release=str(root),
        artifact_sha256=sha256(root/'artifact_sha256.json'),manifest_sha256=sha256(root/'manifest.json'),queries=len(queries),
        known_errors_complete=coverage['errors_with_full_candidate_and_history_inputs'],reachable=coverage['errors_with_legal_certified_pure_positive'],
        gpu_wall_hours=ledger['total_gpu_wall_hours'],disk_bytes=sum(f.stat().st_size for f in root.rglob('*') if f.is_file()))))


if __name__=='__main__':main()
