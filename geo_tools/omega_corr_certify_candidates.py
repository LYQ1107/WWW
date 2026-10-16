"""Separate offline track-purity labels from production native candidate inputs."""
import collections
import json
from omega_corr_common import ROOT,RUNTIME,NATIVE,jsonl,sha256,save_new


def main():
    index=json.loads((ROOT/'results/omega_corr_candidate_index_v1.json').read_text())
    assert sha256(index['input']['path'])==index['input']['sha256']
    inputs=json.loads(open(index['input']['path']).read())
    offline=json.loads((RUNTIME/'native_candidate_offline_labels_v1.json').read_text())
    by_query={tuple(q['query']):q for q in offline['rows']};sources={}
    for scene in sorted({q['scene'] for q in inputs['queries']}):
        for e in jsonl(NATIVE/f'scene{scene}_full_v1/events.jsonl.gz'):sources[tuple(e['key'][1:])]=e['query_gt_identity']
    recovered=0
    for q in inputs['queries']:
        truth=by_query[tuple(q['query'])]
        if truth['query_gt_identity'] is None:continue
        for ref in q['full_legal_references']:
            if ref['track'] not in truth['positive_legal_tracks']:continue
            key=tuple(ref['source']);value=truth['query_gt_identity']
            if key in sources:assert sources[key]==value
            else:sources[key]=value;recovered+=1
    rows=[];counts=collections.Counter()
    for q in inputs['queries']:
        truth=by_query[tuple(q['query'])];identities=collections.defaultdict(set)
        for r in q['full_legal_references']:identities[r['track']].add(sources.get(tuple(r['source'])))
        statuses={}
        for track,values in identities.items():
            if truth['query_gt_identity'] is None:status='UNKNOWN_QUERY'
            elif None in values:status='UNKNOWN_REFERENCE_HISTORY'
            elif len(values)>1:status='MIXED_REFERENCE_HISTORY'
            else:status='CORRECT' if next(iter(values))==truth['query_gt_identity'] else 'WRONG'
            if status=='CORRECT':assert track in truth['positive_legal_tracks']
            statuses[str(track)]=status
        native=q['original_native_ranked_tracks'][0] if q['original_native_ranked_tracks'] else None
        rows.append(dict(**truth,track_status=statuses,native_global_top1=native,
            native_global_top1_status=statuses.get(str(native),'NO_CANDIDATE'),
            reference_source_gt={':'.join(map(str,r['source'])):sources.get(tuple(r['source'])) for r in q['legal_synchronous_references']}))
        counts['query_gt_known']+=truth['query_gt_identity'] is not None
        counts['queries_with_sync_pure_positive']+=bool(truth['sync_pure_positive_tracks'])
        counts['native_global_known_wrong_rank']+=statuses.get(str(native))=='WRONG'
    path=RUNTIME/'candidate_certifications_v1.json';save_new(path,dict(rows=rows,offline_only=True,no_new_GT_matching=True,
        source_query_labels_from_sealed_events=True,initial_source_recovered_only_by_existing_pure_positive_certificates=recovered))
    save_new(ROOT/'results/omega_corr_candidate_certification_v1.json',dict(status='OFFLINE_CERTIFICATION_READY',counts=dict(counts),
        input=dict(path=str(path),sha256=sha256(path)),never_encoder_input=True,no_new_GT_matching=True,
        initial_source_recoveries=recovered,missing_or_mixed_retained_unknown=True))
    print(dict(counts),flush=True)

if __name__=='__main__':main()
