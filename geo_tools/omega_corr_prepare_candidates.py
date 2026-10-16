"""Read sealed native queries/candidates; inference inputs and offline labels remain separate."""
import collections
import json
from omega_corr_common import ROOT,RUNTIME,DATA,jsonl,sha256,save_new,verified_native,role


def main():
    cohort_path=ROOT/'manifests/omega_corr_cohort_v1.json';cohort=json.loads(cohort_path.read_text())
    metadata={i['id']:i for i in json.loads((DATA/'annotations/train.json').read_text())['images']}
    pair_by_group={(p['scene'],p['group']):p for p in cohort['pairs']}
    wanted={i['image_id'] for p in cohort['pairs'] for i in p['images']}
    inputs=[];labels=[];proposals={};receipts=[];counts=collections.Counter()
    for scene in sorted({p['scene'] for p in cohort['pairs']}):
        directory,receipt=verified_native(scene);receipts.append(receipt)
        events={tuple(e['key'][1:]):e for e in jsonl(directory/'events.jsonl.gz')}
        # Full native track rank is original evidence; bind its hash now and check
        # its first16 against every sealed event, never substitute a new ranking.
        rank_path=directory/'rankings.jsonl.gz';rank_iter=iter(jsonl(rank_path))
        receipt['rankings_sha256']=sha256(rank_path)
        for row in jsonl(directory/'detector_provenance.jsonl.gz'):
            source=row['source']
            if source[0] not in wanted:continue
            i=metadata[source[0]];x,y,w,h=row['detection']['bbox']
            key=f'{source[0]}:{source[1]}'
            proposals[key]=dict(source=source,scene=scene,camera=i['view_id'],timestamp=i['frame_id'],
                bbox_xyxy=[x,y,x+w,y+h],bbox_xywh=[x,y,w,h],image_id=source[0])
        for call in jsonl(directory/'actual_decisions.jsonl.gz'):
            if call['stage']=='run_memory_tracker':continue
            ranking=next(rank_iter);assert ranking['queries']==call['queries']
            for qi,q in enumerate(call['queries']):
                e=events[tuple(q)]
                assert ranking['ranked_tracks'][qi][:16]==e['top16_tracks']
                meta=metadata[q[0]];pair=pair_by_group.get((scene,meta['frame_id']))
                if pair is None:continue
                counts['all_selected_primary_queries']+=1
                sync=[]
                for r in call['references']:
                    rm=metadata[r['source'][0]]
                    assert rm['video_id']==scene and rm['frame_id']<=meta['frame_id']
                    if rm['view_id']!=meta['view_id'] and rm['frame_id']==meta['frame_id']:sync.append(r)
                input_row=dict(pair_id=pair['pair_id'],query=q,scene=scene,role=role(scene),timestamp=meta['frame_id'],camera=meta['view_id'],
                    legal_synchronous_references=sync,full_legal_references=call['references'],
                    original_native_ranked_tracks=ranking['ranked_tracks'][qi],original_primary_accepted_track=call['accepted_track_ids'][qi],
                    original_final_track=call['final_track_ids'][qi],native_relative_margin=ranking['relative_margin'][qi],
                    source_primary_call=e['source_primary_call'])
                positives=set(e['positive_tracks']);syncpos=sorted({r['track'] for r in sync}&positives)
                categories=[]
                if e['known_wrong'] and syncpos:categories.append('A_REAL_ERROR_WITH_SYNC_PURE_POSITIVE')
                if e['actual_primary_state']=='correct_link' and e['relative_margin']<=.1:categories.append('B_NATIVE_CORRECT_LOW_MARGIN')
                if e['actual_primary_state']=='correct_link' and e['relative_margin']>=.5:categories.append('D_NATIVE_CORRECT_EASY')
                if not sync or not syncpos:categories.append('E_NO_SYNC_CERTIFIED_POSITIVE_OR_UNKNOWN')
                inputs.append(input_row)
                labels.append(dict(pair_id=pair['pair_id'],query=q,scene=scene,role=role(scene),categories=categories,
                    query_gt_identity=e['query_gt_identity'],positive_legal_tracks=e['positive_tracks'],
                    observed_positive_including_impure=e['observed_positive_including_impure'],sync_pure_positive_tracks=syncpos,
                    native_ranking_state=e['ranking_state'],native_primary_state=e['actual_primary_state'],known_wrong=e['known_wrong']))
                counts['selected_known_primary_errors']+=bool(e['known_wrong'])
                counts['selected_errors_with_sync_pure_positive']+=bool(e['known_wrong'] and syncpos)
                counts['selected_no_sync_references']+=not bool(sync)
                counts['selected_no_sync_pure_positive']+=not bool(syncpos)
        try:next(rank_iter);raise ValueError('unmatched native rank stream')
        except StopIteration:pass
    # All candidate physical observations must exist, including initial seed refs.
    for q in inputs:
        assert ':'.join(map(str,q['query'])) in proposals
        for r in q['legal_synchronous_references']:assert ':'.join(map(str,r['source'])) in proposals
    path=RUNTIME/'native_candidates_v1.json'
    save_new(path,dict(schema='omega-corr-native-candidates/v1',cohort_sha256=sha256(cohort_path),
        proposals=proposals,queries=inputs,receipts=receipts,GT_labels_input=False,no_GMT_inference=True))
    save_new(ROOT/'results/omega_corr_candidate_index_v1.json',dict(status='NATIVE_CANDIDATES_READY',input=dict(path=str(path),sha256=sha256(path)),
        counts=dict(counts),proposals=len(proposals),pairs=len(cohort['pairs']),no_new_GT_matching=True,no_GMT_inference=True,
        C_view_change='Not yet independently camera-angle annotated; different image scale/aspect proxy will be disclosed separately, never asserted as calibrated view angle.'))
    save_new(RUNTIME/'native_candidate_offline_labels_v1.json',dict(schema='omega-corr-offline-labels/v1',rows=labels,
        never_model_input=True,existing_native_labels_only=True))
    print(json.dumps(dict(counts=counts,proposals=len(proposals))),flush=True)

if __name__=='__main__':main()
