"""Join existing sealed labels to existing native references, never re-label GT.

This is a conservative metadata-only coverage bound. Missing initialization
labels remain UNKNOWN. Source-tag purity does not certify actual averaged
native descriptors; final feature-version masks remain a separate admission.
Existing positive certificates are retained even if initialization-source labels
were never exported. No raw GT file, detector, teacher, video or fitted head is read.
"""
import argparse
from collections import Counter,defaultdict
import gzip
import json
from pathlib import Path
import time
from audit_pirga_fulltrain import FIT,checked,read_rows


def audit(source_manifest,runtime,out):
    if out.exists():raise FileExistsError(out)
    out.mkdir();pins=json.loads(source_manifest.read_text())['artifacts'];all_counts=Counter();by_scene=[];by_person=defaultdict(Counter);started=time.monotonic()
    for scene in FIT:
        base=runtime/'full_train_audit_v1'/f'scene{scene}_full_v1'
        ep=base/'events.jsonl.gz';cp=base/'actual_decisions.jsonl.gz';checked(ep,pins[str(ep)]);checked(cp,pins[str(cp)])
        events=list(read_rows(ep));by_key={tuple(e['key'][1:]):e for e in events}
        if len(by_key)!=len(events):raise ValueError('Duplicate existing physical labels')
        count=Counter();seen=set();initial_sources=set();primary_index=0;eligible_people=set();prediction_people=set()
        with gzip.open(out/f'scene{scene}_metadata_only_supervision.jsonl.gz','xt') as f:
            for c in read_rows(cp):
                if c['stage']=='run_memory_tracker':continue
                refs=defaultdict(list)
                for r in c['references']:
                    src=tuple(r['source']);refs[r['track']].append(src)
                    if src not in by_key:initial_sources.add(src)
                ids=sorted(refs);pure={}
                for t,source_keys in refs.items():
                    labels=[by_key[k]['query_gt_identity'] if k in by_key else None for k in source_keys]
                    pure[t]=labels[0] if labels and labels[0] is not None and all(p==labels[0] for p in labels) else None
                for source in c['queries']:
                    key=tuple(source);e=by_key[key]
                    if key in seen or e['source_primary_call']!=primary_index or e['candidate_count']!=len(ids):raise ValueError('Complete call/event alignment differs')
                    seen.add(key);person=e['query_gt_identity'];positive=[t for t in ids if person is not None and pure[t]==person]
                    if not set(positive)<=set(e['positive_tracks']):raise ValueError('Conservative source join contradicts sealed certificate')
                    sealed_positive=list(e['positive_tracks'])
                    if person is None and sealed_positive or not set(sealed_positive)<=set(ids):raise ValueError('Invalid existing positive certificate')
                    # Retain already sealed positive labels even when initial
                    # physical-source query labels were never exported. No new
                    # initialization label is inferred or assigned.
                    for track in sealed_positive:
                        if any(src in by_key and by_key[src]['query_gt_identity']!=person for src in refs[track]):
                            raise ValueError('Existing positive certificate contradicts an exported physical-source label')
                    certificate_people={t:(person if t in sealed_positive else pure[t]) for t in ids}
                    negatives=[t for t in ids if person is not None and pure[t] is not None and pure[t]!=person]
                    past=any(src in by_key and by_key[src]['frame']<e['frame'] for source_keys in refs.values() for src in source_keys)
                    cross=any(src in by_key and by_key[src]['frame']<e['frame'] and by_key[src]['camera']!=e['camera'] for t in sealed_positive for src in refs[t])
                    rank=bool(sealed_positive and negatives and past)
                    # Query current GT is masked if unknown; known source identities
                    # do not make unknown query examples known positives/negatives.
                    count['all_actual_queries']+=1;count['GT_known']+=person is not None
                    count['sealed_raw_source_positive_available']+=bool(e['positive_tracks'])
                    count['source_join_pure_positive_available']+=bool(positive)
                    count['sealed_positive_preserved_despite_missing_initial_source_label']+=bool(sealed_positive) and not positive
                    count['source_join_known_negative_available']+=bool(negatives)
                    count['metadata_ranking_eligible_before_feature_version_masks']+=rank
                    count['metadata_crossview_prediction_eligible_before_feature_version_masks']+=rank and cross
                    count['metadata_native_hard_rank_eligible_before_feature_version_masks']+=rank and e['known_wrong']
                    count['missing_initial_label_reduces_positive_coverage']+=bool(e['positive_tracks']) and not positive
                    count['source_alias_positive_queries']+=len(sealed_positive)>1
                    if person is not None:
                        g=(scene,person);by_person[g]['queries']+=1;by_person[g]['metadata_rank_eligible']+=rank;by_person[g]['metadata_crossview_eligible']+=rank and cross
                        if rank:eligible_people.add(g)
                        if rank and cross:prediction_people.add(g)
                    row=dict(event_key=e['key'],GT_person_existing=e['query_gt_identity'],candidate_ids_from_native_call=ids,
                             raw_source_candidate_persons=[pure[t] for t in ids],sealed_positive_tracks=e['positive_tracks'],
                             candidate_persons_from_existing_certificates=[certificate_people[t] for t in ids],
                             actual_primary_state_existing=e['actual_primary_state'],metadata_rank_eligible=rank,
                             metadata_crossview_eligible=rank and cross,actual_descriptor_version_certified=False,TRAINING_ADMITTED=False)
                    f.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n')
                primary_index+=1
        if seen!=set(by_key):raise ValueError('Full native query coverage incomplete')
        count['unlabelled_initial_physical_sources']=len(initial_sources)
        by_scene.append(dict(scene=scene,counts=dict(count),metadata_rank_person_groups=len(eligible_people),metadata_crossview_person_groups=len(prediction_people)))
        all_counts.update(count);print(json.dumps(dict(scene=scene,counts=dict(count))),flush=True)
    report=dict(status='PASS_EXISTING_LABEL_JOIN_METADATA_ONLY_NOT_TRAINING_ADMISSION',fit_scenes=list(FIT),counts=dict(all_counts),per_scene=by_scene,
                rank_person_groups=sum(v['metadata_rank_person_groups'] for v in by_scene),crossview_person_groups=sum(v['metadata_crossview_person_groups'] for v in by_scene),
                actual_fully_input_eligible_training_queries=None,actual_trained_queries=0,
                unknown_initial_labels_imputed=False,raw_GT_label_source_recomputed=False,new_GT_matching=False,
                descriptor_version_masks_pending=True,supervision_lower_bound_claim=False,
                limitation='These are source-metadata opportunities. Missing initialization-source labels stay UNKNOWN; existing sealed candidate positives are preserved without imputing those labels. Genuine Bank descriptor purity still needs feature-version admission. Not a claim that all counted rows are trainable.',
                GPU_wall_hours=0,wall_seconds=time.monotonic()-started)
    (out/'METADATA_SUPERVISION_COVERAGE.json').write_text(json.dumps(report,indent=2)+'\n')
    import csv
    with open(out/'METADATA_SUPERVISION_PER_PERSON.csv','w') as f:
        w=csv.writer(f,lineterminator='\n');w.writerow(['scene','person_offline_only','queries','metadata_rank_eligible','metadata_crossview_eligible','actual_trained'])
        for (scene,person),v in sorted(by_person.items()):w.writerow([scene,person,v['queries'],v['metadata_rank_eligible'],v['metadata_crossview_eligible'],0])
    print(json.dumps(report['counts']),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-manifest',type=Path,required=True);p.add_argument('--runtime',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();audit(a.source_manifest,a.runtime,a.out)
