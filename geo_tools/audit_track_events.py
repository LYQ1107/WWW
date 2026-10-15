"""Known errors are distinct from impure/unknown reference histories."""
import collections
import torch
from common import *
from collect_track_hard_events import NEW,attach_labels,pure_reference_identity
from gtr.geometry.roi_feature_extractor import PadMapping
from gtr.geometry.identity_guided_sampler import patch_regions


def main():
    data=train_data();reports=[];inputs={}
    for tag in ('fit5_prefix400_v1','fit8_prefix400_v1','validation12_prefix400_v1'):
        folder=NEW/'events'/tag;path=folder/'manifest.json';manifest=json.loads(path.read_text())
        assert manifest['status']=='PASS_READONLY_EVENTS_POST_PARITY_LABELS' and manifest['prefix40_raw_prediction_exact']
        for file,digest in manifest['artifacts'].items():assert sha256(folder/file)==digest;inputs[str(folder/file)]=digest
        inputs[str(path)]=sha256(path)
        payload=torch.load(folder/'events.pt',map_location='cpu')
        images=video_records(data,manifest['video_id'],manifest['max_frames']);detectors={}
        import gzip
        with gzip.open(folder/'detector_provenance.jsonl.gz','rt') as stream:
            for line in stream:
                row=json.loads(line);detectors[tuple(row['source'])]=row
        labels,hashes=attach_labels(data,manifest['video_id'],images,detectors)
        assert hashes==manifest['raw_gt_sha256']
        for file,digest in hashes.items():assert sha256(file)==digest;inputs[file]=digest
        counts=collections.Counter();bins=collections.Counter();rows=[]
        for event in payload['events']:
            call=payload['calls'][event['call_index']];key=tuple(event['query_source']);query=payload['observations'][key]
            reference=collections.defaultdict(list)
            for ref in call['references']:reference[ref['track']].append(labels[tuple(ref['source'])])
            candidate_truth={track:pure_reference_identity(value) for track,value in reference.items()}
            assert labels[key]==event['query_gt_identity']
            assert event['positive_tracks']==[track for track in event['candidate_tracks'] if labels[key] is not None and candidate_truth[track]==labels[key]]
            first=candidate_truth[event['candidate_tracks'][0]]
            state='unknown_query' if labels[key] is None else 'ambiguous_reference' if first is None else 'correct_link' if first==labels[key] else 'wrong_link'
            counts['selected_events']+=1;counts['native_top1_'+state]+=1
            counts['reachable_known_top1_errors']+=state=='wrong_link' and event['top8_positive']
            counts['actual_primary_'+event['actual_primary_state']]+=1
            counts['loss_eligible']+=event['loss_eligible'];counts['top8_has_positive']+=event['top8_positive']
            _,inside,_,_=patch_regions(PadMapping(query['width'],query['height']),query['box'])
            count=int(inside.sum());bucket='0' if count==0 else '1-3' if count<4 else '4-15' if count<16 else '16+'
            bins[bucket]+=1
            rows.append(dict(key=event['key'],native_top1_state=state,actual_primary_state=event['actual_primary_state'],top8_has_known_positive=event['top8_positive'],valid_patch_centers=count,patch_bin=bucket))
        reports.append(dict(tag=tag,video_id=manifest['video_id'],scope=manifest['scope'],feature_dim=payload['feature_dim'],
            selected_counts=dict(counts),patch_size_bins=dict(bins),all_collected_coverage=manifest['all_candidate_coverage'],
            all_collected_solver_counts=manifest['solver_decision_counts'],selected_rows=rows))
    assert sum(row['selected_counts']['selected_events'] for row in reports if row['scope']=='fit')==1000
    assert {row['video_id'] for row in reports if row['scope']=='fit'}=={5,8} and {row['video_id'] for row in reports if row['scope']=='validation'}=={12}
    save_json(ROOT/'results/trackvggt_hard_events_audit_v1.json',dict(status='VERIFIED_REAL_EVENT_COHORT_WITH_UNKNOWNS',rows=reports,source_artifacts=inputs,
        warning='Legacy v1 collector field top8_correctable_native_top1_errors counts not-known-correct predictions including ambiguous references; it is NOT a known-error count. This audit gives authoritative known/ambiguous breakdown without rewriting raw collection.',
        scope='three actual400-group causal GMT collectors, observed native histories, GT post-only after fresh OFF40 exact parity; fit selection used labels for not-known-correct priority, validation selection uses only native margins; same-scene observations are correlated',provenance=provenance()))
    lines=['# Real TrackVGGT hard-event cohort','',
        'Fresh frozen GMT history, actual detector/solver provenance and native logits; no old teacher vectors, GT-built histories or future observations. Fit scenes5/8; held-out TRAIN12. Each run has400 synchronized groups and a second real40-group OFF equality gate before any GT CSV labels.','',
        'The first raw collector called all not-known-correct Top1 predictions errors, including impure reference histories. That label is too broad. Raw files are retained; the table below is authoritative and splits known wrong, ambiguous and unknown cases. Actual Hungarian misses are separate from ranking errors. Fit sampling prioritized reachable not-known-correct queries then native margins; held-out sampling used native margins only and is tested not to require GT fields.','',
        '| Scene/scope | Selected | Top1 known correct | Top1 known wrong | Top1 ambiguous | Unknown query | Selected actual primary known error/miss | Patch bins0 /1-3 /4-15 /16+ |','|---|---:|---:|---:|---:|---:|---:|---|']
    for row in reports:
        c=row['selected_counts'];b=row['patch_size_bins']
        lines.append(f"| {row['video_id']}/{row['scope']} | {c['selected_events']} | {c.get('native_top1_correct_link',0)} | {c.get('native_top1_wrong_link',0)} | {c.get('native_top1_ambiguous_reference',0)} | {c.get('native_top1_unknown_query',0)} | {c.get('actual_primary_wrong_link',0)}/{c.get('actual_primary_missed_known_positive',0)} | {' / '.join(str(b.get(k,0)) for k in ['0','1-3','4-15','16+'])} |")
    lines+=['','Each Top8 coverage denominator and all-collection solver states remain in the machine audit. Five known-positive queries in held-out TRAIN12 lose all pure positive tracks at Top8 despite a correct observation still existing in an impure track; never relabel those pruned cases as true unmatched. Unmatched/ambiguous proposal labels are masked for supervised positive assignment; they remain reported in evaluation coverage. No teacher efficacy follows from event counts.','']
    (ROOT/'docs/TRACKVGGT_HARD_EVENTS.md').write_text('\n'.join(lines))
    print(json.dumps([dict(scene=row['video_id'],counts=row['selected_counts'],patch_bins=row['patch_size_bins']) for row in reports]))


if __name__=='__main__':main()
