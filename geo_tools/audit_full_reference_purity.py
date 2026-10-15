"""Add missing mixed-vs-unknown/cross-camera fields using saved traces only."""
import collections
from common import *
from audit_decision_trace import records, decision_state
from audit_full_train_identity_events import original_truth, label_detectors
from collect_track_hard_events import pure_reference_identity


def main():
    root=Path('/data1/liuyeqiang/WWW_trackvggt_runtime/full_train_audit_v1');data=train_data();rows=[];artifacts={}
    for scene in [4,5,6,7,8,9,12,13,17,20,21,24]:
        folder=root/f'scene{scene}_full_v1';mp=folder/'manifest.json'
        if not mp.exists() or json.loads(mp.read_text())['status']!='COMPLETE_FULL_CAUSAL_AUDIT':raise ValueError('required full scan incomplete')
        m=json.loads(mp.read_text());artifacts[str(mp)]=sha256(mp)
        for name in ['detector_provenance.jsonl.gz','actual_decisions.jsonl.gz']:
            if sha256(folder/name)!=m['artifacts'][name]:raise ValueError('saved trace changed')
            artifacts[str(folder/name)]=sha256(folder/name)
        images=video_records(data,scene);meta={i['image_id']:i for i in images}
        truth,blocked,hashes,_=original_truth(data,scene,images);artifacts.update(hashes)
        labels,_,_,_=label_detectors(folder,truth,blocked);counts=collections.Counter();selected_states=collections.Counter();examples=[]
        for decision in records(folder/'actual_decisions.jsonl.gz'):
            if decision['stage']=='run_memory_tracker':continue
            refs=collections.defaultdict(list);cameras=collections.defaultdict(set)
            for ref in decision['references']:
                key=tuple(ref['source']);refs[ref['track']].append(labels[key]);cameras[ref['track']].add(meta[key[0]]['view_id'])
            known={t:{v for v in values if v is not None} for t,values in refs.items()}
            pure={t:pure_reference_identity(v) for t,v in refs.items()}
            mixed={t for t,v in known.items() if len(v)>1};unknown={t for t,v in refs.items() if any(a is None for a in v)}
            unknown_only={t for t,v in known.items() if not v};single_with_unknown={t for t,v in known.items() if len(v)==1 and t in unknown}
            for source,selected in zip(decision['queries'],decision['accepted_track_ids']):
                key=tuple(source);identity=labels[key];camera=meta[key[0]]['view_id'];positive=[t for t,v in pure.items() if identity is not None and v==identity]
                state=decision_state(identity,refs,selected)
                counts['queries']+=1;counts['query_with_known_multi_ID_reference_track']+=bool(mixed)
                counts['query_with_any_unmatched_or_unreliable_reference_observation']+=bool(unknown)
                counts['query_with_all_unknown_reference_track']+=bool(unknown_only);counts['query_with_single_known_ID_and_unknown_reference_track']+=bool(single_with_unknown)
                if selected>=0:
                    subtype='known_mixed_and_unknown' if selected in mixed and selected in unknown else 'known_mixed_only' if selected in mixed else 'all_unknown' if selected in unknown_only else 'single_known_ID_plus_unknown' if selected in single_with_unknown else 'pure_known'
                    selected_states[subtype]+=1
                if identity is not None and not positive:
                    observed=[t for t,v in known.items() if identity in v]
                    counts['missing_pure_positive_correct_observation_only_in_impure_tracks']+=bool(observed)
                    counts['missing_pure_positive_no_matching_observation_in_current_legal_context']+=not observed
                    if observed and len(examples)<20:examples.append(dict(key=[scene,*source],impure_tracks_with_correct_observation=observed))
                cross=any(any(c!=camera for c in cameras[t]) for t in positive);same=any(camera in cameras[t] for t in positive)
                if state in ('wrong_link','missed_known_positive') and positive:
                    counts['correctable_with_any_other_camera_pure_positive']+=cross
                    counts['correctable_only_other_camera_pure_positive']+=cross and not same
                    counts['correctable_with_same_camera_pure_positive']+=same
                if state=='wrong_link':
                    counts['known_wrong_link_selected_other_camera_only']+=camera not in cameras[selected]
        rows.append(dict(scene=scene,counts=dict(counts),selected_reference_purity_states=dict(selected_states),examples=examples))
        print(scene,json.dumps(dict(counts)),flush=True)
    save_json(ROOT/'results/trackvggt_reference_purity_supplement_v1.json',dict(status='COMPLETE_ALL12_SAVED_TRACE_SUPPLEMENT',rows=rows,source_artifacts=artifacts,
        scope='New missing fields from existing real inference traces; no GMT/VGGT replay or overwrite of sealed outputs. Counts are query exposures, not independent reference tracks.',
        interpretation='Known multi-ID history is distinguished from unmatched/GT-unreliable reference observations. Both are impure for supervision. Correct observation only in impure history is not a pure candidate. Missing without matching observation has unresolved causes; do not infer detector failure/new target/lost track from absence alone.',
        overlap='Known mixed and unknown reference flags can coexist; cross-camera-positive and same-camera-positive flags overlap when a pure track spans cameras.',provenance=provenance()))


if __name__=='__main__':main()
