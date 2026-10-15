"""Bind locked V3 indices to verified native inputs and bounded cycle targets."""
import argparse
import collections
import hashlib
import torch
from common import *

NEW=Path('/data1/liuyeqiang/WWW_trackvggt_runtime')


def cycle_round_robin(events,limit):
    pools=collections.defaultdict(list)
    for e in events:
        if e.get('cycle') is not None:pools[e['key'][0]].append(e)
    queues={s:collections.deque(sorted(v,key=lambda e:hashlib.sha256(json.dumps(e['key']).encode()).hexdigest())) for s,v in pools.items()}
    selected=[]
    while len(selected)<limit and any(queues.values()):
        for scene in sorted(queues):
            if queues[scene]:selected.append(queues[scene].popleft())
            if len(selected)>=limit:break
    return selected


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',default='hard_events_expanded_v3');a=p.parse_args();out=NEW/'cohorts'/a.tag
    if out.exists():raise FileExistsError(out)
    index_path=ROOT/'manifests/trackvggt_expanded_training_v3.json';index=json.loads(index_path.read_text())
    original_path=NEW/'cohorts/hard_events_expanded_v2/cohort.json';old=json.loads(original_path.read_text())
    if sha256(original_path)!=index['base_cohort_sha256']:raise ValueError('original V2 changed')
    if old['fit']!=index['preserved_V2_fit_rows'] or old['validation']!=index['validation_rows']:raise ValueError('locked membership drift')
    artifacts=dict(old['source_artifacts']);artifacts[str(original_path)]=sha256(original_path);artifacts[str(index_path)]=sha256(index_path)
    payloads={};lookup={};native_manifests=[];payload_paths=[]
    for scene in index['statistics']['scenes']:
        folder=NEW/'selected_native_v3'/f'scene{scene}_full_cycle_v1';mp=folder/'manifest.json';m=json.loads(mp.read_text())
        if m['status']!='COMPLETE_LOCKED_V3_SELECTED_NATIVE_FEATURES' or not m['full_prediction_file_bytes_exact'] or not m['selected_actual_decisions_exact']:raise ValueError('native feature/history gate incomplete')
        if not m['gt_labels_after_prediction_and_solver_parity'] or m['future_history'] or m['GT_query_features']:raise ValueError('invalid causal feature scope')
        if m['V3_index_sha256']!=sha256(index_path):raise ValueError('V3 index changed during capture')
        artifacts[str(mp)]=sha256(mp);native_manifests.append(str(mp))
        for name,digest in m['artifacts'].items():
            if sha256(folder/name)!=digest:raise ValueError('native source changed')
            artifacts[str(folder/name)]=digest
        path=str(folder/'events.pt');payloads[path]=torch.load(path,map_location='cpu');payload_paths.append(path)
        for i,event in enumerate(payloads[path]['events']):
            key=tuple(event['key'])
            if key in lookup:raise ValueError('duplicate physical event')
            lookup[key]=(path,i,event)
    wanted={tuple(r['key']) for r in old['fit']+index['additional_fit']}
    if set(lookup)!=wanted:raise ValueError('native capture not exactly V3 fit plus declared V2 cycle controls')
    fit=list(old['fit'])
    for row in index['additional_fit']:
        path,i,event=lookup[tuple(row['key'])]
        if event['top8_positive']!=row['positive_in_top8'] or event['known_unmatched_within_all_recorded_history']!=row['genuine_NULL_within_full_legal_history'] or event['loss_eligible']!=row['supervised_assignment_eligible']:raise ValueError('supervision mask drift')
        fit.append(dict(key=event['key'],scene=row['scene'],scope='fit',event_file=path,event_index=i,
            native_top1_state=event['native_top1_state'],actual_primary_state=event['actual_primary_state'],relative_margin=event['relative_margin'],
            reachable_known_error=event['top8_positive'] and (event['native_top1_state']=='wrong_link' or event['actual_primary_state'] in ('wrong_link','missed_known_positive'))))
    if len(fit)!=3000 or len({tuple(r['key']) for r in fit})!=3000:raise ValueError('V3 unique budget changed')
    old_keys={tuple(r['key']) for r in old['fit']};all_events=[v[2] for v in lookup.values()]
    selected_v2=cycle_round_robin([e for e in all_events if tuple(e['key']) in old_keys],128)
    selected_v3=cycle_round_robin(all_events,128)
    union={tuple(e['key']):e for e in selected_v2+selected_v3};cycle_rows=[];auxiliary={}
    main_keys={tuple(r['key']) for r in fit+old['validation']}
    for key,event in sorted(union.items()):
        path,i,_=lookup[key];payload=payloads[path];graph=event['cycle'];nodes=graph['B_current'][:4]
        target=payload['cycle_targets'][key];b_indices=list(range(len(nodes)))
        visible=[identity is not None and identity==target['current_A_GT'] and any(target['legal_AB_pairs'][ai][bi] and target['B_current_GT'][bi]==identity for bi in b_indices) for ai,identity in enumerate(target['A_past_GT'])]
        cycle_rows.append(dict(key=list(key),event_file=path,event_index=i,B_indices=b_indices,visible_rows_after_fixed_B_subset=visible,
            in_V2_cycle_subset=key in {tuple(e['key']) for e in selected_v2},in_V3_cycle_subset=key in {tuple(e['key']) for e in selected_v3}))
        for node in nodes:
            physical=(key[0],*node['source'])
            if physical in main_keys:continue # same physical target is already declared
            if physical not in auxiliary:auxiliary[physical]=dict(key=list(physical),scene=key[0],source=node['source'],event_file=path,
                observation_only_cycle_B_target=True,parents=[])
            auxiliary[physical]['parents'].append(list(key))
    feature_paths={b:[str(NEW/'tokens'/f'{b}_scene{s}_full_v2'/'manifest.json') for s in [4,5,6,7,8,12]]+
        [str(NEW/'tokens'/f'{b}_scene{s}_new_v3'/'manifest.json') for s in index['statistics']['scenes']] for b in ['vggt','dinov2']}
    out.mkdir(parents=True)
    cohort=dict(status='LOCKED_SCENE_DISJOINT_REAL_EVENT_COHORT',tag=a.tag,cohort_version=3,fit=fit,validation=old['validation'],native_feature_dim=1152,
        fit_scene_ids=index['statistics']['scenes'],validation_scene_ids=[12],source_artifacts=artifacts,feature_manifest_paths=feature_paths,
        auxiliary_target_index=list(auxiliary.values()),cycle_payloads=payload_paths,cycle_rows=cycle_rows,
        cycle_subset_rule='Observation-only existence of actual A(t-1), earlier-camera B(t), selected A(t); stable-key scene rotation, at most128 per cohort. First at most4 B nodes in native At Top8 order. GT never selects indices; post-parity masks determine visible rows.',
        cycle_subset_counts=dict(V2=len(selected_v2),V3=len(selected_v3),union=len(cycle_rows),missing_auxiliary_B_targets=len(auxiliary)),
        primary_new_target_keys=[r['key'] for r in index['additional_fit']],reused_V2_target_keys=[r['key'] for r in old['fit']+old['validation']],
        feature_extraction_status_at_cohort_lock='NOT_YET_RUN',validation300_unchanged=True,official_test_used=False,used_teacher_results_for_selection=False,
        source_native_manifests=native_manifests,V3_locked_index=str(index_path),V3_locked_index_sha256=sha256(index_path),provenance=provenance())
    save_json(out/'cohort.json',cohort)
    save_json(ROOT/'results/trackvggt_v3_native_cohort_v1.json',dict(status='COMPLETE_NATIVE_V3_COHORT_TARGET_EXTRACTION_PENDING',cohort=str(out/'cohort.json'),cohort_sha256=sha256(out/'cohort.json'),
        fit_queries=len(fit),validation_queries=300,new_queries=2000,reused_fit_queries=1000,cycle_subset_counts=cohort['cycle_subset_counts'],
        visible_cycle_queries_V2=sum(any(r['visible_rows_after_fixed_B_subset']) for r in cycle_rows if r['in_V2_cycle_subset']),
        visible_cycle_queries_V3=sum(any(r['visible_rows_after_fixed_B_subset']) for r in cycle_rows if r['in_V3_cycle_subset']),
        target_extraction='NOT_RUN',V2_unchanged=True,source_artifacts=artifacts))
    print(json.dumps(dict(fit=3000,validation=300,cycles=cohort['cycle_subset_counts'],cohort=str(out/'cohort.json'))))


if __name__=='__main__':main()
