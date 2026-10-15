"""Bounded scene-disjoint cohort fixed before any teacher result."""
import argparse,collections
import torch
from common import *

NEW=Path('/data1/liuyeqiang/WWW_trackvggt_runtime')
TAGS=['fit4_prefix400_known_v2','fit5_prefix400_v1','fit6_prefix400_known_v2','fit7_prefix400_known_v2','fit8_prefix400_v1','validation12_prefix400_v1']


def scene_round_robin(pools,limit,used):
    result=[];queues={scene:collections.deque(rows) for scene,rows in pools.items()}
    while len(result)<limit and any(queues.values()):
        for scene in sorted(queues):
            while queues[scene] and tuple(queues[scene][0]['key']) in used:queues[scene].popleft()
            if queues[scene] and len(result)<limit:
                row=queues[scene].popleft();used.add(tuple(row['key']));result.append(row)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',default='hard_events_expanded_v2');a=p.parse_args()
    out=NEW/'cohorts'/a.tag
    if out.exists():raise FileExistsError(out)
    initial=json.loads((ROOT/'results/trackvggt_hard_events_audit_v1.json').read_text())
    states={tuple(row['key']):row['native_top1_state'] for group in initial['rows'] for row in group['selected_rows']}
    pools=collections.defaultdict(list);validation=[];sources={};summaries=[]
    for tag in TAGS:
        folder=NEW/'events'/tag;path=folder/'manifest.json';manifest=json.loads(path.read_text())
        assert manifest['status']=='PASS_READONLY_EVENTS_POST_PARITY_LABELS' and manifest['prefix40_raw_prediction_exact'] and not manifest['gt_query_features'] and not manifest['future_query_features']
        sources[str(path)]=sha256(path)
        for file,digest in manifest['artifacts'].items():assert sha256(folder/file)==digest;sources[str(folder/file)]=digest
        data=torch.load(folder/'events.pt',map_location='cpu')
        if data['feature_dim']!=1152:raise ValueError('observed native dimension changed; revise model construction explicitly')
        counts=collections.Counter()
        for index,event in enumerate(data['events']):
            state=event.get('native_top1_state',states.get(tuple(event['key'])))
            if state is None:raise ValueError('missing known/ambiguous ranking state')
            reachable=event['top8_positive'] and (state=='wrong_link' or event['actual_primary_state'] in ('wrong_link','missed_known_positive'))
            row=dict(key=event['key'],scene=manifest['video_id'],event_file=str(folder/'events.pt'),event_index=index,
                native_top1_state=state,actual_primary_state=event['actual_primary_state'],reachable_known_error=reachable,
                relative_margin=event['relative_margin'],scope=manifest['scope'])
            if manifest['scope']=='fit':pools[row['scene']].append(row)
            else:validation.append(row)
            counts[state]+=1;counts['reachable_known_error']+=reachable
        summaries.append(dict(tag=tag,scene=manifest['video_id'],scope=manifest['scope'],selected_states=dict(counts)))
    if set(pools)!={4,5,6,7,8} or {row['scene'] for row in validation}!={12} or len(validation)!=300:raise ValueError('scene-disjoint predefined cohort incomplete')
    used=set();hard={scene:sorted([row for row in rows if row['reachable_known_error']],key=lambda row:row['key']) for scene,rows in pools.items()}
    fit=scene_round_robin(hard,500,used)
    remaining={scene:sorted(rows,key=lambda row:(row['relative_margin'],row['key'])) for scene,rows in pools.items()}
    fit+=scene_round_robin(remaining,1000-len(fit),used)
    if len(fit)!=1000 or len(used)!=1000:raise ValueError('unique fit budget incomplete')
    if set(tuple(row['key']) for row in validation)&used:raise ValueError('physical query leakage')
    out.mkdir(parents=True)
    value=dict(status='LOCKED_SCENE_DISJOINT_REAL_EVENT_COHORT',tag=a.tag,fit=fit,validation=validation,
        native_feature_dim=1152,fit_scene_ids=sorted(pools),validation_scene_ids=[12],summaries=summaries,
        fit_reachable_known_errors=sum(row['reachable_known_error'] for row in fit),
        fit_known_ranking_errors=sum(row['native_top1_state']=='wrong_link' for row in fit),
        selected_per_fit_scene=dict(collections.Counter(row['scene'] for row in fit)),source_artifacts=sources,
        selection='fixed TRAIN known-error first scene-round-robin capped500, then native-margin scene-round-robin to1000; held-outTRAIN12 initial300 unchanged',
        used_teacher_results_for_selection=False,official_test_used=False,provenance=provenance())
    save_json(out/'cohort.json',value);save_json(ROOT/'results/trackvggt_expanded_cohort_v2.json',value)
    print(json.dumps(dict(status=value['status'],fit=len(fit),validation=len(validation),fit_reachable_known_errors=value['fit_reachable_known_errors'],fit_known_ranking_errors=value['fit_known_ranking_errors'],selected_per_fit_scene=value['selected_per_fit_scene'])))


if __name__=='__main__':main()
