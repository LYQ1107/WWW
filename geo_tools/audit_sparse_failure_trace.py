"""Read-only analysis of real dense/cache traces; never select update times."""
import argparse
import collections
from common import *


def first_partition_change(rows):
    """Find a split/merge witness, invariant to arbitrary track-ID renaming."""
    base_to_new={};new_to_base={}
    for row in rows:
        base,new=row['baseline_id'],row['method_id']
        if base in base_to_new and base_to_new[base]['method_id']!=new:
            return dict(kind='split_baseline_identity',previous=base_to_new[base],current=row)
        if new in new_to_base and new_to_base[new]['baseline_id']!=base:
            return dict(kind='merge_baseline_identities',previous=new_to_base[new],current=row)
        base_to_new.setdefault(base,row);new_to_base.setdefault(new,row)
    return None


def detection_key(row):
    return (int(row['image_id']),int(row['category_id']),float(row['score']),*map(float,row['bbox']))


def read_run(path):
    report=json.loads((path/'manifest.json').read_text())
    if sha256(path/'predictions.json')!=report['predictions_sha256']:
        raise ValueError('changed predictions: '+str(path))
    return report,json.loads((path/'predictions.json').read_text())


def summarize_trace(rows,max_age):
    ages=collections.Counter()
    for row in rows:
        if 'query_feature_valid' not in row:raise ValueError('source-age diagnostic fields missing')
        if len(row['feature_ages'])!=len(row['query_feature_valid']):raise ValueError('trace dimension mismatch')
        for valid,age in zip(row['query_feature_valid'],row['feature_ages']):
            if valid:
                if age is None or not 0<=age<=max_age:raise ValueError('invalid/future/stale source age')
                ages[int(age)]+=1
            elif age is not None:raise ValueError('invalid feature must have null age')
    return dict(observed_association_calls=len(rows),query_observations=sum(row['query_count'] for row in rows),
                valid_query_feature_observations=sum(row['valid_query_features'] for row in rows),
                nonzero_residual_calls=sum(row['changed_logit_pairs']>0 for row in rows),
                counterfactual_assignment_change_calls=sum(row['counterfactual_assignment_changes']>0 for row in rows),
                counterfactual_assignment_changes=sum(row['counterfactual_assignment_changes'] for row in rows),
                valid_source_age_histogram=dict(sorted(ages.items())),
                maximum_abs_logit_delta=max((row['max_abs_logit_delta'] for row in rows),default=0),
                maximum_abs_gta_delta=max((row['max_abs_gta_delta'] for row in rows),default=0))


def main():
    parser=argparse.ArgumentParser()
    for name in ('cached-trace','dense-trace','baseline','dense-production','cached-production'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--tag',required=True);args=parser.parse_args()
    paths=dict(cached_trace=args.cached_trace,dense_trace=args.dense_trace,baseline=args.baseline,
               dense_production=args.dense_production,cached_production=args.cached_production)
    loaded={key:read_run(path) for key,path in paths.items()}
    first=loaded['baseline'][0]
    for report,_ in loaded.values():
        for field in ('video_id','checkpoint_sha256','config_sha256','annotation_sha256','seed','causal_online'):
            if report[field]!=first[field]:raise ValueError('unpaired diagnostic: '+field)
        if not report['causal_online']:raise ValueError('causal runs required')
    n=loaded['cached_trace'][0]['synchronized_groups']
    if loaded['dense_trace'][0]['synchronized_groups']!=n:raise ValueError('equal trace horizons required')
    scene=first['video_id'];records=video_records(train_data(),scene);index={row['image_id']:row for row in records}
    frames=sorted({row['frame_id'] for row in records})[:n]
    image_ids={row['image_id'] for row in records if row['frame_id'] in frames}
    prefixes={key:[row for row in predictions if row['image_id'] in image_ids] for key,(_,predictions) in loaded.items()}
    for name in ('cached','dense'):
        traced=loaded[name+'_trace'][0];production=loaded[name+'_production'][0]
        if traced['full_video'] or not production['full_video'] or traced['speed_eligible']:
            raise ValueError('partial diagnostic and complete production reference required')
        if traced['head_checkpoint_sha256']!=production['head_checkpoint_sha256']:
            raise ValueError('head mismatch')
        for field in ('FEATURE_WEIGHT','MAX_RESIDUAL','FEATURE_LAYERS','FEATURE_BACKBONE','PROJECTION_DIM','POOLING',
                      'SYMMETRIZE_VIEW_ORDER','GEOMETRY_WEIGHT','UPDATE_INTERVAL','UPDATE_POLICY','TOTAL_CALL_BUDGET'):
            if traced['geo_options'].get(field)!=production['geo_options'].get(field):
                raise ValueError('production/trace option mismatch: '+field)
        if prefixes[name+'_trace']!=prefixes[name+'_production']:
            raise ValueError('diagnostic changed actual production identity prefix')
    base={detection_key(row):row for row in prefixes['baseline']}
    dense={detection_key(row):row for row in prefixes['dense_production']}
    if len(base)!=len(prefixes['baseline']) or len(dense)!=len(prefixes['dense_production']) or base.keys()!=dense.keys():
        raise ValueError('exact unique detector membership required for first partition diagnosis')
    ordered=sorted(base,key=lambda key:(index[key[0]]['frame_id'],index[key[0]]['view_id'],key))
    pairs=[dict(frame_id=index[key[0]]['frame_id'],camera=index[key[0]]['view_id'],image_id=key[0],bbox=base[key]['bbox'],
                baseline_id=base[key]['track_id'],method_id=dense[key]['track_id']) for key in ordered]
    change=first_partition_change(pairs);traces={};sources={}
    for name in ('cached','dense'):
        report=loaded[name+'_trace'][0];trace=report['association_trace'];path=Path(trace['path'])
        if sha256(path)!=trace['sha256']:raise ValueError('changed association trace')
        traces[name]=json.loads(path.read_text());sources[str(path)]=sha256(path)
    focus={}
    if change:
        group=frames.index(change['current']['frame_id'])
        focus=dict(dataset_frame_id=change['current']['frame_id'],observed_group_index=group,
                   by_method={name:[row for row in rows if row.get('frame')==group and row.get('camera')==change['current']['camera']] for name,rows in traces.items()})
    save_json(ROOT/'results'/f'{args.tag}.json',dict(status='PASS',scene=scene,groups=n,
        traced_identity_prefix_exact_to_production=dict(cached=True,dense=True),
        cached_prefix_exact_baseline=prefixes['cached_production']==prefixes['baseline'],
        first_dense_partition_change=change,focus=focus,
        trace_summaries={name:summarize_trace(rows,loaded[name+'_trace'][0]['geo_options'].get('MAX_GEOMETRY_AGE',20)) for name,rows in traces.items()},
        manifest_sha256={str(path/'manifest.json'):sha256(path/'manifest.json') for path in paths.values()},trace_sha256=sources,
        limitations='Offline DEV diagnosis only, no GT read and no scheduler tuning. First partition change is a split/merge witness, not proof of correctness. Counterfactual assignment counts use each live method history, can repeat query observations, and are not actual independent-baseline corrections/harms or full-video metrics. Query feature ages do not describe every historical reference tensor.',
        speed_eligible=False,provenance=provenance()))


if __name__=='__main__':main()
