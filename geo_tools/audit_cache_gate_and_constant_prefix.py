"""Verify real gate telemetry and encoder-matched fixed-score prefix control."""
import collections
from common import *


def read_run(tag):
    path=RUNTIME/'online'/tag;run=json.loads((path/'manifest.json').read_text())
    if sha256(path/'predictions.json')!=run['predictions_sha256']:raise ValueError('changed predictions')
    trace=run['association_trace']
    if sha256(trace['path'])!=trace['sha256']:raise ValueError('changed trace')
    return path,run,json.loads((path/'predictions.json').read_text()),json.loads(Path(trace['path']).read_text())


def main():
    gate=read_run('cache_gate_trace40_scene1_v1');original=read_run('cache_trace40_scene1_v1')
    constant=read_run('constant_vggt_evidence_trace40_scene1_v1');dense=read_run('dense_trace40_scene1_v1')
    runs=(gate,original,constant,dense)
    for _,run,_,_ in runs:
        if not run['causal_online'] or run['full_video'] or run['synchronized_groups']!=40 or run['speed_eligible'] or run['geo_failures']:
            raise ValueError('clean40-group causal diagnostic required')
        for field in ('checkpoint_sha256','config_sha256','annotation_sha256','head_checkpoint_sha256','seed','video_id','images'):
            if run[field]!=gate[1][field]:raise ValueError('unpaired diagnostic: '+field)
    if gate[2]!=original[2]:raise ValueError('gate telemetry changes production prefix')
    if gate[1]['target_feature_cache']!=original[1]['target_feature_cache']:
        raise ValueError('gate telemetry changes cache state/counters')
    observations=[row for row in gate[3] if row['cache_match_gates']['status']=='CURRENT_REUSE_MATCH_RECOMPUTED']
    if not observations or not all(row['cache_match_gates']['original_matches_actual_valid'] for row in observations):
        raise ValueError('recomputed gates disagree with actual reused features')
    counts=collections.Counter();focus=[]
    for row in observations:
        diagnostics=row['cache_match_gates'];original_gates=diagnostics['original']['gate_pass']
        for index,accepted in enumerate(diagnostics['original']['accepted']):
            counts['query_reuse_observations']+=1;counts['accepted']+=accepted
            if not accepted:
                failed=[name for name,values in original_gates.items() if not values[index]]
                for name in failed:counts['failed_'+name]+=1
                if len(failed)==1:counts['failed_only_'+failed[0]]+=1
            counts['spatial_pool_counterfactual_accepted']+=diagnostics['spatial_filter_counterfactual']['accepted'][index]
        if row.get('frame')==26 and row.get('camera')==1:focus.append(row)
    if counts['accepted']!=gate[1]['target_feature_cache']['stats']['reused_targets']:
        raise ValueError('per-call observed reuse does not cover actual prefix cache statistics')
    if constant[1].get('validation_constant_head_evidence')!=1. or dense[1].get('validation_constant_head_evidence') is not None:
        raise ValueError('explicit learned-versus-constant evidence comparison required')
    if constant[1]['total_vggt_calls']!=dense[1]['total_vggt_calls'] or constant[1]['total_vggt_calls']!=80:
        raise ValueError('actual encoder-matched control required')
    feature_fields=('frame','camera','query_count','history_count','query_feature_valid','feature_source_frames','feature_ages','query_boxes_original_xyxy')
    if [{key:row.get(key) for key in feature_fields} for row in constant[3]]!=[{key:row.get(key) for key in feature_fields} for row in dense[3]]:
        raise ValueError('feature eligibility/observations differ between constant and learned control')
    save_json(ROOT/'results/cache_gate_and_constant_prefix_audit_v1.json',dict(status='PASS_SCOPED_DIAGNOSTIC',groups=40,scene=1,
        gate_replay_exact_original_identity_prefix=True,gate_replay_exact_cache_counters=True,
        gate_observations=dict(counts),critical_group26_camera1=focus,cache_thresholds=gate[1]['target_feature_cache'],
        constant_evidence_exact_dense_identity_prefix=constant[2]==dense[2],
        constant_evidence_exact_baseline_identity_prefix=constant[2]==original[2],
        equal_actual_vggt_calls=80,equal_feature_validity_and_observations=True,
        assignment_counterfactual_changes={name:sum(row['counterfactual_assignment_changes'] for row in value[3]) for name,value in [('constant',constant),('learned',dense)]},
        source_manifests={str(value[0]/'manifest.json'):sha256(value[0]/'manifest.json') for value in runs},
        source_traces={value[1]['association_trace']['path']:value[1]['association_trace']['sha256'] for value in runs},
        interpretation='For this early prefix, exact learned-head tracking behavior can be reproduced by a fixed+1 pair score while retaining actual VGGT encoders/eligibility. This does not establish the full-video or full-DEV constant-prior result. The spatial candidate pool is only a diagnostic and never used for tracking.',
        limitations='Specific early failure diagnosis, not a DEV threshold search. Gate failures can overlap. Counterfactual assignments share each method live history and are not independent-baseline GT correction counts. All runs speed-ineligible; equal calls do not establish a speed benefit.',provenance=provenance()))


if __name__=='__main__':main()
