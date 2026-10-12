"""Fixed same-state native H64 interventions; future actions/state are recomputed."""
import argparse
import gc
import torch
from jev_oneweek_common import *
from jev_oneweek_online import policy


def main(arms,seed,phase,video):
    protect();assert video==16
    protocol=REPORTS/'PAIRED_NATIVE_PROTOCOL.json'
    setup=dict(status='FROZEN_BEFORE_PAIRED_RESULTS',video=16,source='Existing immutable XV frozen-Multi native prefix32/0',
               start_frame=32,stop_frame=95,horizon=64,arms=arms,seed=seed,phase=phase,
               no_result_selected_event=True,all_future_actions_recomputed=True,
               interpretation='Exact common parent state; not independent full-video cold starts and not future action-value training')
    if protocol.exists():assert read(protocol)==setup
    else:save(protocol,setup)
    from jev_phase14_artifacts import load_dense
    from jev_phase13_runtime import build_tracker,cache_inputs,run
    from gtr.modeling.jev_oneweek.native import attach_jevtrack
    from gtr.modeling.jev_native_state import fingerprint
    prefix_path=PHASE15/'commitment_dataset_v2/video16/NATIVE_PREFIX_32_0.pth.xz'
    prefix_ref=ref(prefix_path);initial=load_dense(prefix_path,map_location='cpu');expected=fingerprint(initial)
    values,frames,reader=cache_inputs(video);cases=[]
    for arm in arms:
        folder=OUT/'paired_native_v1'/phase/arm/f'seed{seed}'/'attempt01'
        folder.mkdir(parents=True,exist_ok=False)
        actor,checkpoint=policy(arm,seed,phase);model=build_tracker(video,actor,react_learned=False);executor=attach_jevtrack(model)
        prefix=load_dense(prefix_path,map_location='cuda:0');assert fingerprint(prefix)==expected
        commits=[];first=[None]
        def before(**d):
            if first[0] is None:
                first[0]=dict(key=[d['context']['frame'],d['context']['view']],refs=d['refs'],
                              state_input_sha=fingerprint(d['batch']))
        def after(**d):
            commits.append(dict(key=[d['frame'],d['view']],ids=d['instances'][-1].track_ids.cpu().tolist(),events=d['events']))
        executor.observer=before;executor.commit_observer=after
        with torch.no_grad():raw,_=run(model,values,frames,stop=95,prefix=prefix)
        assert len(commits)==128
        from run_jev_phase10_closed_loop import raw_predictions
        from jev_phase13_common import ANNOTATIONS
        images={(i['frame_id']-1,i['view_id']-1):i for i in read(ANNOTATIONS)['images'] if i['video_id']==video}
        predictions=raw_predictions(raw,images);save(folder/'RAW_PREDICTIONS.json',predictions)
        from jev_phase7_offline import IdentityEvaluator
        evaluator=IdentityEvaluator(video);aligned=evaluator.align(predictions)
        duplicate={}
        import collections
        for image in evaluator.images.values():
            counts=collections.Counter(a['instance_id'] for a in evaluator.gt[image['id']])
            duplicate[image['frame_id']-1,image['view_id']-1]={g for g,n in counts.items() if n>1}
        owners={};status=[]
        for key,value in sorted(aligned.items()):
            gt=value['gt'];identity=value['id']
            if gt in duplicate[key[:2]]:gt=None
            if gt is not None:owners.setdefault(identity,gt)
            if key[0]>=32:
                status.append(dict(key=key,identity=identity,GT_OFFLINE_ONLY=gt,owner=owners.get(identity),
                                   correct=owners.get(identity)==gt if gt is not None else None))
        result=dict(status='COMPLETE',arm=arm,checkpoint=checkpoint,prefix=prefix_ref,
                    identical_starting_state_sha=expected,first_actual_input=first[0],committed_payloads=128,
                    predictions=ref(folder/'RAW_PREDICTIONS.json'),future_known_wrong_observations=sum(r['correct'] is False for r in status),
                    UNKNOWN_observations=sum(r['correct'] is None for r in status),
                    current_future_H64_same_state_scope=True,full_video_cold_start_scope=False,
                    original_prefix_states_or_results_overwritten=False)
        save(folder/'ROW_STATUS.json',status);save(folder/'COMMITS.json',commits);save(folder/'RESULT.json',result);cases.append(result)
        del raw,model,actor,executor,prefix;gc.collect();torch.cuda.empty_cache()
    assert len({c['first_actual_input']['state_input_sha'] for c in cases})==1
    save(REPORTS/'PAIRED_NATIVE_RESULTS.json',dict(status='COMPLETE',protocol=ref(protocol),cases=cases,
         exact_initial_tensor_candidate_and_state_parity=True,GT_or_cached_future_actions_in_inference=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arms',nargs='+',default=['parent','jev','mlp','attention','extra_hungarian'])
    p.add_argument('--seed',type=int,default=20261009);p.add_argument('--phase',default='formal');p.add_argument('--video',type=int,default=16);a=p.parse_args()
    try:main(a.arms,a.seed,a.phase,a.video)
    except Exception:
        import traceback
        failure('paired_native',traceback.format_exc());raise
