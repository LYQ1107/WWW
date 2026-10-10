"""A new policy's real mutated future from frozen TRAIN diagnostic prefixes."""
import argparse
import gzip
import time
import torch
from jev_phase15_common import *
from jev_phase13_runtime import build_tracker, cache_inputs, run
from jev_phase14_artifacts import load_dense
from jev_phase14_native_risk import NativeRisk
from jev_phase15_causal_metrics import FrozenClearFuture
from jev_phase15_causal_commitment import risk_at_horizon
from gtr.modeling.jev_phase15.commitment_state import CommitmentState
from gtr.modeling.jev_phase15.native_commit_adapter import attach
from gtr.modeling.jev_phase15.model import PersistentIdentityPolicy
from gtr.modeling.jev_native_state import fingerprint


def converted_prefix(prefix):
    boundary=tuple(prefix['key'][1:]);state=CommitmentState()
    for index,item in enumerate(prefix['instances']):
        frame,view=divmod(index,2)
        if (frame,view)>=boundary:break
        if not item.has('track_ids'):assert len(item)==0;continue
        for row,identity in enumerate(item.track_ids.tolist()):
            state.update(identity,item.reid_features[row,:1024],item.pred_boxes.tensor[row],item.image_size,frame,view)
    prefix['phase13_identity_meta']=dict(version=1,identity_memory=prefix['phase13_identity_meta'],commitment_state=state.state_dict())
    return prefix


def main(video,version=1,arm='F_full',phase='pilot'):
    protect();assert video in TRAIN
    training=OUT/f'training_full_payload_v{version}'/arm/'seed20261009'/phase/'RESULT.json'
    result=read(training);assert result['status']=='COMPLETE'
    checkpoint=result['checkpoint'];assert sha(checkpoint['path'])==checkpoint['SHA256']
    torch.set_num_threads(1);torch.manual_seed(20261009)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    policy=PersistentIdentityPolicy(arm).cuda().eval();policy.load_state_dict(torch.load(checkpoint['path'],map_location='cpu')['model'],strict=True)
    case=OUT/'train_commitment_prefixes_v1'/f'video{video:02d}';manifest=read(case/'RESULT.json')
    values,frames,reader=cache_inputs(video)
    metric=FrozenClearFuture('multi_question',20261009,video,case_override=case,duplicate_gt_diagnostic=video==14)
    out=OUT/f'pilot_native_v{version}'/arm/phase/f'video{video:02d}';out.mkdir(parents=True,exist_ok=True)
    assert not (out/'RESULT.json').exists()
    source=binding(seed=20261009,checkpoints=[checkpoint],dataset=ref(case/'RESULT.json'),
        evaluator=metric.evaluator,scope='matched TRAIN start, new entire continuation policy; not isolated single-action attribution or full-video tracking')
    source['perception_input_provenance']=perception_provenance(video)
    outcomes=[];begin=time.monotonic()
    for entry in manifest['counterfactual_prefixes']:
        descriptor=entry['prefix'];assert sha(descriptor['path'])==descriptor['SHA256']
        prefix=load_dense(descriptor['path'],map_location='cuda:0')
        assert fingerprint(prefix)==entry['starting_state_SHA256']
        prefix=converted_prefix(prefix);initial=fingerprint(prefix)
        boundary=tuple(entry['key'][1:]);model=build_tracker(video,policy=policy,react_learned=False);executor=attach(model)
        risk=NativeRisk(video,reader,prefix);risk.prefix_votes={identity:dict(votes) for identity,votes in risk.votes.items()}
        risk.prefix_latest=dict(risk.latest);risk.prefix_ever=set(risk.ever)
        executor.observer=risk.before;executor.commit_observer=risk.after
        with torch.no_grad():raw,_=run(model,values,frames,stop=min(frames-1,boundary[0]+31),prefix=prefix)
        ids=metric.ids(raw);horizons={}
        folder='causal_commitment_v2_duplicate_diagnostic' if video==14 else 'causal_commitment_v1'
        baseline_path=OUT/folder/'pi_multi_frozen20k'/f'video{video:02d}'/(f'frame{boundary[0]:06d}_view{boundary[1]}_row{entry["row"]}_SELECT_JEV_BEST_ID.json')
        baseline=read(baseline_path);assert baseline['status']=='COMPLETE'
        assert baseline['starting_state_SHA256']==entry['starting_state_SHA256']
        for horizon in [8,16,32]:
            key=str(horizon);future=metric.horizon(ids,boundary,horizon,frames)
            actual=risk_at_horizon(risk,boundary,min(frames,boundary[0]+horizon))
            old=baseline['H'][key];fields=['new_cross_GT_gallery_mix','false_birth','false_split_birth','writes_into_already_polluted_history',
                'pure_fragment_hops','history_corrective_switches','past_pure_owner_wrong_observations','UNKNOWN_selected']
            delta={field:actual['counts'].get(field,0)-old['identity_consequences']['counts'].get(field,0) for field in fields}
            delta['future_CLEAR_IDSW']=future['future_CLEAR_IDSW']-old['future_CLEAR_IDSW']
            horizons[key]=dict(**future,identity_consequences=actual,delta_vs_original_entire_pi_multi=delta)
        first=next(item for item in risk.trace if item['key']==list(boundary) and item['row']==entry['row'])
        artifact=out/f'frame{boundary[0]:06d}_view{boundary[1]}_row{entry["row"]}.jsonl.gz'
        with gzip.open(artifact,'wt') as stream:
            for item in risk.trace:stream.write(json.dumps(item,allow_nan=False)+'\n')
        item=dict(key=entry['key'],row=entry['row'],category=entry['category'],
            original_prefix=descriptor,original_native_state_SHA256=entry['starting_state_SHA256'],derived_causal_commitment_state_SHA256=initial,
            reconstructed_state_contains_GT_or_future=False,actual_mutated_future=True,
            first_actual_ID=first['identity'],first_actual_action=first['action'],first_selected_tag=first['selected_tag'],
            first_previous_ID=entry['previous_native_id'],H=horizons,baseline=ref(baseline_path),trace=ref(artifact),
            strict_CLEAR_gate_eligible=video!=14)
        outcomes.append(item)
        save(out/'PROGRESS.json',dict(status='RUNNING',done=len(outcomes),total=len(manifest['counterfactual_prefixes']),last_key=entry['key']))
        del prefix,model,risk;torch.cuda.empty_cache()
        print('PHASE15_PILOT_NATIVE',version,video,entry['key'],{h:d['delta_vs_original_entire_pi_multi'] for h,d in horizons.items()},flush=True)
    save(out/'RESULT.json',dict(status='COMPLETE',binding=source,version=version,arm=arm,phase=phase,video=video,
        cases=outcomes,seconds=time.monotonic()-begin,all_declared_prefixes_retained=True,
        policy_specific_value='pi_new_'+arm,original_policy='pi_multi_frozen20k',purely_single_action_causal_attribution=False))
    save(out/'PROGRESS.json',dict(status='COMPLETE',done=len(outcomes)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--video',type=int,required=True);p.add_argument('--version',type=int,default=1)
    p.add_argument('--arm',default='F_full');p.add_argument('--phase',default='pilot');a=p.parse_args()
    main(a.video,a.version,a.arm,a.phase)
