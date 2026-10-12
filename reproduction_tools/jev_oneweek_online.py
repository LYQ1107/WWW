"""Complete raw mutated-state multi-camera tracking, with GT evaluated afterwards."""
import argparse
import collections
import gzip
import traceback
import torch
from torch import nn
from jev_oneweek_common import *
from jev_oneweek_assess import load_model
from gtr.modeling.jev_oneweek.plans import plans
from gtr.modeling.jev_stage2.assignment import lawful_choice


class FrozenPlanControl(nn.Module):
    def __init__(self,identity,kind):
        super().__init__();self.identity=identity;self.kind=kind
        for p in self.parameters():p.requires_grad_(False)
    def forward(self,x):
        z=self.identity({k:v for k,v in x.items() if k!='commitment_features'})
        if not len(z[0]):self.runtime_choice=[];return z
        candidates=plans(z[0],x['legal'][0]) if self.kind!='parent' else [tuple(lawful_choice(z[0],x['legal'][0]))]
        modified=z[0].clone()
        if self.kind=='fixed_threshold':modified[:,:-1]-=.1
        elif self.kind=='fixed_fusion':
            p=x['pair_evidence'][0];c=x['commitment_features'][0]
            modified[:,:-1]+=.1*(p[:,:,3]+c[:,:,0]-p[:,:,10:14].square().sum(-1).sqrt())
        q,k=x['legal'][0].shape
        scores=[sum(float(modified[r,k if col<0 else col]) for r,col in enumerate(plan)) for plan in candidates]
        selected=max(range(len(scores)),key=lambda i:scores[i])
        self.runtime_choice=list(candidates[selected])
        self.runtime_details=dict(plans=[candidates],plan_scores=[torch.tensor(scores,device=z.device)],base_logits=z)
        return z


class SolverFactorControl(nn.Module):
    """Same learned edge scorer with one/repeated/constraint-changing solves.

    This isolates the additive component. Hungarian cannot optimize the full
    non-additive joint plan value, which remains a separate same-plan comparison.
    """
    def __init__(self,model,kind):
        super().__init__();self.model=model;self.kind=kind
        self.solve_calls=0;self.masked_edges=0
        self.triggered_events=0;self.reassigned_rows=0
    def forward(self,x):
        if not x['question_mask'].any():
            self.runtime_choice=[]
            return x['detection_visual'].new_empty(1,0,x['identity_mask'].shape[1]+1)
        details=self.model.encode(x)
        score=details['base_logits'] if self.kind.startswith('base_') else details['edge_logits']
        legal=x['legal'][0];choice=lawful_choice(score[0],legal);self.solve_calls+=1
        if self.kind.endswith('repeat'):
            repeat=lawful_choice(score[0],legal);self.solve_calls+=1
            assert repeat==choice,'Unchanged Hungarian rerun must reproduce the same result'
        elif self.kind.endswith('rule'):
            # A frozen rejection rule changes the mask. This is deliberately
            # separate from the identical-input solver-count-only intervention.
            proposed=lawful_choice(details['base_logits'][0],legal)
            mask=legal.clone();rejected=0
            for row,col in enumerate(proposed):
                if col>=0 and float(details['base_logits'][0,row,col]-details['base_logits'][0,row,-1])<.1:
                    mask[row,col]=False;rejected+=1
            choice=lawful_choice(score[0],mask);self.solve_calls+=2;self.masked_edges+=rejected
            self.triggered_events+=bool(rejected)
        self.reassigned_rows+=sum(a!=b for a,b in zip(lawful_choice(details['base_logits'][0],legal),choice))
        self.runtime_choice=list(choice)
        self.runtime_details=dict(base_logits=details['base_logits'],edge_logits=details['edge_logits'])
        return score


def policy(arm,seed,phase):
    from jev_phase15_common import load_old_policy
    if arm=='cosine':return None,None
    if arm in ['parent','extra_hungarian','fixed_threshold','fixed_fusion']:
        original,checkpoint,_=load_old_policy('multi_question',seed)
        return FrozenPlanControl(original,arm).eval(),checkpoint
    if arm in ['base_single','base_repeat','base_rule','edge_single','edge_repeat','edge_rule']:
        network,checkpoint,_=load_model('jev',seed,phase)
        return SolverFactorControl(network,arm).eval(),checkpoint
    if arm=='XIII_fixed_reference':
        from gtr.modeling.jev_stage2.model import GlobalIdentityJev
        manifest=read(REPORTS/'REUSED_EVIDENCE_REGISTRY.json')
        descriptor=next(r['asset'] for r in manifest['weights'] if r['label']==f'XIII fixed_question seed{seed}')
        assert sha(descriptor['path'])==descriptor['SHA256']
        network=GlobalIdentityJev('fixed_question').cuda().eval()
        network.load_state_dict(torch.load(descriptor['path'],map_location='cpu')['model'],strict=True)
        return FrozenPlanControl(network,'parent').eval(),descriptor
    if arm=='XV_V3_reference':
        assert seed==20261009,'Only the actual qualified frozen XV V3 seed09 pilot exists'
        from jev_phase15_evaluate import load_policy
        return load_policy('F_full',seed,3,'pilot')[:2]
    return load_model(arm,seed,phase)[:2]


def main(arm,seed,phase,video):
    protect();assert video in TRAIN+DEV;seed_all()
    folder=OUT/'online_v1'/phase/arm/f'seed{seed}'/f'video{video:02d}'
    folder.mkdir(parents=True,exist_ok=False)
    from jev_phase13_runtime import build_tracker,cache_inputs,run
    from gtr.modeling.jev_oneweek.native import attach_jevtrack
    from jev_phase14_latency import prune
    from gtr.modeling.jev_native_state import fingerprint
    from run_jev_phase10_closed_loop import raw_predictions,metrics
    actor,checkpoint=policy(arm,seed,phase)
    model=build_tracker(video,policy=actor,react_learned=False)
    prune(model);executor=attach_jevtrack(model)
    values,frames,reader=cache_inputs(video);source=binding('Complete actual two-camera raw native mutated-state tracking')
    counts=collections.Counter();final=[None];stamp=[0.];begin=time.monotonic()
    journal=gzip.open(folder/'COMMITS.jsonl.gz','wt')
    def after(**d):
        ids=d['instances'][-1].track_ids.cpu().tolist();assert len(ids)==len(set(ids))
        counts['camera_commits']+=1;counts['detections']+=len(ids)
        for event in d['events']:counts[event['action']]+=1
        detail=getattr(actor,'runtime_details',{}) if actor is not None else {}
        if detail.get('plans') and detail['plans'][0]:
            scores=detail['plan_scores'][0];counts['plans_seen']+=len(scores)
            counts['nonbase_plans']+=int(int(scores.argmax())!=0)
        journal.write(json.dumps(dict(key=[video,d['frame'],d['view']],ids=ids,events=d['events']))+'\n')
        if d['frame']==frames-1 and d['view']==1:
            from gtr.modeling.meta_arch.gtr_rcnn import poss_ids,old_reids
            final[0]=fingerprint(dict(memory=executor.memory.state_dict(),id_count=d['id_count'],
                hits=d['hits'],galleries=d['galleries'],bank=old_reids.old_reids,possible_ids=poss_ids.poss_ids))
        if time.monotonic()-stamp[0]>=10:
            journal.flush();save(folder/'PROGRESS.json',dict(status='NATIVE_TRACKING',key=[video,d['frame'],d['view']],frames=frames,counts=dict(counts),elapsed=time.monotonic()-begin))
            print('NATIVE_ONLINE_PROGRESS',arm,video,d['frame'],frames,flush=True);stamp[0]=time.monotonic()
    executor.commit_observer=after
    save(folder/'START.json',dict(binding=source,checkpoint=checkpoint,video=video,frames=frames,
        arm=arm,seed=seed,current_perception_index=ref(PHASE13/'stage1_cache_v1'/f'video{video:02d}'/'index.jsonl'),GT_actor_inputs=False))
    try:
        with torch.no_grad():raw,_=run(model,values,frames)
    finally:journal.close()
    assert counts['camera_commits']==frames*2-1
    solver_diagnostics={key:getattr(actor,key) for key in
                        ['solve_calls','masked_edges','triggered_events','reassigned_rows']
                        if actor is not None and hasattr(actor,key)}
    from jev_phase13_common import ANNOTATIONS
    images={(i['frame_id']-1,i['view_id']-1):i for i in read(ANNOTATIONS)['images'] if i['video_id']==video}
    predictions=raw_predictions(raw,images);save(folder/'RAW_PREDICTIONS.json',predictions)
    save(folder/'PROGRESS.json',dict(status='FROZEN_OFFLINE_METRICS',frames=frames,counts=dict(counts)))
    score,metric_root=metrics(folder/'RAW_PREDICTIONS.json',[video],folder/'strict_eval')
    # Identity diagnostics use immutable first-known owners and actual committed
    # predictions; do not infer correction from IDSW or mix same-state forks.
    from jev_phase7_offline import IdentityEvaluator
    evaluator=IdentityEvaluator(video);aligned=evaluator.align(predictions);identity=evaluator.summarize(aligned)
    save(folder/'IDENTITY_ANALYSIS.json',identity)
    result=dict(status='COMPLETE',binding=source,arm=arm,seed=seed,phase=phase,video=video,frames=frames,
                counts=dict(counts),solver_diagnostics=solver_diagnostics,
                solver_diagnostics_scope='Factor controls: counted solver calls exclude diagnostic base-choice calculation; reassigned rows compare same-state frozen base assignment',
                raw_online_metrics=score,metrics_source=ref(metric_root/'metrics.json'),
                raw_predictions=ref(folder/'RAW_PREDICTIONS.json'),commits=ref(folder/'COMMITS.jsonl.gz'),
                actual_mutated_state_online=True,no_GT_actor_input=True,current_Bank_NEW_fallback_unmodified=True,
                frozen_four_slot_history_and_native_causal_order=True,per_camera_capacity_one=True,
                final_state_sha256=final[0],full_scene_FPS=None,
                efficiency_scope='Cached perception with journals/evaluation; not image-to-tracks FPS',
                identity_summary={k:v for k,v in identity.items() if k not in ['wrong_ID_episodes','identity_index_query']},
                cross_camera={k:v for k,v in identity['identity_index_query'].items() if k!='queries'},
                raw_online_primary=True,future_short_track_filter_applied=False,
                elapsed_seconds_including_eval=time.monotonic()-begin,official_TEST=False,sealed20_21_22=True)
    save(folder/'RESULT.json',result)
    save(REPORTS/f'ONLINE_{arm}_{phase}_SEED{seed}_VIDEO{video:02d}.json',result)
    print('NATIVE_ONLINE_COMPLETE',arm,video,score,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',required=True);p.add_argument('--seed',type=int,default=20261009)
    p.add_argument('--phase',default='pilot');p.add_argument('--video',type=int,required=True);a=p.parse_args()
    try:main(a.arm,a.seed,a.phase,a.video)
    except Exception:
        failure('online',traceback.format_exc());raise
