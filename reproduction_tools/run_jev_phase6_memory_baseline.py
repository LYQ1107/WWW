"""Fixed lightweight MEMORY baselines under frozen B2 MATCH and native READ."""
import argparse
import json
from pathlib import Path

import torch

from jev_phase6_common import OUT,B2,save,sha,protect_anchor,new_output
from jev_phase6_rollouts import NativeReplayLab


def main(video,condition,device,native_contract=False):
    output=OUT/f"{'memory_baselines_native' if native_contract else 'memory_baselines'}/video{video:02d}/{condition}"
    if (output/'result.json').exists():
        complete=json.loads((output/'result.json').read_text())
        if (complete['status']=='PASS' and complete['video']==video and complete['condition']==condition
                and complete.get('native_MATCH_transition_contract')==native_contract
                and complete.get('checkpoint_sha256')==sha(B2)
                and complete['prediction_sha256']==sha(complete['result']['predictions'])):
            protect_anchor();print(json.dumps({'status':'REUSED_COMPLETE_BOUND_RUN','video':video,'condition':condition}));return
        raise RuntimeError('existing result binding does not match this baseline')
    lab=NativeReplayLab(video,device,native_match_validation=native_contract,fast_match=native_contract,compact_context=native_contract)
    collection='lifecycle_native' if native_contract else 'lifecycle'
    if native_contract:
        manifest=json.loads((OUT/f'{collection}/video{video:02d}/CANONICAL_MANIFEST.json').read_text())
        if not manifest['native_MATCH_transition_contract']:raise AssertionError('canonical native prefix required')
    prototypes={}
    original_candidates=lab.pilot.reactivation_candidates
    def candidates(state,bank_size=10):
        existing=set(state.reactivation_bank)
        ids,recent=original_candidates(state,bank_size=bank_size)
        if condition in {'M3','M4','M5'}:
            for track in set(ids)-existing:
                if track in prototypes:state.reactivation_bank[track]=prototypes[track].detach().cpu().clone()
        return ids,recent
    lab.pilot.reactivation_candidates=candidates
    def choose(feature,question,legal,action,context):
        if condition=='M1' and question=='MEMORY_DECISION':
            score=float(lab.current['detection_scores'][context['detection_index']])
            return 'WRITE_MEMORY' if score>=.6 else 'SKIP_MEMORY'
        return action
    def after(key,payload,state,result,kwargs,rows):
        if condition=='M2':
            for track in state.memory:state.memory[track]=state.memory[track][-10:]
        if condition not in {'M3','M4','M5'}:return
        for row,action in kwargs.get('memory_actions',{}).items():
            if action!='WRITE_MEMORY':continue
            track=int(result['committed_track_ids'][row]);feature=torch.as_tensor(payload['reid_features'][row]).float().cpu()
            confidence=float(payload['detection_scores'][row])
            old_weight=.9 if condition=='M3' else .2 if condition=='M5' else .9+.1*(1-max(0,min(1,(confidence-.4)/.6)))
            prototypes[track]=feature.clone() if track not in prototypes else old_weight*prototypes[track]+(1-old_weight)*feature
    if condition=='M0':
        new_output(output);lab.pilot.PILOT=output
        result=json.loads((OUT/f'{collection}/video{video:02d}/baseline/result.json').read_text())
    else:result=lab.run(output,intervention=choose,after_step=after)
    dataset=lab.pilot.prepare_eval_dataset(lab.subset)
    prepared,evaluated=lab.pilot.run_eval('jev',Path(result['predictions']),dataset)
    metrics=lab.pilot.extract_metrics(evaluated)
    value={'status':'PASS','video':video,'condition':condition,'metrics':metrics,'result':result,
           'prediction_sha256':sha(result['predictions']),'baseline_B2_MATCH_frozen':True,
           'representation':'raw GMT ReID scale; EMA weights borrowed from sources, not their full normalized pipelines',
           'bank_eligibility':'native minimum10 actual writes, stale candidate computation and persistent bank order',
           'M0_reuses_identical_collected_native_run':condition=='M0',
           'official_test_read':False,'full24_authorized':False}
    value['native_MATCH_transition_contract']=native_contract
    value['checkpoint_sha256']=sha(B2)
    if condition=='M2':
        reference=json.loads((OUT/f'{collection}/video{video:02d}/baseline/result.json').read_text())
        value['latest10_bank_equivalence_prediction_identical']=sha(result['predictions'])==sha(reference['predictions'])
        if not value['latest10_bank_equivalence_prediction_identical']:raise AssertionError('bounded latest10 gallery changed the native last10 mean')
    save(output/'result.json',value);protect_anchor()
    print(json.dumps({'video':video,'condition':condition,'metrics':metrics}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--video',type=int,choices=[23,24],required=True)
    p.add_argument('--condition',choices=['M0','M1','M2','M3','M4','M5'],required=True);p.add_argument('--device',default='cuda:0')
    p.add_argument('--native-contract',action='store_true')
    a=p.parse_args();main(a.video,a.condition,a.device,a.native_contract)
