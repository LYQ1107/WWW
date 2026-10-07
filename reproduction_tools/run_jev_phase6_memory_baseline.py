"""Fixed lightweight MEMORY baselines under frozen B2 MATCH and native READ."""
import argparse
import json
from pathlib import Path

import torch

from jev_phase6_common import OUT,save,sha,protect_anchor
from jev_phase6_rollouts import NativeReplayLab


def main(video,condition,device):
    lab=NativeReplayLab(video,device)
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
    output=OUT/f'memory_baselines/video{video:02d}/{condition}'
    result=lab.run(output,intervention=choose,after_step=after)
    dataset=lab.pilot.prepare_eval_dataset(lab.subset)
    prepared,evaluated=lab.pilot.run_eval('jev',Path(result['predictions']),dataset)
    metrics=lab.pilot.extract_metrics(evaluated)
    value={'status':'PASS','video':video,'condition':condition,'metrics':metrics,'result':result,
           'prediction_sha256':sha(result['predictions']),'baseline_B2_MATCH_frozen':True,
           'representation':'raw GMT ReID scale; EMA weights borrowed from sources, not their full normalized pipelines',
           'bank_eligibility':'native minimum10 actual writes, stale candidate computation and persistent bank order',
           'official_test_read':False,'full24_authorized':False}
    if condition=='M2':
        reference=json.loads((OUT/f'lifecycle/video{video:02d}/baseline/result.json').read_text())
        value['latest10_bank_equivalence_prediction_identical']=sha(result['predictions'])==sha(reference['predictions'])
        if not value['latest10_bank_equivalence_prediction_identical']:raise AssertionError('bounded latest10 gallery changed the native last10 mean')
    save(output/'result.json',value);protect_anchor()
    print(json.dumps({'video':video,'condition':condition,'metrics':metrics}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--video',type=int,choices=[23,24],required=True)
    p.add_argument('--condition',choices=['M0','M1','M2','M3','M4','M5'],required=True);p.add_argument('--device',default='cuda:0')
    a=p.parse_args();main(a.video,a.condition,a.device)
