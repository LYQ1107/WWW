"""Supplemental native-equivalent B2 run; never overwrite the positive anchor."""
import argparse
from collections import Counter
import json
from pathlib import Path

from jev_phase6_common import OUT,REPORTS,B2,sha,save,protect_anchor
from jev_phase6_rollouts import NativeReplayLab


def main(video,device):
    lab=NativeReplayLab(video,device,native_match_validation=True,fast_match=True)
    output=OUT/f'native_b2/video{video:02d}'
    counts=Counter();validation=[]
    def after(key,payload,state,result,kwargs,rows):
        counts.update((r['question'],r['action']) for r in rows)
        validation.extend(lab.native_resolver.validations)
    result=lab.run(output,after_step=after)
    save(output/'native_round2_decisions.json',validation)
    dataset=lab.pilot.prepare_eval_dataset(lab.subset)
    prepared,evaluated=lab.pilot.run_eval('jev',Path(result['predictions']),dataset)
    metrics=lab.pilot.extract_metrics(evaluated)
    value={'status':'PASS','video':video,'condition':'G5_NATIVE_TRANSITION','metrics':metrics,'result':result,
           'checkpoint_sha256':sha(B2),'prediction_sha256':sha(result['predictions']),
           'native_second_round_calls':len(validation),'native_second_round_actions':dict(Counter(r['action'] for r in validation)),
           'first_round_actions':{q:{a:n for (qq,a),n in counts.items() if qq==q} for q in ['MATCH_DECISION','MEMORY_DECISION','REACTIVATION_DECISION']},
           'frozen_research_anchor_overwritten':False,'official_test_read':False,'full24_authorized':False}
    save(output/'result.json',value);protect_anchor();print(json.dumps({k:v for k,v in value.items() if k!='result'}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--video',type=int,choices=[1,2,3,5],required=True);p.add_argument('--device',default='cuda:0')
    a=p.parse_args();main(a.video,a.device)
