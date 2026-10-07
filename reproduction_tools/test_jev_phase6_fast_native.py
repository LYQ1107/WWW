"""Complete native B2 run: cache optimization must preserve every logit and prediction."""
import argparse
import json
import torch
from jev_phase6_common import OUT,REPORTS,sha,save,protect_anchor
from jev_phase6_rollouts import NativeReplayLab

def main(device,compact_context=False):
    lab=NativeReplayLab(1,device,native_match_validation=True,fast_match=True,compact_context=compact_context)
    count=0
    def verify(feature,question,legal,action,context):
        nonlocal count
        if question=='MATCH_DECISION' and legal!=['START_NEW']:
            with torch.no_grad():
                raw=lab.controller(feature.detach().cpu().reshape(1,64),[question],[legal])['logits']
                cached=lab.fast_match.logits(feature,legal)
            if not torch.equal(raw,cached):raise AssertionError('cache changes native first-round logits')
            count+=1
        return action
    result=lab.run(OUT/f"smoke/{'compact_context' if compact_context else 'fast_native'}_full_video01",intervention=verify)
    reference=json.loads((OUT/'native_b2/video01/result.json').read_text())
    if sha(result['predictions'])!=reference['prediction_sha256']:
        raise AssertionError('cache optimization changes complete native B2 predictions')
    report={'status':'PASS','video':1,'complete_sequence':True,'real_native_MATCH_logits_bitwise_equal':count,
        'prediction_sha256':sha(result['predictions']),'native_B2_prediction_identical':True,
        'native_second_round':'unchanged eager frozen B2','weights_changed':False,
        'what_did_we_learn':'Caching fixed semantic tokens preserves all first-round logits and complete native commits on the development sequence.'}
    report['compact_unused_diagnostic_context']=compact_context
    save(REPORTS/('COMPACT_NATIVE_COMPLETE_REGRESSION.json' if compact_context else 'FAST_NATIVE_COMPLETE_REGRESSION.json'),report);protect_anchor();print(json.dumps(report))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:0');p.add_argument('--compact-context',action='store_true')
    a=p.parse_args();main(a.device,a.compact_context)
