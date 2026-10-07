"""Require exact cached/eager logits on actual frozen-B2 first-round states."""
import json
import time
import torch

from jev_phase6_common import OUT,REPORTS,B2,load_controller,save,protect_anchor
from jev_phase6_fast_match import FastMatchPolicy


def main():
    model=load_controller(B2);fast=FastMatchPolicy(model);count=0;max_error=0.
    for line in (OUT/'gating_tracking/G5/online_decisions.jsonl').open():
        row=json.loads(line)
        if row['question']!='MATCH_DECISION' or row['legal_actions']==['START_NEW']:continue
        x=torch.tensor(row['feature_vector']).reshape(1,64);legal=row['legal_actions']
        with torch.no_grad():eager=model(x,['MATCH_DECISION'],[legal])['logits'];traced=fast.logits(x,legal)
        max_error=max(max_error,float((eager-traced).abs().max()))
        if not torch.equal(eager,traced):raise AssertionError('fast first-round graph changes real B2 logits')
        if fast.action(x,legal)!=row['action']:raise AssertionError('fast first-round graph changes a B2 action')
        count+=1
    x=torch.tensor(row['feature_vector']).reshape(1,64);legal=['ACCEPT_CURRENT','REASSOCIATE','START_NEW']
    stamps=[]
    for call in (lambda:model(x,['MATCH_DECISION'],[legal]),lambda:fast.logits(x,legal)):
        start=time.monotonic()
        with torch.no_grad():
            for _ in range(1000):call()
        stamps.append(time.monotonic()-start)
    report={'status':'PASS','real_B2_records':count,'max_logit_error':max_error,'all_logits_bitwise_equal':True,
            'all_actions_identical':True,'eager_seconds_1000':stamps[0],'cached_seconds_1000':stamps[1],
            'weights_changed':False,'fixed_batch_size':1,'native_second_round_controller':'unchanged eager B2',
            'implementation':'cache fixed question and action keys; original eager state/query arithmetic and softmax, no JIT graph',
            'what_did_we_learn':'Frozen semantic tokens can be cached without changing any actual first-round B2 logit or action; retain eager arithmetic after rejecting tiny JIT numerical differences.'}
    save(REPORTS/'FAST_MATCH_OPERATOR_AUDIT.json',report);protect_anchor();print(json.dumps(report))


if __name__=='__main__':main()
