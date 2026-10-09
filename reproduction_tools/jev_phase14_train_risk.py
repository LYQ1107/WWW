"""Matched TRAIN prefixes under each frozen20k/24k native policy, without training."""
import argparse,gzip,time,torch
from jev_phase14_common import *
from jev_phase13_runtime import build_tracker,cache_inputs,run
from jev_phase14_online import load_policy
from jev_phase14_native_risk import NativeRisk
from gtr.modeling.jev_phase14.memory import CachedIdentityMemory

def main(variant,seed,phase):
    protect();source=binding();torch.set_num_threads(1);torch.manual_seed(20261009)
    root=OUT/'matched_TRAIN_native_risk_v1'/phase/f'{variant}_seed{seed}';root.mkdir(parents=True,exist_ok=True)
    policy,trained,_=load_policy(variant,seed,phase);results=[]
    for video in TRAIN:
        out=root/f'video{video:02d}';out.mkdir(exist_ok=True)
        if (out/'RESULT.json').exists():results.append(read(out/'RESULT.json'));continue
        values,frames,reader=cache_inputs(video);risk=NativeRisk(video,reader);model=build_tracker(video,policy=policy);ex=model.jev_stage2_executor;ex.memory_factory=CachedIdentityMemory
        ex.observer=risk.before;ex.commit_observer=risk.after;begin=time.monotonic()
        with torch.no_grad():raw,_=run(model,values,frames,stop=255)
        assert risk.counts['payloads']==511
        trace=out/'COMMIT_OFFLINE_AUDIT.jsonl.gz'
        with gzip.open(trace,'wt') as h:
            for event in risk.trace:h.write(json.dumps(event)+'\n')
        result=dict(status='COMPLETE',binding=source,phase=phase,variant=variant,seed=seed,video=video,trained=trained,frames=256,actual_mutated_state_online=True,
            GT_actor_inputs=False,audit=risk.summary(),trace=dict(path=str(trace),SHA256=sha(trace)),seconds=time.monotonic()-begin)
        save(out/'RESULT.json',result);results.append(result);print('PHASE14_MATCHED_TRAIN_NATIVE_RISK',phase,variant,seed,video,result['audit'],flush=True)
        del model,raw,risk;torch.cuda.empty_cache()
    save(root/'RESULT.json',dict(status='COMPLETE',binding=source,phase=phase,variant=variant,seed=seed,results=results))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--variant',required=True);a.add_argument('--seed',type=int,required=True);a.add_argument('--phase',required=True,choices=['formal','onpolicy','oldloss_onpolicy','offpolicy']);v=a.parse_args();main(v.variant,v.seed,v.phase)
