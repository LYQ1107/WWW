"""Post-training permutation, dynamic options, question calibration and sharing.

Calibration is descriptive on all pre-exposed TRAIN16 natural payloads. No
temperature, threshold or checkpoint selection is fitted on these results.
"""
import argparse
import math
import numpy as np
import torch
from jev_oneweek_common import *
from jev_oneweek_assess import load_model
from jev_oneweek_tests import fixture
from jev_oneweek_learning import batch,load_records


def subset(x,indices):
    out={key:value.clone() for key,value in x.items()}
    for key in ['history_visual','history_mask','identity_meta','identity_mask']:out[key]=out[key][:,indices]
    for key in ['pair_evidence','legal','commitment_features']:out[key]=out[key][:,:,indices]
    return out


def calibration(probability,target):
    p=np.asarray(probability,dtype=np.float64);t=np.asarray(target,dtype=np.float64)
    if not len(t):return dict(known=0,Brier=None,ECE20=None)
    bins=[];ece=0.
    for i in range(20):
        mask=(p>=i/20)&(p<(i+1)/20 if i<19 else p<=1)
        if not mask.any():continue
        pp=float(p[mask].mean());tt=float(t[mask].mean());n=int(mask.sum())
        bins.append(dict(bin=i,count=n,mean_probability=pp,positive_rate=tt));ece+=n/len(t)*abs(pp-tt)
    predicted=p>=.5
    tp=int(((predicted)&(t==1)).sum());fp=int((predicted&(t==0)).sum());fn=int((~predicted&(t==1)).sum())
    return dict(known=len(t),positive=int(t.sum()),Brier=float(((p-t)**2).mean()),ECE20=ece,
                precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None,bins=bins)


def main(seed):
    protect();model,checkpoint,_=load_model('jev',seed,'formal');records=fixture();checks=[]
    for r in records:
        x,_=batch([r]);k=x['identity_mask'].shape[1];permutation=torch.arange(k-1,-1,-1,device='cuda:0')
        with torch.no_grad():a=model.details(x);b=model.details(subset(x,permutation))
        inverse=permutation.argsort();q=len(r['rows'])
        def mapped(plan):return tuple(int(permutation[c]) if c>=0 else -1 for c in plan)
        max_error=max(float((a[key]-b[key][:,:,inverse]).abs().max()) if k else 0. for key in ['cross_logits','stability_logits'])
        chosen_a=a['plans'][0][int(a['plan_scores'][0].argmax())]
        chosen_b=mapped(b['plans'][0][int(b['plan_scores'][0].argmax())])
        dynamic=[]
        for count in sorted(set([0,1,k//2,k])):
            z=subset(x,torch.arange(count,device='cuda:0'))
            with torch.no_grad():d=model.details(z)
            for plan in d['plans'][0]:
                assert len(plan)==q
                assert len([c for c in plan if c>=0])==len({c for c in plan if c>=0})
                assert all(c<0 or bool(z['legal'][0,row,c]) for row,c in enumerate(plan))
            assert all(torch.isfinite(v).all() for v in d['plan_scores'])
            dynamic.append(dict(K=count,valid=True,plan_count=len(d['plans'][0])))
        checks.append(dict(key=r['key'],aux_permutation_max_abs_error=max_error,
                           aux_equivariant_within_2e_5=max_error<=2e-5,
                           proposal_set_equivariant=set(a['plans'][0])=={mapped(p) for p in b['plans'][0]},
                           committed_primary_choice_equivariant=chosen_a==chosen_b,dynamic_K=dynamic))
    x,_=batch(records[:1]);calls=[0];original=model.identity.core.encode_state
    def encoded(value):calls[0]+=1;return original(value)
    model.identity.core.encode_state=encoded
    timings={};outputs={};encodes={}
    with torch.no_grad():
        for arm in ['jev','no_shared']:
            model.arm=arm
            for _ in range(8):model.details(x)
            samples=[];calls[0]=0
            for _ in range(64):
                torch.cuda.synchronize();begin=time.perf_counter();outputs[arm]=model.details(x)
                torch.cuda.synchronize();samples.append((time.perf_counter()-begin)*1000)
            timings[arm]=dict(median_ms=float(np.median(samples)),p95_ms=float(np.quantile(samples,.95)))
            encodes[arm]=calls[0]/64
    model.identity.core.encode_state=original;model.arm='jev'
    equal=all(torch.equal(outputs['jev'][key],outputs['no_shared'][key]) for key in ['base_logits','edge_logits','cross_logits','stability_logits'])
    assert equal and encodes==dict(jev=1.,no_shared=2.)
    train,validation,_=load_records();del train
    start=read(OUT/'training_v1/jev'/f'seed{seed}/formal/START.json');weights=start['pos_weights_TRAIN_only']
    probabilities={n:[] for n in ['cross','stability']};corrected={n:[] for n in probabilities};targets={n:[] for n in probabilities};unknown={n:0 for n in probabilities}
    disagreements=0;total_joint=0;difference=0.;payloads=0
    with torch.no_grad():
        for start_index in range(0,len(validation),4):
            x,y=batch(validation[start_index:start_index+4]);d=model.details(x);payloads+=len(validation[start_index:start_index+4])
            joint=(y['cross']>=0)&(y['stability']>=0)&x['legal']&x['question_mask'][:,:,None]
            total_joint+=int(joint.sum());disagreements+=int(((y['cross']!=y['stability'])&joint).sum())
            difference+=float((d['cross_logits'].sigmoid()-d['stability_logits'].sigmoid()).abs()[joint].sum())
            for name in probabilities:
                valid=(y[name]>=0)&x['legal']&x['question_mask'][:,:,None]
                unknown[name]+=int(((y[name]<0)&x['legal']&x['question_mask'][:,:,None]).sum())
                logit=d[name+'_logits'][valid]
                probabilities[name].extend(logit.sigmoid().cpu().tolist())
                corrected[name].extend((logit-math.log(weights[name])).sigmoid().cpu().tolist())
                targets[name].extend(y[name][valid].cpu().tolist())
    save(REPORTS/f'TRAINED_CONTRACT_CALIBRATION_SEED{seed}.json',dict(status='COMPLETE',binding=binding('Post-training real TRAIN inputs and natural TRAIN16 calibration'),
         checkpoint=checkpoint,permutation_and_dynamic_candidates=checks,
         actual_shared_state_encoder_calls_per_event=encodes,same_weights_shared_and_recomputed_bitwise_equal=equal,
         single_payload_model_timing=timings,timing_scope='64 synchronised real TRAIN12 payload runs including plan generation; possibly contended GPU, diagnostic only, not primary full FPS',
         calibration={name:dict(raw_weighted_sigmoid=calibration(probabilities[name],targets[name]),
                              TRAIN_class_weight_log_odds_corrected=calibration(corrected[name],targets[name]),
                              UNKNOWN_edges=unknown[name],training_pos_weight=weights[name]) for name in probabilities},
         TRAIN16_payloads=payloads,joint_known_edges=total_joint,joint_target_disagreements=disagreements,
         mean_question_probability_difference_on_joint_known=difference/total_joint if total_joint else None,
         calibration_fitted_parameters=0,independent_generalization=False,
         weighted_sigmoid_is_not_assumed_calibrated_probability=True,
         distinct_architectural_heads_do_not_prove_distinct_semantic_value=True))
    print('TRAINED_CONTRACT_COMPLETE',seed,checks,encodes,'TARGET_DISAGREEMENTS',disagreements,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,default=20261009);a=parser.parse_args()
    try:main(a.seed)
    except Exception:
        import traceback
        failure('trained_contract',traceback.format_exc());raise
