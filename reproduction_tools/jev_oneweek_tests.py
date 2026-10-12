"""Actual TRAIN-input legality, new initialization, leakage and native state tests."""
import gc
import traceback
import torch
from jev_oneweek_common import *
from jev_phase14_artifacts import load_dense
from jev_oneweek_learning import initialize,batch
from gtr.modeling.jev_oneweek.model import ARMS
from gtr.modeling.jev_oneweek.plans import plans
from gtr.modeling.jev_stage2.assignment import lawful_choice


def fixture():
    metadata=read(REPORTS/'SUPERVISION_AUDIT.json')
    labels={tuple(r['key']):r for r in load_dense(metadata['labels']['path'])['labels']}
    source=read(PHASE15/'commitment_dataset_v2/video12/RESULT.json')['DATASET']
    records=load_dense(source['path'])['records']
    choices=[r for r in records if r['key'][1]>=8 and not r['audit_block'] and r['supervised'].any()]
    return [dict(choices[i],oneweek_labels=labels[tuple(choices[i]['key'])]) for i in [0,len(choices)//2,len(choices)-1]]


def main():
    protect();seed_all();records=fixture();x,y=batch(records)
    results=[];counts={}
    for arm in ARMS:
        model,checkpoint=initialize(arm,20261009);model.eval()
        with torch.no_grad():
            details=model.details(x)
            expected=model.identity({k:v for k,v in x.items() if k!='commitment_features'})
        assert torch.equal(details['base_logits'],expected)
        assert torch.equal(details['edge_logits'],expected)
        for index,r in enumerate(records):
            q=len(r['rows']);k=len(r['refs'])
            score=torch.cat([expected[index,:q,:k],expected[index,:q,-1:]],-1)
            assert details['plans'][index][int(details['plan_scores'][index].argmax())]==tuple(lawful_choice(score,x['legal'][index,:q,:k]))
        count=sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert count<=500000;counts[arm]=count
        for key in ['cross_logits','stability_logits']:
            assert torch.isfinite(details[key]).all()
        results.append(dict(arm=arm,initial_logits_bitwise_exact=True,initial_native_choices_exact=True,new_trainable_parameters=count))
        del model,details,expected;gc.collect();torch.cuda.empty_cache()
    model,_=initialize('jev',20261009);model.eval()
    local=batch(records[:1])[0];k=local['identity_mask'].shape[1];perm=torch.arange(k-1,-1,-1,device='cuda:0')
    changed={key:value.clone() for key,value in local.items()}
    for key in ['history_visual','history_mask','identity_meta','identity_mask']:changed[key]=local[key][:,perm]
    for key in ['pair_evidence','legal','commitment_features']:changed[key]=local[key][:,:,perm]
    with torch.no_grad():original=model.details(local);permuted=model.details(changed)
    restored=permuted['cross_logits'][:,:,perm]
    assert torch.allclose(original['cross_logits'],restored,atol=2e-5,rtol=1e-5)
    assert torch.allclose(original['stability_logits'],permuted['stability_logits'][:,:,perm],atol=2e-5,rtol=1e-5)
    actor=dict(local,GT_LABELS_OFFLINE_ONLY=torch.zeros(1,device='cuda:0'))
    try:model(actor)
    except ValueError:leak_rejected=True
    else:raise AssertionError('GT field admitted')
    # Real zero-history payload: all current detections must retain private DEFER.
    empty={key:value.clone() for key,value in local.items()}
    for key in ['history_visual','history_mask','identity_meta','identity_mask']:empty[key]=local[key][:,:0]
    for key in ['pair_evidence','legal','commitment_features']:empty[key]=local[key][:,:,:0]
    with torch.no_grad():d=model.details(empty)
    assert all(col<0 for col in d['plans'][0][0])
    # Same opaque ID may appear once in each distinct camera event. The solver
    # is intentionally called separately, not with a global cross-view capacity.
    logits=torch.tensor([[2.,0.]],device='cuda:0');legal=torch.ones(1,1,dtype=torch.bool,device='cuda:0')
    assert lawful_choice(logits,legal)==[0] and lawful_choice(logits,legal)==[0]
    # Native same-state initialization parity over real committed observations.
    from jev_phase13_runtime import build_tracker,cache_inputs,run
    from jev_phase15_common import load_old_policy
    from gtr.modeling.jev_phase15.native_commit_adapter import attach,FrozenOriginalPolicy
    from gtr.modeling.jev_oneweek.native import attach_jevtrack
    from gtr.modeling.jev_native_state import fingerprint
    from audit_jev_stage2_gta_free import phase13_prefix
    values,frames,_=cache_inputs(12);states=[];outputs=[];events=[]
    for new in [False,True]:
        seed_all()
        if new:policy,_=initialize('jev',20261009)
        else:policy,_,_=load_old_policy('multi_question',20261009);policy=FrozenOriginalPolicy(policy)
        tracker=build_tracker(12,policy,react_learned=False)
        ex=attach_jevtrack(tracker) if new else attach(tracker)
        committed=[];last=[None]
        def observed(**d):
            committed.append((d['frame'],d['view'],d['instances'][-1].track_ids.cpu().tolist(),d['events']))
            fields={key:value for key,value in d.items() if key!='events'}
            last[0]=phase13_prefix(tracker,fields)
        ex.commit_observer=observed;policy.eval() if new else None
        seed_all()
        with torch.no_grad():raw,_=run(tracker,values,frames,stop=95)
        states.append(fingerprint(last[0]));outputs.append(fingerprint(raw[:192]));events.append(committed)
        del raw,tracker,ex,policy;gc.collect();torch.cuda.empty_cache()
    assert states[0]==states[1] and outputs[0]==outputs[1] and events[0]==events[1]
    result=dict(status='PASS',binding=binding('Actual TRAIN12 tensors and192-camera-payload native initializer parity'),
                initializers=results,new_parameters=counts,candidate_permutation_aux_outputs=True,
                dynamic_zero_K_private_DEFER=True,GT_key_rejected=leak_rejected,
                same_Global_ID_allowed_across_two_camera_events=True,
                native_payloads_compared=len(events[0]),every_native_commit_and_event_exact=True,
                full_native_final_state_exact=True,raw_Instance_fields_exact=True,
                baseline='Frozen Phase XIV Multi + Phase XV interface with new module OFF; not original separate-front GMT',
                native_final_state_sha=states[0],raw_prefix_sha=outputs[0])
    save(REPORTS/'MODEL_AND_NATIVE_CONTRACT_TESTS.json',result)
    registry=read(REPORTS/'EXPERIMENT_REGISTRY.json');registry['jobs']['initial_parity'].update(status='COMPLETE',results=ref(REPORTS/'MODEL_AND_NATIVE_CONTRACT_TESTS.json'));save(REPORTS/'EXPERIMENT_REGISTRY.json',registry)
    print('ONEWEEK_NATIVE_AND_MODEL_TESTS_PASS',counts,flush=True)


if __name__=='__main__':
    try:main()
    except Exception:
        failure('contract_tests',traceback.format_exc());raise
