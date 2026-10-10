"""Native unconditional parity plus admission snapshot contracts, no benefit gate."""
from jev_phase16_common import *
import copy
import gzip
import torch
from jev_phase14_artifacts import load_dense
from jev_phase15_pilot_native import converted_prefix
from jev_phase13_runtime import build_tracker, cache_inputs, run
from gtr.modeling.jev_phase15.native_commit_adapter import attach as attach_xv
from gtr.modeling.jev_phase16.native_memory_commit_adapter import attach
from gtr.modeling.jev_phase16.safe_memory_admission import RULES
from gtr.modeling.jev_phase16.safe_matching_memory import SafeMatchingMemory
from gtr.modeling.jev_native_state import fingerprint


def main():
    protect();torch.set_num_threads(1)
    entry=read(XV/'train_commitment_prefixes_v1/video12/RESULT.json')['counterfactual_prefixes'][0]
    assert entry['key']==[12,3,1] and entry['row']==6
    prefix=load_dense(entry['prefix']['path'],map_location='cuda:0')
    assert fingerprint(prefix)==entry['starting_state_SHA256']
    prefix=converted_prefix(prefix);initial=fingerprint(prefix)
    values,frames,reader=cache_inputs(12);policy,ck,training=load_policy('v3')
    outcomes=[];baseline=None;aids=None
    for rule in ['XV_ORIGINAL','always_write','bounded_quarantine']:
        model=build_tracker(12,policy,react_learned=False)
        executor=attach_xv(model) if rule=='XV_ORIGINAL' else attach(model,rule,prefix)
        trace=[]
        def after(**d):
            trace.append(dict(key=[d['frame'],d['view']],ids=d['instances'][-1].track_ids.tolist()))
        executor.commit_observer=after
        with torch.no_grad():raw,_=run(model,values,frames,stop=6,prefix=copy.deepcopy(prefix))
        memory=executor.memory.state_dict()
        if rule=='XV_ORIGINAL':baseline=fingerprint(memory);aids=fingerprint(trace)
        elif rule=='always_write':
            assert baseline==fingerprint(memory['raw'])==fingerprint(memory['trusted'])
            assert aids==fingerprint(trace)
        if rule!='XV_ORIGINAL':
            restored=SafeMatchingMemory(rule);restored.load_state_dict(memory)
            assert fingerprint(memory)==fingerprint(restored.state_dict())
            if restored.admission.records:
                next(iter(restored.admission.records.values()))[0]['status']='TEST_MUTATION'
                assert fingerprint(executor.memory.state_dict())==fingerprint(memory)
        outcomes.append(dict(rule=rule,committed_IDs_SHA256=fingerprint(trace),memory_SHA256=fingerprint(memory),
            unconditional_parity=rule!='bounded_quarantine',restore_no_alias=rule!='XV_ORIGINAL'))
        del model,executor;torch.cuda.empty_cache()
    assert fingerprint(prefix)==initial
    save(OUT/'P2_contract/RESULT.json',dict(status='PASS',binding=binding(checkpoints=[ck],inputs=[entry['prefix'],ref(REPORTS/'P2_P3_PROTOCOL.json')],
        native_state=initial,evaluator='actual earliest harmful prefix native parity and independent state snapshots',scope='engineering only; safe benefit not established'),
        cases=outcomes,original_prefix_unchanged=True,earliest_harmful_prefix_retained=True))
    save(REPORTS/'NATIVE_MEMORY_PARITY.json',dict(status='PASS_EARLIEST_NATIVE_CONTRACT',result=ref(OUT/'P2_contract/RESULT.json'),cases=outcomes,
        broader_all24_parity='PENDING factorial A vs frozen XV',bounded_quarantine_parity_not_expected=True))
    print('PHASE16_SAFE_MEMORY_NATIVE_CONTRACT_PASS',flush=True)


if __name__=='__main__':main()
