"""Bounded history/opaque-candidate contracts on the actual harmful prefix."""
from jev_phase16_common import *
import torch
from jev_phase14_artifacts import load_dense
from jev_phase15_pilot_native import converted_prefix
from gtr.modeling.jev_phase15.native_commit_adapter import CommitmentMemory
from gtr.modeling.jev_phase16.multi_prototype_memory import MultiPrototypeMemory,VARIANTS
from gtr.modeling.jev_native_state import fingerprint


def main():
    protect();torch.set_num_threads(1)
    entry=read(XV/'train_commitment_prefixes_v1/video12/RESULT.json')['counterfactual_prefixes'][0]
    assert entry['key']==[12,3,1] and entry['row']==6
    prefix=load_dense(entry['prefix']['path'],map_location='cpu')
    assert fingerprint(prefix)==entry['starting_state_SHA256']
    raw_hash=fingerprint(prefix['galleries']);memory=CommitmentMemory()
    memory.load_state_dict(converted_prefix(prefix)['phase13_identity_meta'])
    refs=sorted(prefix['galleries']);frame,view=prefix['key'][1:]
    current=prefix['instances'][-1];x=memory.build(current,refs,prefix['galleries'],frame,view,set(refs))
    original_hash=fingerprint(x);permutation=torch.arange(len(refs)-1,-1,-1)
    permuted=dict(x)
    for k in ['history_visual','history_mask','identity_meta','identity_mask']:permuted[k]=x[k][:,permutation]
    for k in ['pair_evidence','legal','commitment_features']:permuted[k]=x[k][:,:,permutation]
    cases=[]
    for variant in VARIANTS[1:]:
        p=MultiPrototypeMemory.from_prefix(variant,prefix);state=p.state_dict()
        restored=MultiPrototypeMemory(variant);restored.load_state_dict(state)
        assert fingerprint(state)==fingerprint(restored.state_dict())
        assert set(p.identities)==set(refs) and max(p.stored_vectors(i) for i in refs)<=16
        a,provenance=p.batch(x,refs,view);b,_=p.batch(permuted,[refs[j] for j in permutation],view)
        assert torch.equal(a['history_visual'][:,permutation],b['history_visual'])
        assert torch.equal(a['pair_evidence'][:,:,permutation],b['pair_evidence'])
        assert a['history_visual'].shape==x['history_visual'].shape
        assert all(torch.equal(a[k],x[k]) for k in x if k not in ['history_visual','history_mask','pair_evidence'])
        assert torch.equal(a['pair_evidence'][...,4:],x['pair_evidence'][...,4:])
        assert fingerprint(x)==original_hash and fingerprint(prefix['galleries'])==raw_hash
        first=next(iter(restored.identities));r=restored.identities[first]
        if 'recent' in r:r['recent'][0]['sum'].zero_()
        elif 'temporal' in r:r['temporal'].items[0]['sum'].zero_()
        else:next(iter(r['camera'].values())).items[0]['sum'].zero_()
        assert fingerprint(p.state_dict())==fingerprint(state)
        assert all(key[0:2]<(frame,view) or key[0]==0 for values in provenance for item in values if item for key in item['keys'])
        cases.append(dict(variant=variant,identities=len(refs),max_stored_vectors=max(p.stored_vectors(i) for i in refs),
            state_SHA256=fingerprint(state),batch_SHA256=fingerprint(a),deep_restore_no_alias=True,
            candidate_permutation_equivariance=True,native_causal_fields_unchanged=True,raw_Gallery_unchanged=True))
    result=dict(status='PASS',binding=binding(inputs=[entry['prefix']],native_state=entry['starting_state_SHA256'],
        evaluator='actual harmful prefix tensor, candidate-permutation and independent snapshot contracts',
        scope='engineering qualification only; no native recovery or safety claim'),cases=cases)
    save(OUT/'P1_contract_replay/RESULT.json',result)
    save(REPORTS/'BOUNDED_HISTORY_CONTRACT.json',dict(status='PASS',result=ref(OUT/'P1_contract_replay/RESULT.json'),
        original_inline_engineering_contract_retained=ref(OUT/'P1_contract/RESULT.json'),cases=cases))
    print('PHASE16_BOUNDED_HISTORY_ACTUAL_CONTRACT_PASS',len(cases),flush=True)


if __name__=='__main__':main()
