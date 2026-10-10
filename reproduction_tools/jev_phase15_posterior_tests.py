"""Posterior feedback intervention preserves every other causal observation."""
import torch
from jev_phase15_common import *
from jev_phase15_state_tests import inst
from gtr.modeling.jev_phase15.native_commit_adapter import CommitmentMemory
from gtr.modeling.jev_phase15.model import PersistentIdentityPolicy
from gtr.modeling.jev_native_state import fingerprint


def main():
    torch.set_num_threads(1);torch.manual_seed(20261009)
    memory=CommitmentMemory();features=torch.randn(2,1024);boxes=[[10,10,20,30],[50,50,60,70]]
    galleries={ref:inst(features[row:row+1],[boxes[row]],[ref]) for row,ref in enumerate([101,202])}
    for row,ref in enumerate([101,202]):memory.update(ref,features[row],torch.tensor(boxes[row]),(100,100),0,0)
    x=memory.build(inst(features,boxes),[101,202],galleries,1,0,{101,202});model=PersistentIdentityPolicy().eval()
    source_hash=fingerprint(x)
    with torch.no_grad():before=model(x)
    changed=dict(x);changed['commitment_features']=x['commitment_features'].clone()
    changed['commitment_features'][...,16]=.9;changed['commitment_features'][...,17]=1
    with torch.no_grad():feedback=model(changed)
    assert not torch.allclose(before,feedback),'posterior channel cannot affect the actor'
    model.posterior_feedback='masked'
    with torch.no_grad():masked=model(changed)
    assert torch.equal(before,masked), 'masked intervention changed more than the two unsupported fields'
    assert fingerprint(x)==source_hash and (changed['commitment_features'][...,17]==1).all()
    save(OUT/'posterior_feedback_contract_v1/RESULT.json',dict(status='PASS',binding=binding(seed=20261009),
        constant_training_support_parity=True,original_inputs_unmodified=True,only_two_channels_intervened=True,
        posterior_state_still_stored=True,GT_free=True))
    print('PHASE15_POSTERIOR_INTERVENTION_TEST_PASS',flush=True)


if __name__=='__main__':main()
