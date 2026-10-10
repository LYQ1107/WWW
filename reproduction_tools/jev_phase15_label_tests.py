"""Certificates distinguish concrete continuation, correction and UNKNOWN."""
import collections
import torch
from types import SimpleNamespace
from jev_phase15_common import *
from jev_phase15_commitment_labels import certify
from jev_phase15_losses import mass_loss


def main():
    risk=SimpleNamespace(labels=SimpleNamespace(video=12,current=lambda f,v:[1,2]),
        votes={10:collections.Counter({1:5}),20:collections.Counter({2:5}),30:collections.Counter({1:2,2:2})},
        observations={10:5,20:5,30:4},latest={(1,0):(4,10),(2,0):(4,20)})
    x={'legal':torch.ones(1,2,3,dtype=torch.bool),'commitment_features':torch.zeros(1,2,3,20)}
    x['commitment_features'][0,:,0,0]=.8
    y=certify(x,[10,20,30],(5,0),risk)
    assert y['commit_kind'].tolist()==[1,2]
    assert y['commit_positive'].tolist()==[[True,False,False,False],[False,True,False,False]]
    assert y['trust'].tolist()==[1,1,0]
    assert y['known_options'][:,2].eq(False).all()
    assert y['safety'][:,2].eq(-1).all(), 'pollution is not a certified wrong person'
    z=torch.randn(2,4,requires_grad=True)
    loss=mass_loss(z,y['commit_positive'],y['known_options'],y['commit_kind']>0);loss.backward()
    assert z.grad[:,2].eq(0).all(), 'UNKNOWN received a false-negative gradient'
    x['commitment_features'][:]=0
    assert certify(x,[10,20,30],(5,0),risk)['commit_kind'].tolist()==[0,0]
    risk.labels.video=17
    try:certify(x,[10,20,30],(5,0),risk)
    except AssertionError:pass
    else:raise AssertionError('DEVELOPMENT accepted by training label function')
    save(OUT/'label_contract_v1/RESULT.json',dict(status='PASS',binding=binding(seed=20261009),
        safe_continuation=True, necessary_correction=True, mixture_not_wrong_person=True,
        ambiguous_commitment_UNKNOWN=True, unknown_mass_loss_gradient_zero=True,development_refused=True))
    print('PHASE15_LABEL_CONTRACT_PASS',flush=True)


if __name__=='__main__':main()
