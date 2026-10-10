"""Recent contamination certificates cannot arise from mixtures or gaps alone."""
import torch
from jev_phase15_common import *
from jev_phase15_recent_labels import recent_owner,patch_record


def main():
    assert recent_owner([(1,2),(2,2),(3,2)],4)==2
    for segment in [[(1,2),(2,None),(3,2)],[(1,2),(2,1),(3,2)],[(0,2),(2,2),(3,2)]]:
        assert recent_owner(segment,4) is None
    assert recent_owner([(1,2),(2,2),(3,2)],6) is None
    r=dict(key=(12,4,0),rows=[0],refs=[10,20],GT_labels_OFFLINE_ONLY=[1],
        positive=torch.tensor([[False,True,False]]),known_options=torch.tensor([[False,True,True]]),
        commit_positive=torch.zeros(1,3,dtype=torch.bool),commit_kind=torch.zeros(1,dtype=torch.int8),safety=torch.tensor([[-1,1]]),
        inputs=dict(legal=torch.ones(1,1,2,dtype=torch.bool),commitment_features=torch.zeros(1,1,2,20)))
    r['inputs']['commitment_features'][0,0,0,0]=.8
    patch,_=patch_record(r,{(10,0):[(1,2),(2,2),(3,2)]})
    assert patch['commit_kind'].tolist()==[2] and patch['safety'].tolist()==[[0,1]]
    assert patch['commit_known_options'][0,0] and not r['known_options'][0,0]
    assert not r['commit_positive'].any() and r['commit_kind'].tolist()==[0], 'original label overwritten'
    for segment in [[(1,1),(2,1),(3,1)],[(1,2),(2,None),(3,2)]]:
        assert not patch_record(r,{(10,0):segment})[0]
    save(OUT/'recent_label_contract_v2/RESULT.json',dict(status='PASS',binding=binding(seed=20261009),
        past_only=True,UNKNOWN_gap_not_negative=True,mixture_alone_not_negative=True,WHO_label_unchanged=True,
        immutable_original=True,recent_wrong_owner_correction=True))
    print('PHASE15_RECENT_LABEL_TEST_PASS',flush=True)


if __name__=='__main__':main()
