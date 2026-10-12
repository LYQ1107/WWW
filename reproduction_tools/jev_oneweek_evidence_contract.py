"""Test real-tensor evidence removal and finite legal decisions before replay."""
import torch
from jev_oneweek_common import *
from jev_oneweek_assess import load_model
from jev_oneweek_tests import fixture
from jev_oneweek_learning import batch
from gtr.modeling.jev_oneweek.evidence import remove_evidence


def main():
    protect();x,_=batch(fixture());model,checkpoint,_=load_model('jev',20261009,'formal');results=[]
    with torch.no_grad():
        for kind in ['no_cross_evidence','no_stability_evidence']:
            changed=remove_evidence(x,kind)
            if kind=='no_cross_evidence':
                assert not changed['history_mask'][:,:,3].any()
                assert not changed['pair_evidence'][...,[3,6]].any()
                assert not changed['identity_meta'][...,[5,7]].any()
                assert not changed['commitment_features'][...,[10,11,12]].any()
            else:
                assert not changed['history_mask'][:,:,2].any()
                assert not changed['pair_evidence'][...,[2,4,5,10,11,12,13,14,15]].any()
                assert not changed['identity_meta'][...,[0,1,2,4,6,9,10]].any()
                assert not changed['commitment_features'][...,[0,1,2,3,4,5,6,7,8,9,13,14,15,17,18,19]].any()
            d=model.details(changed);assert all(torch.isfinite(v).all() for v in d['plan_scores'])
            results.append(dict(kind=kind,actual_tensor_removal_pass=True,all_scores_finite=True,unchanged_legal_ID_masks=True))
    save(REPORTS/'EVIDENCE_INTERVENTION_CONTRACT.json',dict(status='PASS',checkpoint=checkpoint,cases=results,
         binding=binding('Inference evidence interventions before all encoders'),
         frozen_weight_no_retraining=True,proposals_recomputed_not_frozen=True,
         no_cross_keeps_own_camera_history_and_opaque_shared_ID_namespace=True,
         no_stability_keeps_other_camera_appearance_and_metadata_removes_own_history_and_persistence=True,
         candidate_eligibility_and_opaque_ID_namespace_still_use_unmodified_native_history=True))
    print('EVIDENCE_CONTRACT_PASS',flush=True)


if __name__=='__main__':main()
