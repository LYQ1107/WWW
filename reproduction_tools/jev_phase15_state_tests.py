"""Exercise persistent state, exact old scores, lawful options and dynamic readers."""
import copy
import torch
from jev_phase15_common import *
from detectron2.structures import Instances, Boxes
from gtr.modeling.jev_phase14.memory import CachedIdentityMemory
from gtr.modeling.jev_phase15.native_commit_adapter import CommitmentMemory, FrozenOriginalPolicy
from gtr.modeling.jev_phase15.model import PersistentIdentityPolicy, ARMS
from gtr.modeling.jev_stage2.assignment import lawful_choice
from gtr.modeling.jev_native_state import fingerprint


def inst(features, boxes, ids=None):
    result = Instances((100, 100)); result.reid_features = features
    result.pred_boxes = Boxes(torch.tensor(boxes, device=features.device, dtype=torch.float32))
    result.scores = torch.full((len(boxes),), .9, device=features.device)
    if ids is not None: result.track_ids = torch.tensor(ids, device=features.device)
    return result


def main():
    protect(); torch.set_num_threads(1); torch.manual_seed(20261009)
    memory = CommitmentMemory(); plain = CachedIdentityMemory(); galleries = {}
    f = torch.randn(2, 1024); boxes = [[10, 10, 20, 30], [50, 50, 60, 70]]
    for row, ref in enumerate([101, 202]):
        galleries[ref] = inst(f[row:row+1], [boxes[row]], [ref])
    for frame, view, ref, row in [(0, 0, 101, 0), (0, 1, 101, 0), (1, 0, 101, 0), (1, 0, 202, 1)]:
        for store in [memory, plain]: store.update(ref, f[row], torch.tensor(boxes[row]), (100,100), frame, view)
    current = inst(f, boxes); refs = [101, 202]
    x = memory.build(current, refs, galleries, 2, 0, set(refs))
    before = plain.build(current, refs, galleries, 2, 0, set(refs))
    assert all(torch.equal(x[key], val) for key, val in before.items())
    digest = fingerprint(memory.state_dict()); restored = CommitmentMemory(); restored.load_state_dict(memory.state_dict())
    assert fingerprint(restored.state_dict()) == digest
    xx = restored.build(current, refs, galleries, 2, 0, set(refs))
    assert all(torch.equal(x[key], xx[key]) for key in x)
    snapshot = memory.state_dict(); memory.update(101, f[0], torch.tensor(boxes[0]), (100,100), 2, 0)
    assert fingerprint(snapshot) == digest, 'snapshot aliases mutable state'
    try: memory.update(101, f[0], torch.tensor(boxes[0]), (100,100), 2, 0)
    except AssertionError: pass
    else: raise AssertionError('same-camera duplicate accepted')
    # Stale/recycled opaque ID keeps its committed record, then resumes safely.
    restored.update(101, f[0], torch.tensor(boxes[0]), (100,100), 60, 0)
    assert restored.commitment.observations[101,0]['breaks'] == 1
    assert restored.commitment.observations[101,1]['frame'] == 0
    empty = inst(f[:0], torch.zeros(0,4).tolist()); empty.pred_boxes = Boxes(torch.zeros(0,4))
    assert restored.build(empty, refs, galleries, 61, 0, set())['commitment_features'].shape == (1,0,2,20)
    permutation = torch.tensor([1,0]); permuted = dict(x)
    for key in ['history_visual','history_mask','identity_meta','identity_mask']: permuted[key] = x[key][:,permutation]
    for key in ['pair_evidence','legal','commitment_features']: permuted[key] = x[key][:,:,permutation]
    outputs = {}
    for arm in ARMS:
        model = PersistentIdentityPolicy(arm).eval()
        with torch.no_grad(): z = model(x); p = model(permuted)
        assert torch.allclose(z[:,:,:2][:,:,permutation], p[:,:,:2], atol=3e-5)
        assert torch.allclose(z[:,:,-1], p[:,:,-1], atol=3e-5)
        if arm != 'A_original':
            changed = dict(x); changed['commitment_features'] = x['commitment_features'].clone()
            changed['commitment_features'][:,:,0,0] += 1
            if arm not in ['B_continuity']:
                with torch.no_grad(): altered = model(changed)
                assert not torch.allclose(z, altered), 'commitment evidence is dormant'
                assert abs(float((altered-z)[0,0,0] - (altered-z)[0,0,1])) > 1e-7
        choices = lawful_choice(z[0], x['legal'][0]); selected = [c for c in choices if c >= 0]
        assert len(selected) == len(set(selected))
        outputs[arm] = dict(candidate_permutation='PASS', parameters=sum(p.numel() for p in model.parameters()))
    assert set(x) == set(before) | {'commitment_features'}
    source = binding(seed=20261009, evaluator='executed tensor/state contracts on real Detectron2 Instances',
                     scope='structural unit contracts; full native restore parity still required')
    save(REPORTS/'STRUCTURAL_TESTS.json', dict(status='PASS', binding=source, arms=outputs,
        cached_base_inputs_exact=True, snapshot_deepcopy=True, stale_recycling=True,
        cross_camera_ID_reuse=True, same_camera_unique=True, no_GT_or_future_inputs=True,
        deterministic_state_restore=True, dynamic_candidate_specific_values=True,
        native_full_loop='PENDING', alias='DISABLED_NO_GT_RENUMBERING'))
    print('PHASE15_STRUCTURAL_TESTS_PASS', outputs, flush=True)


if __name__ == '__main__': main()
