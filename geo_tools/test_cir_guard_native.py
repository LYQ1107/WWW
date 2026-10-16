"""Native guard, genuine evidence/solver/Instances and faulted commit recovery."""
import random
from types import SimpleNamespace,MethodType
from unittest.mock import patch
import unittest
import numpy as np
import torch
import common
from detectron2.structures import Instances,Boxes
from gtr.modeling.meta_arch import gtr_rcnn as native
from gtr.modeling.identity_revision.controller import IdentityRevisionController
from gtr.modeling.identity_revision.state_transaction import fingerprint
from test_cir_native_commit import model


def observation(frame,camera,identity=None):
    inst=Instances((32,32));inst.pred_boxes=Boxes(torch.tensor([[0.,0.,10.,10.]]))
    inst.reid_features=torch.zeros(1,1152);inst.reid_features[0,0 if camera==1 else 1]=1
    inst.scores=torch.ones(1);inst.pred_classes=torch.zeros(1,dtype=torch.long)
    if identity is not None:inst.track_ids=torch.tensor([identity])
    return inst,dict(video_id=4,frame_id=frame,view_id=camera,image_id=frame*10+camera)


def fixture(mode='guard'):
    m=model(True);c=m.cir_controller;c.mode=mode;c.reset()
    refs=[];counts={1:0,2:0};reids={}
    for frame in (1,2,3):
        group=[observation(frame,cam,cam) for cam in (1,2)]
        c.observe([x[0] for x in group],[x[1] for x in group])
        for inst,record in group:
            identity=record['view_id'];refs.append(inst);counts[identity]+=1
            reids[identity]=Instances.cat([reids[identity],inst]) if identity in reids else inst
            c.committed(inst,counts,reids)
        c.published()
    q,record=observation(4,1);c.observe([q],[record]);instances=refs+[q]
    owner=SimpleNamespace(model=m,instances=instances,counts=counts,reids=reids,id_count=2)
    m.cir_owner=owner
    # Tests the guard on an already computed input matrix; C1 formula has its
    # own independent tests. No idealized candidate or GT label is introduced.
    m.propose_identity_revision=lambda scores,*args:scores
    native.poss_ids.poss_ids=set();native.old_ids.old_ids=set();native.old_reids.old_reids=[]
    score=torch.tensor([[.2,.61/3,.2,.61/3,.2,.61/3]])
    return m,owner,score


def run(m,o,score):
    return native.GTRRCNN.run_global_tracker_plus(m,2,o.instances,[score],None,6,2,o.counts,o.reids,[],0)


class NativeGuardTests(unittest.TestCase):
    def test_real_conflict_veto_reuses_native_joint_solver_and_gallery(self):
        m,o,s=fixture();real=native.linear_sum_assignment
        with patch.object(native,'linear_sum_assignment',wraps=real) as solver:result=run(m,o,s)
        self.assertEqual(solver.call_count,2)
        self.assertEqual(result[0][-1].track_ids.tolist(),[1])
        self.assertEqual(o.counts,{1:4,2:3});self.assertEqual(len(o.reids[1]),4)
        self.assertEqual(m.cir_controller.veto_edges,1)
        self.assertEqual(m.cir_controller.guard_joint_solves,1)
        self.assertTrue(m.cir_controller.ledger.validate_invariants())
    def test_missing_local_chain_keeps_native_accept(self):
        m,o,s=fixture();o.instances[-1].pred_boxes=Boxes(torch.tensor([[20.,20.,30.,30.]]))
        result=run(m,o,s)
        self.assertEqual(result[0][-1].track_ids.tolist(),[2]);self.assertEqual(m.cir_controller.veto_edges,0)
    def test_non_target_query_cannot_be_vetoed(self):
        m,o,s=fixture();m.cir_controller.guard_target_sources={(4,999,0)}
        self.assertEqual(run(m,o,s)[0][-1].track_ids.tolist(),[2])
    def test_second_validation_budget_exhaustion_keeps_pre_guard_result(self):
        m,o,s=fixture()
        # An adversarial proposal provider that never validates must not cause
        # an unbounded solve loop or mutate state before the base decision.
        m.cir_controller.precommit=lambda scores,*args,**kwargs:scores.clone()
        real=native.linear_sum_assignment
        with patch.object(native,'linear_sum_assignment',wraps=real) as solver:result=run(m,o,s)
        self.assertEqual(solver.call_count,3)
        self.assertEqual(result[0][-1].track_ids.tolist(),[2])
        self.assertEqual(m.cir_controller.guard_joint_solves,2)
        self.assertEqual(m.cir_controller.guard_aborted,1)
        self.assertFalse(hasattr(o,'cir_window'))
    def test_fault_restores_native_roots_bank_rng_and_commits_original_once(self):
        m,o,s=fixture();controller=m.cir_controller
        before_rng=fingerprint((random.getstate(),np.random.get_state(),torch.get_rng_state()))
        original=m.commit_native_identity_state;attempts=[]
        def commit(*args):
            result=original(*args);attempts.append(1)
            if len(attempts)==1:
                random.random();np.random.random();torch.rand(3)
                native.poss_ids.poss_ids.add(999);native.old_ids.old_ids.add(998)
                native.old_reids.old_reids.append('faulted bank')
                raise RuntimeError('injected after real dictionary/Gallery commit')
            return result
        m.commit_native_identity_state=commit
        result=run(m,o,s)
        self.assertEqual(len(attempts),2);self.assertIs(m.cir_controller,controller)
        self.assertTrue(controller.failed);self.assertEqual(len(controller.failure_events),1)
        self.assertEqual(result[0][-1].track_ids.tolist(),[2])
        self.assertEqual(o.counts,{1:3,2:4});self.assertEqual(len(o.reids[1]),3);self.assertEqual(len(o.reids[2]),4)
        self.assertIs(o.cir_window[-1],o.instances[-1])
        self.assertEqual(native.poss_ids.poss_ids,set());self.assertEqual(native.old_ids.old_ids,set())
        self.assertEqual(native.old_reids.old_reids,[])
        self.assertEqual(before_rng,fingerprint((random.getstate(),np.random.get_state(),torch.get_rng_state())))
        self.assertFalse(controller.wants_guard())
        self.assertIs(controller.propose(s),s)
    def test_empty_group_advances_ledger_without_inventing_observation(self):
        c=IdentityRevisionController(SimpleNamespace(ENABLED=True,MODE='guard',ENABLE_PROVISIONAL=False,
            ENABLE_REVISION=False,ENABLE_RELINK=False,REVISION_WINDOW=0))
        inst,record=observation(1,1,1);inst=inst[:0]
        c.observe([inst],[record]);c.committed(inst,{},{});c.published()
        self.assertEqual(c.ledger.latest_frame,1);self.assertEqual(c.ledger.total_observations,0)


if __name__=='__main__':unittest.main()
