"""CPU rollback tests on real Detectron2 Instances and native storage aliases."""
import random
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from test_cir_native_commit import instance, model, native
from gtr.modeling.identity_revision.state_transaction import StateTransaction, native_state, fingerprint


class RollbackTests(unittest.TestCase):
    def test_restore_instances_rng_and_alias(self):
        history=instance([1,2]);m=model(True)
        tracker=SimpleNamespace(model=m,instances=[history],counts={1:1,2:1},
            reids={1:history[0],2:history[1]},id_count=2,output_buffer=[])
        native.poss_ids.poss_ids={1};native.old_ids.old_ids=set();native.old_reids.old_reids=[history[0]]
        tx=StateTransaction(tracker,native)
        rng=(random.random(),np.random.rand(),torch.rand(3))
        history.reid_features.fill_(99);tracker.id_count=80;tracker.counts[1]=20
        native.poss_ids.poss_ids.add(19);tracker.output_buffer.append('branch')
        tx.restore()
        self.assertEqual(fingerprint(native_state(tracker,native)),tx.before_digest)
        self.assertEqual(random.random(),rng[0]);self.assertEqual(np.random.rand(),rng[1])
        self.assertTrue(torch.equal(torch.rand(3),rng[2]))
        tracker.reids[1].reid_features.fill_(77)
        self.assertTrue((tracker.instances[0].reid_features[0]==77).all())
        self.assertTrue((native.old_reids.old_reids[0].reid_features==77).all())
        tx.restore()
        self.assertFalse((tracker.instances[0].reid_features[0]==77).all())

    def test_metadata_ledger_and_unpublished_buffer_restore(self):
        from gtr.modeling.identity_revision.fragment_ledger import FragmentLedger
        from gtr.modeling.identity_revision.types import SourceKey
        from gtr.modeling.identity_revision.output_buffer import OutputBuffer
        m=model(True);m.cir_controller.ledger=FragmentLedger()
        ledger=m.cir_controller.ledger;ledger.register_commit(SourceKey(4,1,1,10,0),1);ledger.mark_published(1)
        buffer=OutputBuffer(2);buffer.push(1,{'ids':[1]})
        tracker=SimpleNamespace(model=m,instances=[instance([1],1)],counts={1:1},reids={},id_count=1,output_buffer=buffer)
        native.poss_ids.poss_ids=set();native.old_ids.old_ids=set();native.old_reids.old_reids=[]
        before=ledger.state_digest();tx=StateTransaction(tracker,native)
        ledger.register_commit(SourceKey(4,2,1,11,0),1);buffer.push(2,{'ids':[2]})
        tx.restore()
        self.assertEqual(m.cir_controller.ledger.state_digest(),before)
        self.assertEqual(tracker.output_buffer.flush(),[(1,{'ids':[1]})])
        self.assertTrue(m.cir_controller.ledger.validate_invariants())


if __name__=='__main__':unittest.main()
