"""COW engineering qualification; does not enable COW in native inference."""
import random
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from test_cir_native_commit import instance, model, native
from gtr.modeling.identity_revision.state_transaction import fingerprint, native_state
from gtr.modeling.identity_revision.commit_transaction import NativeCommitTransaction


class CommitTransactionTests(unittest.TestCase):
    def make(self):
        history = instance([1, 2])
        m = model(True)
        m.cir_controller.probe_head = torch.nn.Linear(2, 1)
        tracker = SimpleNamespace(model=m, instances=[history],
            reids={1: history[0]}, id_count=2, counts={1: 1})
        native.poss_ids.poss_ids = {1}
        native.old_ids.old_ids = set()
        native.old_reids.old_reids = [history[0]]
        return tracker

    def test_full_restore_alias_weights_and_rng(self):
        with torch.inference_mode():
            tracker = self.make()
            before = fingerprint(native_state(tracker, native))
            weights = {k: v.clone() for k, v in tracker.model.cir_controller.probe_head.state_dict().items()}
            tx = NativeCommitTransaction(tracker, native)
            with tx:
                rng = (random.random(), np.random.rand(), torch.rand(3))
                tracker.instances[0].reid_features.add_(99)
                tracker.model.cir_controller.probe_head.weight.fill_(42)
                tracker.counts[1] = 999
                tracker.id_count = 99
                native.old_reids.old_reids.append('bad')
                native.poss_ids.poss_ids.add(44)
            tx.restore()
            self.assertEqual(before, fingerprint(native_state(tracker, native)))
            for k, v in weights.items():
                self.assertTrue(torch.equal(v, tracker.model.cir_controller.probe_head.state_dict()[k]))
            self.assertEqual(random.random(), rng[0])
            self.assertEqual(np.random.rand(), rng[1])
            self.assertTrue(torch.equal(torch.rand(3), rng[2]))
            tracker.reids[1].reid_features.fill_(77)
            self.assertTrue((tracker.instances[0].reid_features[0] == 77).all())
            self.assertTrue((native.old_reids.old_reids[0].reid_features == 77).all())
            with self.assertRaisesRegex(RuntimeError, 'single-use'):
                tx.restore()

    def test_readonly_rhs_has_no_backup(self):
        tracker = self.make()
        tx = NativeCommitTransaction(tracker, native)
        with tx:
            fresh = torch.zeros_like(tracker.instances[0].reid_features)
            fresh.add_(tracker.instances[0].reid_features)
        self.assertEqual(len(tx.saved), 0)
        tx.restore()

    def test_resize_rejected_before_mutation(self):
        tracker = self.make()
        before = fingerprint(native_state(tracker, native))
        tx = NativeCommitTransaction(tracker, native)
        with self.assertRaisesRegex(RuntimeError, 'resize/rebind'):
            with tx:
                tracker.instances[0].reid_features.resize_(1)
        tx.restore()
        self.assertEqual(before, fingerprint(native_state(tracker, native)))

    def test_metadata_rebind_and_mutation_saved_once(self):
        tracker = self.make()
        before = fingerprint(native_state(tracker, native))
        tx = NativeCommitTransaction(tracker, native)
        with tx:
            tracker.instances[0].reid_features.add_(1)
            tracker.instances[0].reid_features.mul_(2)
            tracker.instances[0].reid_features = torch.zeros_like(tracker.instances[0].reid_features)
        self.assertEqual(len(tx.saved), 1)
        tx.restore()
        self.assertEqual(before, fingerprint(native_state(tracker, native)))


if __name__ == '__main__':
    unittest.main()
