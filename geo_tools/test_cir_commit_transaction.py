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

    def test_native_bank_and_commit_equal_full_snapshot(self):
        from gtr.modeling.identity_revision.state_transaction import StateTransaction
        outcomes = []
        for tx_type in (StateTransaction, NativeCommitTransaction):
            tracker = self.make()
            m = tracker.model
            history = tracker.instances[0]
            query = instance(n=2)
            tracker.cir_window = [history, query]
            tracker.cir_older = [instance([1], 1)]
            tracker.counts = {1: 1, 2: 1, 40: 2}
            tracker.reids = {1: history[0], 2: history[1], 40: instance([40, 40])}
            tracker.id_count = 40
            native.poss_ids.poss_ids = {40}
            m.get_asso = lambda inst, k, geo_context=False: ([torch.full((len(inst[-1]), len(inst[0])), .8)], None, None, None, None)
            before = fingerprint(native_state(tracker, native))
            head = {k: v.clone() for k, v in m.cir_controller.probe_head.state_dict().items()}
            tx = tx_type(tracker, native)
            from contextlib import nullcontext
            with tx if tx_type is NativeCommitTransaction else nullcontext():
                ids = m.resolve_bank_fallback(tracker.cir_window, 1, torch.tensor([-1, -1]),
                    torch.tensor([1, 2]), tracker.counts, tracker.reids, tracker.cir_older, 0)
                result = m.commit_native_identity_state(tracker.cir_window, 1, ids,
                    tracker.id_count, tracker.counts, tracker.reids)
                tracker.id_count = result[1]
                outcomes.append(fingerprint(dict(window=result[0], counts=result[2], reids=tracker.reids,
                    poss=native.poss_ids.poss_ids, old=native.old_reids.old_reids)))
                with torch.no_grad():
                    m.cir_controller.probe_head.weight.add_(99)
            tx.restore()
            self.assertEqual(before, fingerprint(native_state(tracker, native)))
            for k, v in head.items():
                self.assertTrue(torch.equal(v, m.cir_controller.probe_head.state_dict()[k]))
        self.assertEqual(outcomes[0], outcomes[1])

    def test_real_guard_loop_fault_matches_full_backend(self):
        from test_cir_guard_native import NativeGuardTests, fixture, run
        from unittest.mock import patch
        # Reuse the native guard's existing post-commit failure assertions on
        # both backends; COW context exits before restoration and raw recovery.
        real_fixture = fixture
        def cow_fixture(*args, **kwargs):
            m, o, scores = real_fixture(*args, **kwargs)
            m.cir_controller.commit_snapshot = 'cow_engineering'
            return m, o, scores
        with patch('test_cir_guard_native.fixture', side_effect=cow_fixture):
            test = NativeGuardTests('test_fault_restores_native_roots_bank_rng_and_commits_original_once')
            result = unittest.TestResult()
            test.run(result)
            self.assertEqual(result.errors, [])
            self.assertEqual(result.failures, [])
            m, o, scores = fixture()
            m.cir_controller.commit_snapshot = 'cow_engineering'
            self.assertEqual(run(m, o, scores)[0][-1].track_ids.tolist(), [1])


if __name__ == '__main__':
    unittest.main()
