"""Real ledger contracts; no native revision qualification implied."""
import unittest
import common
from gtr.modeling.identity_revision.fragment_ledger import FragmentLedger
from gtr.modeling.identity_revision.types import SourceKey,RevisionAction


def source(frame,camera=1,video=4,row=0):return SourceKey(video,frame,camera,1000*camera+frame,row)


class LedgerContracts(unittest.TestCase):
    def test_deterministic_bounded_metadata(self):
        a,b=FragmentLedger(2,2),FragmentLedger(2,2)
        for ledger in (a,b):
            for f in range(1,20):ledger.register_commit(source(f),1);ledger.mark_published(f)
            self.assertTrue(ledger.validate_invariants())
            fragment=next(iter(ledger.fragments.values()))
            self.assertEqual(len(fragment.observation_source_keys),2);self.assertEqual(fragment.observations_seen,19)
        self.assertEqual(a.state_digest(),b.state_digest())
    def test_crosscamera_same_identity_is_legal(self):
        ledger=FragmentLedger();ledger.register_commit(source(1),1);ledger.register_commit(source(1,2),1)
        self.assertTrue(ledger.validate_invariants());self.assertEqual(len(ledger.fragments),2)
    def test_samecamera_identity_collision_rejected(self):
        ledger=FragmentLedger();ledger.register_commit(source(1),1)
        with self.assertRaises(ValueError):ledger.register_commit(source(1,row=1),1)
    def test_video_reset_required(self):
        ledger=FragmentLedger();ledger.register_commit(source(1),1)
        with self.assertRaises(ValueError):ledger.register_commit(source(1,video=5),1)
        ledger.reset();ledger.register_commit(source(1,video=5),1)
        self.assertEqual(ledger.video,5)
    def test_no_future_or_unpublished_eviction(self):
        ledger=FragmentLedger(2,1);ledger.register_commit(source(1),1)
        with self.assertRaises(RuntimeError):ledger.register_commit(source(3),1)
        ledger.mark_published(1);ledger.register_commit(source(3),1)
        with self.assertRaises(ValueError):ledger.register_commit(source(2),2)
    def test_gt_cannot_be_passed_as_source_field(self):
        with self.assertRaises(TypeError):SourceKey(4,1,1,2,0,gt_person=3)
    def test_continuous_oldest_camera_fragment_does_not_block_ended_eviction(self):
        ledger=FragmentLedger(16,2)
        ledger.observe_group(4,1)
        first=ledger.register_commit(source(1),1);ledger.register_commit(source(1,2),1);ledger.mark_published(1)
        ledger.observe_group(4,2);ledger.register_commit(source(2),1);ledger.mark_published(2)
        ledger.observe_group(4,3);ledger.register_commit(source(3),1)
        # The oldest-created fragment is current and unpublished. The ended
        # camera2 fragment is published and can safely leave the bounded ledger.
        ledger.register_commit(source(3,2),1)
        self.assertIn(first,ledger.fragments);self.assertEqual(ledger.fragments[first].observations_seen,3)
        self.assertEqual(ledger.evictions,1);self.assertEqual(len(ledger.fragments),2)
        self.assertTrue(ledger.validate_invariants())
    def test_previous_group_fragment_is_retained_for_pending_camera(self):
        ledger=FragmentLedger(16,1);ledger.register_commit(source(1),1);ledger.mark_published(1)
        with self.assertRaises(RuntimeError):ledger.register_commit(source(2,2),1)
        self.assertEqual(ledger.evictions,0)
    def test_future_action_evidence_rejected(self):
        with self.assertRaises(ValueError):RevisionAction('VETO',source(1),-1,3,(source(2),),1)


if __name__=='__main__':unittest.main()
