from dataclasses import replace
import unittest
import common
from gtr.modeling.identity_revision.conflict_detector import ConflictEvidence,FixedConflictDetector
from gtr.modeling.identity_revision.types import ConflictState


class ConflictContracts(unittest.TestCase):
    def event(self):return ConflictEvidence(1,2,True,3,.8,.4,.95,.5,.01)
    def test_joint_conflict_required(self):
        state,reasons=FixedConflictDetector().classify(self.event())
        self.assertEqual(state,ConflictState.SUSPECT);self.assertEqual(len(reasons),5)
    def test_each_missing_support_prevents_veto(self):
        for changes in [dict(local_consecutive_frames=2),dict(local_iou=.6),dict(local_iou_margin=.2),
                        dict(local_cosine=.8),dict(selected_history_cosine=.8),dict(normalized_assignment_margin=.1),
                        dict(local_candidate_legal=False),dict(selected_id=-1)]:
            with self.subTest(changes=changes):
                self.assertEqual(FixedConflictDetector().classify(replace(self.event(),**changes))[0],ConflictState.UNKNOWN)
    def test_same_identity_supported_is_safe(self):
        self.assertEqual(FixedConflictDetector().classify(replace(self.event(),selected_id=2))[0],ConflictState.SAFE)
    def test_unknown_not_negative(self):
        self.assertEqual(FixedConflictDetector().classify(replace(self.event(),local_cosine=float('nan')))[0],ConflictState.UNKNOWN)
    def test_gt_cannot_be_supplied(self):
        with self.assertRaises(TypeError):ConflictEvidence(**dict(selected_id=1,local_id=2,GT_person=3))


if __name__=='__main__':unittest.main()
