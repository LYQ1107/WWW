"""Synthetic identity-event checks; these are not real-data experiment scores."""
from collections import Counter
from types import SimpleNamespace
import unittest
import numpy as np

from attribute_worldobject_v5_frozen_test import DeliveredIdentityAudit, matrix_contrast, global_retention_flags
from worldobject_v5_audit_common import decision_events, compare_events


def observations():
    return [SimpleNamespace(key=(v, 0), camera_id=v, detection_id=0,
            bbox_xywh=np.array([1., 2., 3., 4.]), score=.8) for v in (1, 2)]


def prediction(t, gids=(42, 42), drop=False):
    row = {"timestamp": t, "tracks_2d_by_camera": {}, "decisions": [],
           "tracks_3d": [{"global_id": gid, "status": "CONFIRMED", "missed": 0} for gid in set(gids)]}
    for j, o in enumerate(observations()):
        row["tracks_2d_by_camera"][str(o.camera_id)] = [] if drop and j == 1 else [{
            "detection_id": 0, "bbox_xywh": o.bbox_xywh.tolist(), "score": .8,
            "global_id": gids[j], "visible": True, "evaluation_eligible": True, "projected_from_world": False}]
        row["decisions"].append({"camera_id": o.camera_id, "detection_id": 0, "global_id": gids[j],
                                 "stage": "TRACK_AWARE_DUPLICATE" if drop and j == 1 else "HISTORICAL_MEMBERSHIP"})
    return row


class IdentityEventChecks(unittest.TestCase):
    def test_pair_correction_and_regression_are_symmetric(self):
        obs = observations()
        a, _ = decision_events(obs, [9, 9], {0: ("N", (0,)), 1: ("N", (1,))}, {}, set())
        b, _ = decision_events(obs, [9, 9], {0: ("N", (0, 1)), 1: ("N", (0, 1))}, {}, set())
        counter = Counter()
        compare_events(a, b, counter)
        self.assertEqual(counter["same_person_shared/corrections"], 1)
        compare_events(b, a, counter)
        self.assertEqual(counter["same_person_shared/regressions"], 1)
        self.assertEqual(counter["same_person_shared/net"], 0)

    def test_different_people_shared_new_identity_is_error(self):
        events, counts = decision_events(observations(), [9, 10], {0: ("N", (0, 1)), 1: ("N", (0, 1))}, {}, set())
        self.assertFalse(events["different_person_separate", ((1, 0), (2, 0))])
        self.assertEqual(counts["new_IDs_with_multiple_clean_people"], 1)

    def test_missing_output_is_not_correct_separation(self):
        events, counts = decision_events(observations(), [9, 10], {0: None, 1: ("N", (1,))}, {}, set())
        self.assertFalse(events["different_person_separate", ((1, 0), (2, 0))])
        self.assertEqual(counts["suppressed_clean_label_observations"], 1)

    def test_false_birth_requires_correct_active_history(self):
        _, counts = decision_events(observations(), [9, 9], {0: ("N", (0, 1)), 1: ("N", (0, 1))}, {17: {9}}, {17})
        self.assertEqual(counts["false_birth_observations"], 2)

    def test_global_graph_preserves_shared_identity(self):
        audit = DeliveredIdentityAudit()
        audit.observe(prediction(1), observations(), ["p", "p"], 0, 1, Counter())
        graph, mapping = audit.graph()
        self.assertEqual(mapping, {"p": 42})
        self.assertEqual(graph["global_one_to_one_matched_clean_observations"], 2)

    def test_lost_recovery_and_null_count_are_per_object(self):
        audit = DeliveredIdentityAudit()
        audit.observe(prediction(1), observations(), ["p", "p"], 0, 2, Counter())
        audit.active[42]["status"] = "LOST"
        audit.observe(prediction(2), observations(), ["p", "p"], 1, 2, Counter())
        self.assertEqual(audit.counts["LOST_recovery_correct_pure_identity"], 2)
        self.assertEqual(audit.counts["NULL_unique_active_history_opportunities"], 1)

    def test_contamination_is_never_forgotten(self):
        audit = DeliveredIdentityAudit()
        audit.observe(prediction(1), observations(), ["p", "p"], 0, 3, Counter())
        audit.observe(prediction(2), observations(), ["p", "q"], 1, 3, Counter())
        audit.observe(prediction(3), observations(), ["p", "p"], 2, 3, Counter())
        self.assertEqual(audit.counts["wrong_historical_ID_reuse"], 1)
        self.assertEqual(audit.counts["identity_contamination_onsets"], 1)
        self.assertEqual(audit.counts["reuse_of_cumulatively_mixed_ID"], 2)

    def test_current_box_mutation_is_rejected(self):
        row = prediction(1)
        row["tracks_2d_by_camera"]["1"][0]["bbox_xywh"][0] += 1
        with self.assertRaisesRegex(ValueError, "BOX_OR_SCORE_CHANGED"):
            DeliveredIdentityAudit().observe(row, observations(), ["p", "p"], 0, 1, Counter())

    def test_suppressed_clean_detection_remains_in_denominator(self):
        audit = DeliveredIdentityAudit()
        _, assigned = audit.observe(prediction(1, drop=True), observations(), ["p", "p"], 0, 1, Counter())
        self.assertIsNone(assigned[(2, 0)])
        self.assertEqual(audit.counts["clean_labelled_frozen_candidates"], 2)
        self.assertEqual(audit.counts["clean_labelled_candidates_dropped"], 1)

    def test_delivered_duplicate_birth_corrections_and_regressions(self):
        split, shared = DeliveredIdentityAudit(), DeliveredIdentityAudit()
        a, _ = split.observe(prediction(1, gids=(42, 43)), observations(), ["p", "p"], 0, 1, Counter())
        b, _ = shared.observe(prediction(1), observations(), ["p", "p"], 0, 1, Counter())
        self.assertEqual(split.counts["simultaneous_duplicate_birth_pairs"], 1)
        self.assertEqual(split.counts["same_person_birth_pair_opportunities"], 1)
        self.assertEqual(split.counts["within_timestamp_extra_birth_IDs_per_person"], 1)
        total = Counter()
        compare_events(a, b, total)
        self.assertEqual(total["duplicate_birth_avoided/corrections"], 1)
        self.assertEqual(total["duplicate_birth_avoided/net"], 1)
        compare_events(b, a, total)
        self.assertEqual(total["duplicate_birth_avoided/regressions"], 1)
        self.assertEqual(total["duplicate_birth_avoided/net"], 0)

    def test_dropped_output_cannot_correct_duplicate_birth(self):
        split, dropped = DeliveredIdentityAudit(), DeliveredIdentityAudit()
        a, _ = split.observe(prediction(1, gids=(42, 43)), observations(), ["p", "p"], 0, 1, Counter())
        b, _ = dropped.observe(prediction(1, drop=True), observations(), ["p", "p"], 0, 1, Counter())
        total = Counter()
        compare_events(a, b, total)
        self.assertEqual(dropped.counts["same_person_birth_pair_opportunities"], 1)
        self.assertEqual(total["duplicate_birth_avoided/opportunities"], 1)
        self.assertEqual(total["duplicate_birth_avoided/corrections"], 0)

    def test_history_plus_new_ID_is_not_simultaneous_duplicate_birth(self):
        audit = DeliveredIdentityAudit()
        audit.observe(prediction(1), observations(), ["p", "p"], 0, 2, Counter())
        events, _ = audit.observe(prediction(2, gids=(42, 43)), observations(), ["p", "p"], 1, 2, Counter())
        self.assertEqual(audit.counts["simultaneous_duplicate_birth_pairs"], 0)
        self.assertTrue(events["duplicate_birth_avoided", ((1, 0), (2, 0))])

    def test_vector_pair_counts_include_both_changes(self):
        counter = Counter()
        matrix_contrast(counter, np.array([True, False]), np.array([False, True]), np.array([True, True]))
        self.assertEqual(counter, {"opportunities": 2, "corrections": 1, "regressions": 1, "net": 0})

    def test_no_clean_gt_opportunities_remain_zero_without_invented_success(self):
        audit = DeliveredIdentityAudit()
        audit.observe(prediction(1), observations(), [None, None], 0, 1, Counter())
        _, mapping = audit.graph()
        flags = global_retention_flags([], 0, mapping)
        counter = Counter()
        matrix_contrast(counter, flags, flags, np.ones(flags.shape, bool))
        self.assertEqual(dict(counter), {"opportunities": 0, "corrections": 0, "regressions": 0, "net": 0})


if __name__ == "__main__":
    unittest.main()
