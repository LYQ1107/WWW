"""Generated scope and denominator checks; no real data, weights or inference."""
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from audit_worldobject_v5_train_snapshot import (
    SCENES, TRAIN_SHA, cross_view_counts, snapshot_payload,
    validate_publication, validate_registration)


def registered():
    hashes = {"v5_source_sha256": "f" * 64, "driver_sha256": "d" * 64}
    return {"status": "PREREGISTERED_FINITE_TRAIN_SNAPSHOT_AUDIT", "variant": "B3",
        "split": "train", "scenes": SCENES, "expected_paired_times": 5860,
        "expected_candidates": 172760, "device": "cpu", "threads": 1,
        "TEST_read": False, "optimizer_created": False,
        "model_or_budget_selection_uses_this_audit": False,
        "formal_metric": False, "TRAIN_annotations_sha256": TRAIN_SHA,
        "hashes": hashes, "snapshot_sha256": "a" * 64}, hashes


class TrainSnapshotScopeChecks(unittest.TestCase):
    def test_only_fixed_train6_cpu_scope_is_accepted(self):
        registration, hashes = registered()
        validate_registration(registration, hashes)
        for changed in ({"split": "test"}, {"scenes": [1]}, {"device": "cuda:0"},
                        {"TEST_read": True}, {"model_or_budget_selection_uses_this_audit": True}):
            value = {**registration, **changed}
            with self.assertRaisesRegex(PermissionError, "TRAIN6"):
                validate_registration(value, hashes)

    def test_changed_source_or_original_annotations_are_rejected(self):
        registration, hashes = registered()
        for changed in ({"TRAIN_annotations_sha256": "0" * 64}, {"hashes": {}}):
            with self.assertRaises(PermissionError):
                validate_registration({**registration, **changed}, hashes)

    def test_changed_snapshot_is_rejected_before_torch_load(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "generated.pt"
            path.write_bytes(b"generated scope fixture, not model weights")
            with patch("audit_worldobject_v5_train_snapshot.torch.load") as loader:
                with self.assertRaisesRegex(ValueError, "BEFORE_MODEL_LOAD"):
                    snapshot_payload(path, "0" * 64)
                loader.assert_not_called()

    def test_unpublished_or_changed_registration_is_rejected(self):
        registration, _ = registered()
        receipt = {"status": "GITHUB_REF_TREE_AND_PUBLIC_FETCH_VERIFIED_FOR_TRAIN_SNAPSHOT_AUDIT",
            "registration_sha256": "b" * 64, "snapshot_sha256": registration["snapshot_sha256"],
            "driver_sha256": registration["hashes"]["driver_sha256"], "remote_commit": "c" * 40}
        validate_publication(receipt, "b" * 64, registration)
        for changed in ({"status": "PENDING"}, {"snapshot_sha256": "0" * 64},
                        {"registration_sha256": "0" * 64}):
            with self.assertRaisesRegex(PermissionError, "BEFORE_MODEL_LOAD"):
                validate_publication({**receipt, **changed}, "b" * 64, registration)

    def test_missing_output_never_becomes_a_correct_identity_pair(self):
        observations = [SimpleNamespace(key=(v, 0), camera_id=v) for v in (1, 2)]
        for labels in (["p", "p"], ["p", "q"]):
            counts = Counter()
            cross_view_counts(counts, observations, labels, {(1, 0): None, (2, 0): 7})
            self.assertEqual(counts["missing_output_cross_view_pair_opportunities"], 1)
            self.assertEqual(counts["correct_same_person_shared_ID"], 0)
            self.assertEqual(counts["correct_different_person_separate"], 0)


if __name__ == "__main__":
    unittest.main()
