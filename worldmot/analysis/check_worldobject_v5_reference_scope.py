"""Synthetic scope/adoption tests; no real data, model or GT is accessed."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from worldobject_v5_reference_scope import (equivalent_reference_inputs, require_canonical_or_adopted,
    adoption_path, sha256)
from run_worldobject_v5_controller import live_gpu_reservations
from run_worldobject_v5_reference_queue import terminal_artifact_status
from worldmot.runtime.world_object_v5_contract import (BASE, SCHEMA, require_v5_scope, v5_source_fingerprint,
    registration_path)
from worldmot.runtime.experiment_authorization import APPROVAL_SCHEMA, source_fingerprint


class ReferenceScopeChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.ref = {"reference_only": True, "variants": ["B0", "B1"], "base_commit": BASE,
                    "source_sha256": source_fingerprint(), "v5_source_sha256": v5_source_fingerprint(),
                    "registered_config_sha256": sha256(registration_path()), "calibration_sha256": "cal",
                    "frozen_C1_checkpoint_sha256": "frozen", "original_C1_reference_predictions": "immutable_reference",
                    "splits": {"test": {"scenes": {"1": {"sha": "input"}}}}, "variant_configs": {"B1": {"x": 1}},
                    "schema": SCHEMA, "TEST_parameter_selection": False, "GT_in_model_inputs": False}
        self.canonical = deepcopy(self.ref)
        self.canonical["variants"] = ["B0", "B1", "B2", "B3", "B4"]
        self.canonical["final_weights"] = {v: {"checkpoint_sha256": v} for v in ("B2", "B3", "B4")}
        self.canonical_path = self.root / "canonical.json"
        self.canonical_path.write_text(json.dumps(self.canonical))

    def tearDown(self):
        self.temp.cleanup()

    def test_identical_frozen_inputs_can_be_adopted(self):
        fields = equivalent_reference_inputs(self.ref, self.canonical)
        self.assertIn("splits/test", fields)

    def test_any_changed_roi_or_geometry_manifest_is_rejected(self):
        self.canonical["splits"]["test"]["scenes"]["1"]["sha"] = "different"
        with self.assertRaisesRegex(ValueError, "TEST_INPUTS_DIFFER"):
            equivalent_reference_inputs(self.ref, self.canonical)

    def test_changed_reference_weights_are_rejected(self):
        self.canonical["frozen_C1_checkpoint_sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "WEIGHTS_DIFFER"):
            equivalent_reference_inputs(self.ref, self.canonical)

    def test_changed_b1_config_is_rejected(self):
        self.canonical["variant_configs"]["B1"]["x"] = 2
        with self.assertRaisesRegex(ValueError, "B1_CONFIG_CHANGED"):
            equivalent_reference_inputs(self.ref, self.canonical)

    def test_missing_any_full24_gate_blocks_adoption(self):
        del self.canonical["final_weights"]["B4"]
        with self.assertRaisesRegex(ValueError, "ALL3_FULL24_GATES_REQUIRED"):
            equivalent_reference_inputs(self.ref, self.canonical)

    def test_new_trained_model_cannot_be_reference(self):
        self.ref["variants"].append("B2")
        with self.assertRaisesRegex(ValueError, "ONLY_FROZEN_B0_B1"):
            equivalent_reference_inputs(self.ref, self.canonical)

    def test_no_silent_acceptance_of_other_scope(self):
        with self.assertRaisesRegex(ValueError, "EXPLICIT_REFERENCE_SCOPE_ADOPTION_REQUIRED"):
            require_canonical_or_adopted(self.root, self.canonical_path, "B0", 1, {"scope_sha256": "reference"})

    def test_manifest_mutation_invalidates_adoption(self):
        folder = self.root / "predictions/B0/scene_01"
        folder.mkdir(parents=True)
        manifest = {"scope_sha256": "reference", "output_sha256": {"tracks.jsonl": "immutable-output"}}
        path = folder / "manifest.json"
        path.write_text(json.dumps(manifest))
        receipt = {"canonical_scope_sha256": sha256(self.canonical_path), "reference_scope_sha256": "reference",
                   "original_manifest_sha256": sha256(path), "output_sha256": manifest["output_sha256"],
                   "weights_or_predictions_modified": False, "training_gates_skipped": False}
        target = adoption_path(self.root, "B0", 1)
        target.parent.mkdir(parents=True)
        target.write_text(json.dumps(receipt))
        require_canonical_or_adopted(self.root, self.canonical_path, "B0", 1, manifest)
        path.write_text(json.dumps({**manifest, "changed": True}))
        with self.assertRaisesRegex(ValueError, "ADOPTION_EVIDENCE_MISMATCH"):
            require_canonical_or_adopted(self.root, self.canonical_path, "B0", 1, manifest)

    def test_inference_only_subset_cannot_authorize_scoring_or_new_models(self):
        scope = self.root / "reference.json"
        scope.write_text(json.dumps(self.ref))
        auth = self.root / "synthetic_authorization.json"
        payload = {"schema": APPROVAL_SCHEMA, "approved": True, "source_commit": "OFFLINE_UNIT_TEST_ONLY",
                   "user_authorization_reference": "SYNTHETIC_TEMPORARY_NO_REAL_DATA",
                   "source_sha256": source_fingerprint(), "scope_lock_sha256": sha256(scope),
                   "operations": ["replay", "preflight"], "splits": ["test"],
                   "method_ids": ["V5_B0", "V5_B1", "V5_B2"]}
        auth.write_text(json.dumps(payload))
        with self.assertRaises(PermissionError):
            require_v5_scope(scope, auth, "evaluate", "test", "B0")
        with self.assertRaises(PermissionError):
            require_v5_scope(scope, auth, "replay", "test", "B2")

    def test_uninitialized_cuda_worker_reserves_card_but_dead_pid_does_not(self):
        job = {"pid": 123, "identity": {"start_tick": "1"}, "GPU_UUID": "reserved", "status": "RUNNING"}
        with patch("run_worldobject_v5_controller.process_identity", return_value=job["identity"]):
            self.assertEqual(live_gpu_reservations({"worker": job}), {"reserved"})
        with patch("run_worldobject_v5_controller.process_identity", return_value=None):
            self.assertEqual(live_gpu_reservations({"worker": job}), set())

    def test_worker_finishing_after_first_scan_is_complete_not_failed(self):
        job = {"exit_code": 0}
        self.assertEqual(terminal_artifact_status(job, True), "COMPLETE")
        self.assertNotIn("missing_output_checks", job)

    def test_zero_exit_waits_for_repeat_artifact_checks_without_restart(self):
        job = {"exit_code": 0}
        self.assertEqual(terminal_artifact_status(job, False), "EXITED_PENDING_ARTIFACT")
        self.assertEqual(terminal_artifact_status(job, False), "EXITED_PENDING_ARTIFACT")
        self.assertEqual(terminal_artifact_status(job, False), "ENGINEERING_REVIEW_REQUIRED")

    def test_nonzero_exit_without_output_is_genuine_engineering_failure(self):
        self.assertEqual(terminal_artifact_status({"exit_code": 1}, False), "ENGINEERING_REVIEW_REQUIRED")


if __name__ == "__main__":
    unittest.main()
