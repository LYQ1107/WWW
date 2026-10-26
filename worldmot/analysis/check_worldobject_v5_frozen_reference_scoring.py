"""Synthetic scoring-scope tests; no real weights, data or GT is opened."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from worldobject_v5_frozen_reference_scoring import (
    NAMES, FULL_SHA, paths, require_frozen_scoring, require_publication, validate_metric, same_reference_inputs)
from worldobject_v5_reference_scope import equivalent_reference_inputs
from worldmot.runtime.world_object_v5_contract import (SCHEMA, BASE, registration_path,
    v5_source_fingerprint, require_v5_scope, load_v5_membership_gate, LEGAL_TEST)
from worldmot.runtime.experiment_authorization import APPROVAL_SCHEMA, source_fingerprint
from worldmot.data.worldobject_cache import sha256


class FrozenReferenceScoringChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        private = self.root / "provenance"
        private.mkdir()
        self.runtime = self.root / "runtime"
        self.runtime.mkdir()
        self.reference = {"schema": SCHEMA, "base_commit": BASE, "source_sha256": source_fingerprint(),
            "v5_source_sha256": v5_source_fingerprint(), "registered_config_sha256": sha256(registration_path()),
            "calibration_sha256": "synthetic-calibration", "frozen_C1_checkpoint_sha256": FULL_SHA,
            "human_instruction_sha256": "synthetic-human", "variants": list(NAMES), "reference_only": True,
            "splits": {"test": {"scenes": {}}}, "variant_configs": {"B1": {"frozen": True}},
            "TEST_parameter_selection": False, "GT_in_model_inputs": False}
        self.old_scope = private / "scope_v5_reference_test.json"
        self.save(self.old_scope, self.reference)
        self.scope = {**deepcopy(self.reference), "frozen_reference_scoring_only": True,
            "R2_full24_eligibility": {"checkpoint_sha256": FULL_SHA}, "trained_B2_B3_B4_eligible": False,
            "unchanged_original_locks": {str(self.old_scope): sha256(self.old_scope)}}
        self.scope_path, self.auth_path = paths(self.root)
        self.auth = {"schema": APPROVAL_SCHEMA, "approved": True, "source_commit": "SYNTHETIC_ONLY",
            "source_sha256": source_fingerprint(), "user_authorization_reference": "SYNTHETIC_NO_REAL_OPERATION",
            "scope_lock_sha256": None, "operations": ["evaluate"], "splits": ["test"],
            "method_ids": ["V5_B0", "V5_B1"]}
        self.refresh()

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def save(path, value):
        Path(path).write_text(json.dumps(value))

    def refresh(self):
        self.save(self.scope_path, self.scope)
        self.auth["scope_lock_sha256"] = sha256(self.scope_path)
        self.save(self.auth_path, self.auth)

    def test_only_frozen_B0_B1_can_use_the_scoring_scope(self):
        self.assertEqual(require_frozen_scoring(self.root, "B1")["frozen_C1_checkpoint_sha256"], FULL_SHA)
        for variant in ("B2", "B3", "B4"):
            with self.assertRaisesRegex(PermissionError, "ALREADY_TRAINED_FROZEN"):
                require_frozen_scoring(self.root, variant)

    def test_score_authorization_cannot_train_replay_live_or_diagnose(self):
        for operation in ("train", "replay", "preflight", "live", "diagnostics"):
            with self.assertRaises(PermissionError):
                require_v5_scope(self.scope_path, self.auth_path, operation, "test", "B0")

    def test_extra_permissions_cannot_be_smuggled_into_the_scoring_scope(self):
        self.auth["operations"].append("train")
        self.save(self.auth_path, self.auth)
        with self.assertRaisesRegex(PermissionError, "SCORING_SCOPE_MISMATCH"):
            require_frozen_scoring(self.root, "B0")

    def test_old_lock_mutation_is_not_accepted(self):
        self.old_scope.write_text(self.old_scope.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "ORIGINAL_TRAIN_OR_REFERENCE_LOCK_CHANGED"):
            require_frozen_scoring(self.root, "B0")

    def test_changed_new_scope_cannot_change_the_frozen_method(self):
        self.scope["variant_configs"]["B1"]["frozen"] = False
        self.refresh()
        with self.assertRaisesRegex(ValueError, "EXACT_ALREADY_FROZEN_REFERENCE"):
            require_frozen_scoring(self.root, "B1")

    def test_unpublished_schedule_cannot_open_the_evaluator(self):
        with self.assertRaisesRegex(PermissionError, "PUBLISH_SCORING_ORDER"):
            require_publication(self.root, self.runtime, "B1")

    def test_existing_all3_canonical_adoption_gate_is_not_relaxed(self):
        canonical = deepcopy(self.reference)
        canonical["final_weights"] = {"B2": {}, "B3": {}}
        with self.assertRaisesRegex(ValueError, "ALL3_FULL24_GATES_REQUIRED"):
            equivalent_reference_inputs(self.reference, canonical)

    def test_selection_checkpoint_still_cannot_load_as_a_final_model(self):
        gate = self.root / "selection_gate.json"
        self.save(gate, {"status": "SELECTION_COMPLETE"})
        with self.assertRaisesRegex(ValueError, "FULL_TRAIN_GATE_REQUIRED"):
            load_v5_membership_gate(gate, SimpleNamespace(variant="B3"))

    def metric_fixture(self):
        audit = {"test": {"scenes": [{"scene": s, "name": "scene" + str(s)} for s in range(1, 23)]}}
        inputs = {"scenes": {str(s): {"output_sha256": {"tracks.jsonl": "source" + str(s)}} for s in range(1, 23)}}
        metric = {"status": "COMPLETE_VALID_SCOPE", "raw_GT_modified": False,
            "evaluated_scenes": ["scene" + str(s) for s in LEGAL_TEST],
            "coverage": [{"scene": s, "frames": 1 if s < 22 else 28998, "predictions_sha256": "source" + str(s)} for s in range(1, 23)],
            "blocked": [{"scene": s, "reason": "RAW_GT_DUPLICATE_OR_VERSION_INVALID"} for s in range(1, 23) if s not in LEGAL_TEST]}
        return metric, inputs, audit

    def test_formal_score_requires_all22_and_original_legal14(self):
        metric, inputs, audit = self.metric_fixture()
        validate_metric(metric, inputs, audit)
        metric["coverage"].pop()
        with self.assertRaisesRegex(ValueError, "EXACT_FULL22_INFERENCE"):
            validate_metric(metric, inputs, audit)

    def test_native_metrics_cannot_use_different_prediction_bytes(self):
        metric, inputs, audit = self.metric_fixture()
        metric["coverage"][0]["predictions_sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "DIFFERENT_PREDICTION_BYTES"):
            validate_metric(metric, inputs, audit)


if __name__ == "__main__":
    unittest.main()
