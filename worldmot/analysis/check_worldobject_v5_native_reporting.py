"""A completed-looking metric file cannot replace a bound scoring receipt."""
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from summarize_worldobject_v5 import frozen_native_for_reporting
from worldmot.data.worldobject_cache import sha256
from worldmot.runtime.world_object_v5_contract import FULL_SHA


class ReportingReceiptChecks(unittest.TestCase):
    def test_unscored_reference_remains_pending(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "evaluation_frozen_references/B1"
            folder.mkdir(parents=True)
            (folder / "metrics.json").write_text('{"status":"COMPLETE_VALID_SCOPE"}')
            self.assertIsNone(frozen_native_for_reporting(root, root, "B1"))

    def test_new_model_cannot_use_frozen_control_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "evaluation_frozen_references/B3"
            folder.mkdir(parents=True)
            (folder / "frozen_scoring_receipt.json").write_text('{"status":"COMPLETE_FROZEN_REFERENCE_FULL22_NATIVE14"}')
            self.assertIsNone(frozen_native_for_reporting(root, root, "B3"))

    def fixture(self, root):
        folder = root / "evaluation_frozen_references/B1"
        folder.mkdir(parents=True)
        inputs = root / "provenance/frozen_reference_score_inputs/B1.json"
        inputs.parent.mkdir(parents=True)
        scope_file, auth_file = root / "scope.json", root / "auth.json"
        scope_file.write_text('{}')
        auth_file.write_text('{}')
        metric = folder / "metrics.json"
        metric.write_text('{"status":"COMPLETE_VALID_SCOPE"}')
        scope = {"original_TEST_annotations_sha256": "original-annotations", "v5_source_sha256": "frozen-source"}
        inputs.write_text(json.dumps({"status": "ALL22_FROZEN_REFERENCE_OUTPUT_BYTES_VERIFIED_BEFORE_GT",
            "variant": "B1", "GT_read": False, "scope_sha256": sha256(scope_file),
            "source_sha256": "frozen-source", "scenes": {str(i): {"candidates": 548583 if i == 1 else 0} for i in range(1, 23)}}))
        receipt = {"status": "COMPLETE_FROZEN_REFERENCE_FULL22_NATIVE14", "variant": "B1",
            "metrics_sha256": sha256(metric), "input_byte_verification_sha256": sha256(inputs),
            "checkpoint_sha256": FULL_SHA, "scope_sha256": sha256(scope_file),
            "authorization_sha256": sha256(auth_file), "original_annotations_sha256": "original-annotations",
            "original_GT_modified": False, "TRAIN_parameters_or_budget_rules_changed": False}
        (folder / "frozen_scoring_receipt.json").write_text(json.dumps(receipt))
        return folder, inputs, scope, scope_file, auth_file, receipt

    def rejects(self, change):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder, inputs, scope, scope_file, auth_file, receipt = self.fixture(root)
            change(folder, inputs, receipt)
            (folder / "frozen_scoring_receipt.json").write_text(json.dumps(receipt))
            with ExitStack() as stack:
                stack.enter_context(patch("worldobject_v5_frozen_reference_scoring.require_publication", return_value=scope))
                stack.enter_context(patch("worldobject_v5_frozen_reference_scoring.paths", return_value=(scope_file, auth_file)))
                native_check = stack.enter_context(patch("worldobject_v5_frozen_reference_scoring.validate_metric"))
                with self.assertRaisesRegex(ValueError, "COMPLETED_FROZEN_NATIVE_RECEIPT"):
                    frozen_native_for_reporting(root, root, "B1")
                native_check.assert_not_called()

    def test_changed_metric_bytes_rejected(self):
        self.rejects(lambda folder, inputs, receipt: (folder / "metrics.json").write_text('{"status":"COMPLETE_VALID_SCOPE","edited":true}'))

    def test_different_checkpoint_rejected(self):
        self.rejects(lambda folder, inputs, receipt: receipt.update(checkpoint_sha256="0" * 64))

    def test_unbound_input_proof_rejected(self):
        self.rejects(lambda folder, inputs, receipt: inputs.write_text('{"status":"FORGED_INPUT_PROOF"}'))


if __name__ == "__main__":
    unittest.main()
