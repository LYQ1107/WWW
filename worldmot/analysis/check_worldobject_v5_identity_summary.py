"""Generated records only: protect diagnostic scope and opportunity denominators."""
from collections import Counter
from copy import deepcopy
import hashlib
from pathlib import Path
import tempfile
import unittest

from worldobject_v5_identity_summary import (
    ANNOTATIONS, INITIAL, LEGAL_TEST, VARIANTS, COMPLETE, PENDING,
    build_identity_summary, rates, render_identity_analysis, write_identity_summary)


def fixture():
    source = "f" * 64
    metrics = {v: {k: float(50 + j) for k in ("HOTA", "AssA", "IDF1", "CVIDF1", "MOTA")}
               for j, v in enumerate(VARIANTS)}
    for v in VARIANTS:
        metrics[v]["IDSW"] = 10
    baseline = {k: 80. for k in ("HOTA", "AssA", "IDF1", "CVIDF1", "MOTA")}
    baseline["IDSW"] = 5
    official = {"status": "COMPLETE_NATIVE_LEGAL14_ALL5",
        "canonical_TEST_scope_complete_variants": list(VARIANTS),
        "raw_GT_modified": False, "TEST_parameter_selection": False, "v5_source_sha256": source,
        "V5_official_metrics": metrics, "checkpoint_SHA256": {v: str(j) * 64 for j, v in enumerate(("B2", "B3", "B4"), 1)},
        "historical_baselines": {"GMT Native": baseline, "V4 C1": baseline}, "proofs": {}}
    data = {"status": "COMPLETE_FROZEN_ALL5_FULL22_IDENTITY_ATTRIBUTION", "official_metric": False,
        "TEST_parameter_selection": False, "new_inference_or_training_executed": False,
        "GT_role": "POSTHOC_ONLY_AFTER_ALL110_OUTPUTS_FROZEN", "original_GT_sha256": ANNOTATIONS,
        "legal14_scenes": LEGAL_TEST, "v5_source_sha256": source, "scenes": {}, "proofs": {},
        "totals": {g: {v: Counter() for v in VARIANTS} for g in ("legal14", "invalid8_diagnostic")},
        "paired": {"legal14": {"B3_minus_B4/same_person_shared_ID": {
            "opportunities": 140, "corrections": 14, "regressions": 0, "net": 14}}, "invalid8_diagnostic": {}},
        "legal14_shared_input_strata": {
            "B3_minus_B4/geometry_reliable/same_person_shared_ID": {"opportunities": 70, "corrections": 14, "regressions": 0, "net": 14},
            "B3_minus_B4/geometry_unreliable/same_person_shared_ID": {"opportunities": 70, "corrections": 0, "regressions": 0, "net": 0}},
        "definitions": {"clean_join": "GENERATED_FIXTURE_ONLY"}}
    for v in VARIANTS:
        coverage, proofs = [], []
        for scene in range(1, 23):
            frames, candidates = (1320, 24935) if scene != 22 else (1299, 24948)
            digest = hashlib.sha256((v + "/" + str(scene)).encode()).hexdigest()
            coverage.append({"scene": scene, "frames": frames, "predictions_sha256": digest})
            proofs.append({"scene": scene, "timestamps": frames, "candidates": candidates, "tracks_sha256": digest})
        names = ["generated_scene_" + str(i) for i in LEGAL_TEST]
        official["proofs"][v] = {"coverage": coverage,
            "per_scene_TrackEval": {n: dict(metrics[v]) for n in names},
            "per_scene_MATLAB": {n: {"CVIDF1": metrics[v]["CVIDF1"]} for n in names}}
        data["proofs"][v] = proofs
    for scene in range(1, 23):
        record = {"original_GT_legal": scene in LEGAL_TEST, "counts": {}, "relative_video_quarters": {}}
        group = "legal14" if scene in LEGAL_TEST else "invalid8_diagnostic"
        for v in VARIANTS:
            counts = {"paired_timestamps": 1320 if scene != 22 else 1299,
                "frozen_candidates": 24935 if scene != 22 else 24948, "output_candidates": 24935 if scene != 22 else 24948,
                "clean_labelled_frozen_candidates": 20, "clean_labelled_candidates_retained": 20,
                "same_person_cross_view_pairs": 10, "correct_same_person_shared_ID": 2,
                "same_person_birth_pair_opportunities": 10, "simultaneous_duplicate_birth_pairs": 1,
                "different_person_cross_view_pairs": 20, "false_different_person_shared_ID": 2,
                "false_birth_unique_active_history_observations": 1,
                "wrong_historical_ID_reuse": 1, "clean_observations_into_cumulatively_mixed_ID": 2}
            if v == "B3":
                counts.update(LOST_recovery_clean_observations=4, LOST_recovery_correct_pure_identity=3)
            if scene == 1 and v == "B3":
                counts.update(clean_labelled_candidates_retained=10, clean_observations_into_cumulatively_mixed_ID=9)
            record["counts"][v] = counts
            # Generated counters exercise exact additive exposure accounting.
            record["relative_video_quarters"][v] = [
                {k: n // 4 + int(q < n % 4) for k, n in counts.items()} for q in range(4)]
            data["totals"][group][v].update(counts)
        data["scenes"][str(scene)] = record
    common = {"status": "FROZEN_TRAIN_ONLY_BUDGET", "epochs": 12, "TEST_read": False, "v5_source_sha256": source}
    gates = {}
    for v in ("B2", "B3", "B4"):
        gates[v] = {"status": "FULL_TRAIN_GATE_PASSED", "variant": v, "v5_source_sha256": source,
            "final_epochs": 12, "trainable_parameters": 606468, "seed": 20261027,
            "history_mode": "predicted_from_first_timestamp", "clip_identity_reset": False,
            "initial_state_tensor_sha256": INITIAL, "checkpoint_sha256": official["checkpoint_SHA256"][v],
            "final_scenes": list(range(1, 25)), "final_timestamps": 28754,
            "final_total_training_timestamps": 28754 * 12, "GT_in_model_inputs": False,
            "GT_repair_of_predicted_history": False,
            "final_epoch_coverage": [[{"scene": i, "timestamps": 1198 if i != 24 else 1200}
                                      for i in range(1, 25)] for _ in range(12)]}
    return data, official, gates, common, {"versus_original": {}}


class IdentitySummaryChecks(unittest.TestCase):
    def test_pending_has_no_invented_rates(self):
        value = build_identity_summary(None, {}, {}, None, {})
        self.assertEqual(value["status"], PENDING)
        self.assertEqual(value["rates"], [])
        self.assertIn("空表不代表零错误", render_identity_analysis(value))

    def test_zero_opportunities_are_unmeasured(self):
        values = {r["event"]: r for r in rates({})}
        self.assertIsNone(values["LOST_pure_identity_recovery"]["value"])
        self.assertEqual(values["LOST_pure_identity_recovery"]["denominator"], 0)

    def test_weighted_exposure_and_conditional_recovery(self):
        result = build_identity_summary(*fixture())
        self.assertEqual(result["status"], COMPLETE)
        values = {r["event"]: r for r in result["rates"] if r["scope"] == "legal14" and r["method"] == "B3"}
        self.assertAlmostEqual(values["cumulatively_mixed_ID_output"]["value"], 100 * 35 / 270)
        self.assertEqual(values["LOST_pure_identity_recovery"]["value"], 75.)
        self.assertEqual(values["duplicate_birth_pair"]["value"], 10.)
        self.assertFalse(result["B3_vs_B4_three_primary_metrics_positive"])
        # Positive diagnostic net cannot override negative native metrics.
        self.assertGreater(next(r["net"] for r in result["contrasts"] if r["stratum"] == "all"), 0)

    def test_noncanonical_or_incomplete_native_results_rejected(self):
        args = fixture()
        args[1]["canonical_TEST_scope_complete_variants"].remove("B4")
        with self.assertRaisesRegex(ValueError, "ALL110"):
            build_identity_summary(*args)

    def test_different_prediction_bytes_rejected(self):
        args = fixture()
        args[0]["proofs"]["B3"][0]["tracks_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "SAME_ALL22"):
            build_identity_summary(*args)

    def test_wrong_common_budget_and_missing_full24_coverage_rejected(self):
        for mutate in (lambda args: args[2]["B4"].update(final_epochs=16),
                       lambda args: args[2]["B3"]["final_epoch_coverage"][0].pop()):
            args = fixture()
            mutate(args)
            with self.assertRaisesRegex(ValueError, "FULL24"):
                build_identity_summary(*args)

    def test_geometry_partition_and_quarter_corruption_rejected(self):
        args = fixture()
        args[0]["legal14_shared_input_strata"]["B3_minus_B4/geometry_reliable/same_person_shared_ID"]["opportunities"] += 1
        with self.assertRaisesRegex(ValueError, "PARTITION"):
            build_identity_summary(*args)
        args = fixture()
        args[0]["scenes"]["1"]["relative_video_quarters"]["B3"][0]["frozen_candidates"] += 1
        with self.assertRaisesRegex(ValueError, "QUARTER"):
            build_identity_summary(*args)

    def test_generated_report_keeps_csv_denominators_and_limits(self):
        summary = build_identity_summary(*fixture())
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_identity_summary(summary, root, root)
            text = (root / "V5_FULL_TEST_IDENTITY_ANALYSIS.md").read_text()
            self.assertIn("不是所有可恢复目标的召回率", text)
            self.assertIn("不能识别具体是哪一路观测写入", text)
            self.assertIn("N/A", text)
            self.assertIn("denominator", (root / "v5_identity_event_rates.csv").read_text().splitlines()[0])
            self.assertTrue((root / "v5_identity_video_quarters.csv").exists())


if __name__ == "__main__":
    unittest.main()
