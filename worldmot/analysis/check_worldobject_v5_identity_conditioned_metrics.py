"""Generated boxes demonstrate identity-conditioned native matching counts.

No VisionTrack image, annotation or prediction is read. This is an evaluator
mechanism check, not an additional tracking model or a TEST event attribution.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

from worldmot.data.worldobject_cache import sha256
from worldmot.runtime.world_object_v4_test_contract import install_no_gt_guard
from worldmot.runtime.world_object_v5_contract import atomic_json, v5_source_fingerprint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    before = v5_source_fingerprint()
    guard = install_no_gt_guard()
    dependency = args.repo / "TrackEval"
    expected = {
        "trackeval/metrics/clear.py": "349daafdbb009f6cd71cf66ff1c9b8074940cd3310fcd663151d644b7ff0e5e3",
        "trackeval/metrics/hota.py": "421cb37dd492ecf2591114ea7e4a0d006cd0401225551717fd0b50473c8505d5"}
    if any(sha256(dependency / p) != digest for p, digest in expected.items()):
        raise ValueError("USE_THE_SAME_NATIVE_METRIC_IMPLEMENTATION_AS_FORMAL_TEST")
    sys.path.insert(0, str(dependency))
    from trackeval.metrics import CLEAR, HOTA
    from trackeval.datasets._base_dataset import _BaseDataset
    for cls, name in ((CLEAR, "clear.py"), (HOTA, "hota.py")):
        module_file = Path(sys.modules[cls.__module__].__file__).resolve()
        if module_file != (dependency / "trackeval/metrics" / name).resolve():
            raise ValueError("WRONG_IMPORTED_NATIVE_METRIC_MODULE")

    truth = [np.array([[0., 0., 10., 10.], [4., 0., 14., 10.]]) for _ in range(2)]
    predictions = [truth[0].copy(), np.array([[2., 0., 12., 10.], [-2., 0., 8., 10.]])]
    similarity = [_BaseDataset._calculate_box_ious(g, p, box_format="x0y0x1y1")
                  for g, p in zip(truth, predictions)]
    base = {"num_timesteps": 2, "num_gt_ids": 2, "num_tracker_ids": 2,
            "num_gt_dets": 4, "num_tracker_dets": 4,
            "gt_ids": [np.array([0, 1]), np.array([0, 1])], "similarity_scores": similarity}
    cases = {}
    for name, second_ids in (("fixed_box_order_ids", [0, 1]), ("swapped_ids_at_second_time", [1, 0])):
        data = deepcopy(base)
        data["tracker_ids"] = [np.array([0, 1]), np.array(second_ids)]
        clear = CLEAR({"PRINT_CONFIG": False}).eval_sequence(data)
        hota = HOTA().eval_sequence(data)
        if any(not np.array_equal(x, y) for x, y in zip(similarity, data["similarity_scores"])):
            raise ValueError("ONLY_TRACKER_IDENTITIES_MAY_CHANGE_BETWEEN_CASES")
        cases[name] = {"tracker_ids_by_time": [x.tolist() for x in data["tracker_ids"]],
            "CLEAR": {k: int(clear[k]) for k in ("CLR_TP", "CLR_FP", "CLR_FN", "IDSW")},
            "HOTA_DetA_percent": float(100 * np.mean(hota["DetA"])),
            "HOTA_percent": float(100 * np.mean(hota["HOTA"]))}
    fixed, swapped = cases["fixed_box_order_ids"], cases["swapped_ids_at_second_time"]
    if fixed["CLEAR"] != {"CLR_TP": 3, "CLR_FP": 1, "CLR_FN": 1, "IDSW": 0}:
        raise ValueError("EXPECTED_CONTINUITY_PREFERENCE_CHANGES_VALID_MATCH_COUNT")
    if swapped["CLEAR"] != {"CLR_TP": 4, "CLR_FP": 0, "CLR_FN": 0, "IDSW": 0}:
        raise ValueError("EXPECTED_SAME_BOXES_DIFFERENT_IDENTITIES_GIVE_FOUR_MATCHES")
    if not np.allclose([fixed["HOTA_DetA_percent"], swapped["HOTA_DetA_percent"]],
                       [62.10526315789475, 78.94736842105264], rtol=0., atol=1e-10):
        raise ValueError("HOTA_GLOBAL_IDENTITY_ALIGNMENT_ALSO_CHANGES_DETA")
    if guard["blocked_GT_reads"] or v5_source_fingerprint() != before:
        raise ValueError("REAL_GT_READ_ATTEMPT_OR_MODEL_SOURCE_CHANGED")
    proof = {"status": "PASSED_GENERATED_IDENTITY_CONDITIONED_MATCHING_CHECK",
        "UTC": datetime.now(timezone.utc).isoformat(), "source_sha256": sha256(__file__),
        "native_metric_sha256": expected, "v5_source_sha256": before,
        "generated_timesteps": 2, "generated_cameras": 1,
        "generated_GT_boxes_xyxy": [x.tolist() for x in truth],
        "identical_prediction_boxes_xyxy_in_both_cases": [x.tolist() for x in predictions],
        "identical_IoU_matrices": [x.tolist() for x in similarity], "cases": cases,
        "real_VisionTrack_data_or_predictions_read": False, "blocked_GT_reads": guard["blocked_GT_reads"],
        "neural_models_run": False, "formal_TEST_protocol_changed": False,
        "actual_seven_B0_B1_TP_difference_events_identified": False,
        "scientific_tracking_result": False,
        "conclusion": "Native CLEAR TP/FP/FN and HOTA DetA are conditioned on identity matching; changes alone do not prove a detector changed."}
    atomic_json(args.output, proof)
    print(json.dumps({"status": proof["status"], "cases": cases}), flush=True)


if __name__ == "__main__":
    main()
