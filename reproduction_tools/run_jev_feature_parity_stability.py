"""Repeat corrected OFF feature parity on one fixed GPU.

The experiment is deliberately independent of policy training.  It replays
the same corrected records three times with the same checkpoint, cache, trace,
seed policy, and visible GPU, then records full and per-feature error
statistics.  The result freezes the runtime float tolerance only after all
three repetitions satisfy the exact structural gates.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = Path("/data1/liuyeqiang/WWW/outputs/stage2_single_gpu/model_20000.pth")
CACHE = Path("/data1/liuyeqiang/WWW/outputs/research_final_v2/off/perception_cache_train")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def parity_template(feature_names, expected_count: int, tolerance: float):
    return {
        "schema_version": "jev_runtime_feature_parity_stability_v1",
        "tolerance": float(tolerance),
        "feature_names": list(feature_names(64)),
        "expected_record_count": int(expected_count),
        "compared_records": 0,
        "finite_runtime_records": 0,
        "max_abs_error": 0.0,
        "sum_abs_error": 0.0,
        "per_feature_max_abs_error": [0.0] * 64,
        "per_feature_sum_abs_error": [0.0] * 64,
        "per_feature_count": [0] * 64,
        "seen_record_keys": set(),
        "collect_error_values": True,
        "error_values": [],
        "per_feature_error_values": [[] for _ in range(64)],
    }


def finite_tree(value: Any) -> bool:
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(item) for item in value)
    if isinstance(value, (int, float)):
        return np.isfinite(float(value))
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-id", type=int, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-frame", type=int, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--tolerance", type=float, default=2e-5)
    args = parser.parse_args()
    if args.repetitions < 3:
        raise ValueError("at least three repetitions are required")
    for path in (args.trace, args.records, CHECKPOINT):
        if not path.is_file():
            raise FileNotFoundError(path)

    os.environ["JEV_VIDEO_ID"] = str(int(args.video_id))
    os.environ["JEV_TRACE_PATH"] = str(args.trace.resolve())
    os.environ["JEV_RECORDS_PATH"] = str(args.records.resolve())
    os.environ["JEV_PILOT_ROOT"] = str(args.runtime_root.resolve())
    os.environ["JEV_OFFLINE_ROOT"] = str(args.runtime_root.resolve() / "offline")
    os.environ.setdefault(
        "PYTHONPATH",
        ":".join(
            (
                str(ROOT),
                str(ROOT / "reproduction_tools"),
                str(ROOT / "third_party" / "CenterNet2"),
            )
        ),
    )

    import sys

    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "reproduction_tools"))
    sys.path.insert(0, str(ROOT / "third_party" / "CenterNet2"))
    import run_early_pilot_tracking as pilot

    _annotations, _subset, image_lookup, by_key, all_records = pilot.load_inputs()
    records = {
        key: value
        for key, value in all_records.items()
        if int(value["state"]["online_context"].get("frame", -1)) <= int(args.max_frame)
    }
    if not records:
        raise RuntimeError("the selected corrected subset contains no records")
    modules = pilot.load_runtime_modules()
    (
        build_formal_gmt_engine,
        MutableGMTState,
        FrozenPerceptionCache,
        JEVRuntimePolicy,
        build_controller_from_checkpoint,
        build_state_features,
        association_window_length,
        candidate_entropy,
        count_memory_observations,
        count_track_history,
        feature_names,
        legacy_acceptance_threshold,
    ) = modules

    repetitions = []
    for repetition in range(int(args.repetitions)):
        parity = parity_template(feature_names, len(records), args.tolerance)
        item = pilot.run_method(
            "gmt_off",
            None,
            image_lookup=image_lookup,
            by_key=by_key,
            records=records,
            build_formal_gmt_engine=build_formal_gmt_engine,
            MutableGMTState=MutableGMTState,
            FrozenPerceptionCache=FrozenPerceptionCache,
            JEVRuntimePolicy=JEVRuntimePolicy,
            build_controller_from_checkpoint=build_controller_from_checkpoint,
            build_state_features=build_state_features,
            association_window_length=association_window_length,
            candidate_entropy=candidate_entropy,
            count_memory_observations=count_memory_observations,
            count_track_history=count_track_history,
            legacy_acceptance_threshold=legacy_acceptance_threshold,
            feature_source_mode="runtime",
            parity_report=parity,
            device=args.device,
            max_frame=args.max_frame,
        )
        finalized = pilot.finalize_feature_parity(parity, records, item["counts"])
        all_errors = np.asarray(parity.pop("error_values", []), dtype=np.float64)
        per_feature_values = parity.pop("per_feature_error_values", [[] for _ in range(64)])
        per_feature = []
        for name, values in zip(parity["feature_names"], per_feature_values):
            values_array = np.asarray(values, dtype=np.float64)
            per_feature.append(
                {
                    "name": name,
                    "count": int(values_array.size),
                    "max_abs_error": float(values_array.max(initial=0.0)),
                    "mean_abs_error": float(values_array.mean()) if values_array.size else None,
                    "p99_abs_error": float(np.percentile(values_array, 99)) if values_array.size else None,
                }
            )
        repetitions.append(
            {
                "repetition": repetition + 1,
                "status": "PASS" if finalized["pass"] else "FAIL",
                "compared_records": finalized["compared_records"],
                "expected_records": finalized["expected_record_count"],
                "missing_records": finalized["missing_record_count"],
                "finite_runtime_records": finalized["finite_runtime_records"],
                "off_action_mismatches": finalized["off_action_mismatches"],
                "max_abs_error": finalized["max_abs_error"],
                "mean_abs_error": finalized["mean_abs_error"],
                "p99_abs_error": float(np.percentile(all_errors, 99)) if all_errors.size else None,
                "error_count": int(all_errors.size),
                "per_feature": per_feature,
                "feature_mismatch_examples": finalized.get(
                    "feature_mismatch_examples", []
                ),
                "off_action_mismatch_examples": finalized.get(
                    "off_action_mismatch_examples", []
                ),
                "runtime_method_counts": item["counts"],
            }
        )

    report = {
        "schema_version": "jev_runtime_feature_parity_stability_v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "video_id": int(args.video_id),
        "subset_max_frame": int(args.max_frame),
        "device": str(args.device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "repetitions_requested": int(args.repetitions),
        "tolerance_tested": float(args.tolerance),
        "tolerance_policy_candidate": "FLOAT32_RUNTIME_FEATURE_PARITY_TOLERANCE",
        "seed_policy": {
            "trajectory_rng_master_seed": 20261006,
            "video_rng_seed": 20261006 + int(args.video_id),
            "policy_seed": 20261003,
        },
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": sha256(CHECKPOINT),
        "trace": str(args.trace.resolve()),
        "trace_sha256": sha256(args.trace),
        "records": str(args.records.resolve()),
        "records_sha256": sha256(args.records),
        "cache": str(CACHE),
        "cache_exists": CACHE.is_dir(),
        "same_inputs_each_run": True,
        "structural_exact_requirements": {
            "record_count": "exact",
            "missing_count": 0,
            "finite_count": "exact",
            "off_action_mismatches": 0,
            "semantic_keys": "exact through finalize_feature_parity",
        },
        "repetitions": repetitions,
        "all_repetitions_pass": all(item["status"] == "PASS" for item in repetitions),
        "recommended_frozen_tolerance": float(args.tolerance)
        if all(item["status"] == "PASS" for item in repetitions)
        else None,
        "status": "PASS"
        if all(item["status"] == "PASS" for item in repetitions)
        else "FAIL",
        "classification": "RUNTIME_NUMERICAL_STABILITY_GATE_NOT_TRACKING_RESULT",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "repetitions": len(repetitions), "tolerance": args.tolerance}, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
