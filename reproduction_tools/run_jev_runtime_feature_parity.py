"""Compare canonical H=8 record features with the live mutated-state replay.

This wrapper calls the same runtime implementation used by the closed-loop
pilot, but writes only a standalone diagnostic report. It never changes the
repository's V3 pilot reports and it does not run a learned controller.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-id", type=int, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-frame", type=int)
    args = parser.parse_args()

    pilot_root = args.output.resolve().parent / (args.output.stem + "_runtime")
    os.environ["JEV_VIDEO_ID"] = str(int(args.video_id))
    os.environ["JEV_TRACE_PATH"] = str(args.trace.resolve())
    os.environ["JEV_RECORDS_PATH"] = str(args.records.resolve())
    os.environ["JEV_PILOT_ROOT"] = str(pilot_root)
    os.environ["JEV_OFFLINE_ROOT"] = str(pilot_root / "offline")
    os.environ["JEV_PILOT_DEVICE"] = str(args.device)
    os.environ.setdefault("PYTHONPATH", ":".join((str(ROOT), str(ROOT / "reproduction_tools"), str(ROOT / "third_party/CenterNet2"))))

    import sys

    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "reproduction_tools"))
    sys.path.insert(0, str(ROOT / "third_party/CenterNet2"))
    import run_early_pilot_tracking as pilot

    _annotations, _subset, image_lookup, by_key, records = pilot.load_inputs()
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
    parity = {
        "schema_version": "jev_runtime_feature_parity_v4",
        "source_trace": str(args.trace.resolve()),
        "tolerance": 1e-4,
        "feature_names": list(feature_names(64)),
        "expected_record_count": len(records),
        "compared_records": 0,
        "finite_runtime_records": 0,
        "max_abs_error": 0.0,
        "sum_abs_error": 0.0,
        "per_feature_max_abs_error": [0.0] * 64,
        "per_feature_sum_abs_error": [0.0] * 64,
        "per_feature_count": [0] * 64,
        "seen_record_keys": set(),
    }
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
    if args.max_frame is None:
        parity = pilot.finalize_feature_parity(
            parity,
            records,
            item["counts"],
        )
    else:
        parity["status"] = "BOUNDED_PASS" if parity["max_abs_error"] <= parity["tolerance"] else "BOUNDED_FAIL"
        parity["pass"] = parity["max_abs_error"] <= parity["tolerance"]
        parity.pop("seen_record_keys", None)
    report = {
        "schema_version": "jev_runtime_feature_parity_v4",
        "status": "PASS" if parity.get("pass") else "FAIL",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "video_id": int(args.video_id),
        "records": str(args.records.resolve()),
        "trace": str(args.trace.resolve()),
        "device": str(args.device),
        "max_frame": args.max_frame,
        "classification": "CANONICAL_FEATURE_PARITY_DIAGNOSTIC_NOT_TRACKING_RESULT",
        "parity": parity,
        "runtime_counts": item["counts"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "video_id": args.video_id, "compared_records": parity["compared_records"], "max_abs_error": parity["max_abs_error"]}, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
