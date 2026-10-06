"""Compare native GMT and mutable-replay reactivation candidates.

The native side is a bounded OFF trace produced by ``GTRRCNN`` with the
diagnostic candidate fields enabled.  The replay side is generated from the
same immutable perception cache and the same OFF trace context, but uses only
the mutable replay state.  No GT or future-frame value is read by this gate.

This is intentionally a hard diagnostic gate.  A missing native candidate
field or a mismatched event is a failure, not an invitation to compare only
the events that happen to overlap.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc


def read_json_value(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON at {path}: {exc}") from exc


def event_key(payload: Mapping[str, Any]):
    context = payload.get("context", payload)
    detection_index = context.get("detection_index")
    if detection_index is None:
        # Native trace records call this field ``detection_index`` while the
        # replay decision schema calls the same within-frame position ``row``.
        # Both are produced by the same frozen detection ordering and are
        # required for an exact event join; do not fall back to frame alone.
        detection_index = context.get("row")
    if detection_index is None:
        raise ValueError(
            "reactivation event is missing detection_index/row: "
            f"frame={context.get('frame')!r}, view={context.get('view')!r}"
        )
    return (
        int(context.get("frame")),
        int(context.get("view")),
        int(detection_index),
    )


def finite_list(values):
    return isinstance(values, list) and all(
        isinstance(value, (int, float)) and math.isfinite(float(value))
        for value in values
    )


def native_events(path: Path, video_id: int, max_frame: int):
    events = {}
    duplicate_keys = []
    for payload in read_jsonl(path):
        context = payload.get("context", {})
        if context.get("decision_scope") != "reactivation":
            continue
        if int(context.get("video_id", -1)) != int(video_id):
            continue
        if int(context.get("frame", -1)) > int(max_frame):
            continue
        key = event_key(payload)
        if key in events:
            duplicate_keys.append(list(key))
        events[key] = payload
    return events, duplicate_keys


def replay_events(path: Path):
    events = {}
    duplicate_keys = []
    payloads = read_json_value(path)
    if not isinstance(payloads, list):
        raise ValueError(f"replay decisions must be a JSON list: {path}")
    for payload in payloads:
        if not isinstance(payload, Mapping):
            raise ValueError(f"replay decision is not an object: {path}")
        if payload.get("question") != "REACTIVATION_DECISION":
            continue
        key = event_key(payload)
        if key in events:
            duplicate_keys.append(list(key))
        events[key] = payload
    return events, duplicate_keys


def compare(native, replay, *, tolerance: float):
    mismatches = []
    native_keys = set(native)
    replay_keys = set(replay)
    for key in sorted(native_keys | replay_keys):
        if key not in native:
            mismatches.append({"key": list(key), "reason": "missing_native_event"})
            continue
        if key not in replay:
            mismatches.append({"key": list(key), "reason": "missing_replay_event"})
            continue

        n = native[key].get("context", {})
        r = replay[key]
        native_ids = n.get("native_candidate_track_ids")
        native_scores = n.get("native_candidate_scores")
        replay_ids = r.get("candidate_track_ids")
        replay_scores = r.get("candidate_scores")
        reasons = []
        if not isinstance(native_ids, list):
            reasons.append("native_candidate_ids_missing")
        if not finite_list(native_scores):
            reasons.append("native_candidate_scores_missing_or_nonfinite")
        if not isinstance(replay_ids, list):
            reasons.append("replay_candidate_ids_missing")
        if not finite_list(replay_scores):
            reasons.append("replay_candidate_scores_missing_or_nonfinite")
        if not reasons:
            if [int(value) for value in native_ids] != [int(value) for value in replay_ids]:
                reasons.append("candidate_ids_or_order_mismatch")
            if len(native_scores) != len(replay_scores):
                reasons.append("candidate_score_count_mismatch")
            else:
                max_score_error = max(
                    (abs(float(left) - float(right)) for left, right in zip(native_scores, replay_scores)),
                    default=0.0,
                )
                if max_score_error > float(tolerance):
                    reasons.append("candidate_scores_outside_tolerance")
            if int(n.get("native_candidate_count", len(native_ids))) != len(replay_ids):
                reasons.append("candidate_count_mismatch")
        else:
            max_score_error = None

        native_track = n.get("track_id")
        replay_track = r.get("track_id")
        if native_track is not None and replay_track is not None:
            if int(native_track) != int(replay_track):
                reasons.append("chosen_proposal_id_mismatch")
        else:
            reasons.append("chosen_proposal_id_missing")

        # The trace writer stores the legacy OFF action at the event's
        # top-level field; replay decisions use the same field.  It is not
        # part of ``context`` (which contains the native proposal details).
        native_off = native[key].get("off_action") or native[key].get("proposed_action")
        replay_off = r.get("off_action")
        if native_off != replay_off:
            reasons.append("legacy_off_action_mismatch")

        native_threshold = n.get("native_bank_threshold")
        replay_threshold = r.get("bank_threshold")
        threshold_error = None
        if native_threshold is None or replay_threshold is None:
            reasons.append("bank_threshold_missing")
        else:
            threshold_error = abs(float(native_threshold) - float(replay_threshold))
            if threshold_error > float(tolerance):
                reasons.append("bank_threshold_mismatch")

        if reasons:
            item = {
                "key": list(key),
                "reasons": reasons,
                "native_candidate_track_ids": native_ids,
                "replay_candidate_track_ids": replay_ids,
                "native_candidate_scores": native_scores,
                "replay_candidate_scores": replay_scores,
                "native_track_id": native_track,
                "replay_track_id": replay_track,
                "native_off_action": native_off,
                "replay_off_action": replay_off,
                "native_bank_threshold": native_threshold,
                "replay_bank_threshold": replay_threshold,
                "max_score_error": max_score_error,
                "threshold_error": threshold_error,
            }
            if len(mismatches) < 100:
                mismatches.append(item)

    return mismatches


def run_replay(args: argparse.Namespace) -> Path:
    replay_root = args.replay_root.resolve()
    replay_root.mkdir(parents=True, exist_ok=True)
    os.environ["JEV_VIDEO_ID"] = str(int(args.video_id))
    os.environ["JEV_TRACE_PATH"] = str(args.native_trace.resolve())
    os.environ["JEV_RECORDS_PATH"] = str(args.records.resolve())
    os.environ["JEV_PILOT_ROOT"] = str(replay_root)
    os.environ["JEV_OFFLINE_ROOT"] = str(replay_root / "offline")

    import sys

    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "reproduction_tools"))
    sys.path.insert(0, str(ROOT / "third_party" / "CenterNet2"))
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
        _feature_names,
        legacy_acceptance_threshold,
    ) = modules
    parity = {
        "feature_names": list(_feature_names(64)),
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
    return Path(item["decisions"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-id", type=int, required=True)
    parser.add_argument("--native-trace", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-frame", type=int, required=True)
    parser.add_argument("--tolerance", type=float, default=2e-5)
    args = parser.parse_args()
    for path in (args.native_trace, args.records):
        if not path.is_file():
            raise FileNotFoundError(path)

    replay_path = run_replay(args)
    native, native_duplicates = native_events(
        args.native_trace, args.video_id, args.max_frame
    )
    replay, replay_duplicates = replay_events(replay_path)
    mismatches = compare(native, replay, tolerance=args.tolerance)
    report = {
        "schema_version": "jev_reactivation_candidate_parity_v1",
        "status": "PASS"
        if not native_duplicates
        and not replay_duplicates
        and len(native) == len(replay)
        and not mismatches
        else "FAIL",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "classification": "BOUNDED_RUNTIME_SEMANTIC_GATE_NOT_TRACKING_RESULT",
        "video_id": int(args.video_id),
        "max_frame": int(args.max_frame),
        "native_trace": str(args.native_trace.resolve()),
        "native_trace_sha256": sha256_file(args.native_trace),
        "replay_records": str(args.records.resolve()),
        "replay_records_sha256": sha256_file(args.records),
        "replay_decisions": str(replay_path.resolve()),
        "tolerance": float(args.tolerance),
        "native_candidate_source": "native_trace.context.native_candidate_*",
        "replay_candidate_source": "mutable_state.reactivation_proposal",
        "native_reactivation_decision_count": len(native),
        "replay_reactivation_decision_count": len(replay),
        "missing_native_count": len(set(replay) - set(native)),
        "missing_replay_count": len(set(native) - set(replay)),
        "native_duplicate_keys": native_duplicates,
        "replay_duplicate_keys": replay_duplicates,
        "native_candidate_fields_present": all(
            isinstance(payload.get("context", {}).get("native_candidate_track_ids"), list)
            and isinstance(payload.get("context", {}).get("native_candidate_scores"), list)
            for payload in native.values()
        ),
        "candidate_ids_exact": not any(
            "candidate_ids_or_order_mismatch" in item.get("reasons", [])
            for item in mismatches
        ),
        "candidate_scores_within_tolerance": not any(
            "candidate_scores_outside_tolerance" in item.get("reasons", [])
            for item in mismatches
        ),
        "chosen_proposal_ids_exact": not any(
            "chosen_proposal_id_mismatch" in item.get("reasons", [])
            for item in mismatches
        ),
        "legacy_off_actions_exact": not any(
            "legacy_off_action_mismatch" in item.get("reasons", [])
            for item in mismatches
        ),
        "bank_thresholds_within_tolerance": not any(
            "bank_threshold_mismatch" in item.get("reasons", [])
            for item in mismatches
        ),
        "mismatch_count_capped": len(mismatches),
        "mismatches": mismatches,
        "formal_gate": {
            "candidate_id_set_exact": False,
            "candidate_order_exact_or_canonicalized": False,
            "candidate_scores_within_frozen_tolerance": False,
            "chosen_proposal_ids_exact": False,
            "legacy_off_action_exact": False,
            "reactivation_decision_count_exact": False,
            "missing_reactivation_records": 0,
        },
    }
    report["formal_gate"].update(
        {
            "candidate_id_set_exact": bool(report["candidate_ids_exact"]),
            "candidate_order_exact_or_canonicalized": bool(report["candidate_ids_exact"]),
            "candidate_scores_within_frozen_tolerance": bool(report["candidate_scores_within_tolerance"]),
            "chosen_proposal_ids_exact": bool(report["chosen_proposal_ids_exact"]),
            "legacy_off_action_exact": bool(report["legacy_off_actions_exact"]),
            "reactivation_decision_count_exact": len(native) == len(replay),
            "missing_reactivation_records": int(report["missing_native_count"] + report["missing_replay_count"]),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "native": len(native), "replay": len(replay), "mismatches": len(mismatches)}, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
