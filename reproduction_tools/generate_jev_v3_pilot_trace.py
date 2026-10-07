"""Generate a provenance-carrying OFF trace from a frozen cache video.

This is a bounded v3 pilot artifact.  It does not touch the live Full H=8
builders or their output directory.  The replay uses the formal GMT
association transformer and the canonical runtime state builder.  The
trajectory-slot RNG is branch-local state initialized from a stable per-video
master seed, independent of the policy-training seed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
PILOT_ROOT = Path("/home/liuyeqiang/WWW_jev_full_h8_runtime/formal_current_head_off_trace")
POLICY_SEED = 20261003


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def image_lookup_for_video(annotation_path: Path, video_id: int):
    annotations = json.loads(annotation_path.read_text(encoding="utf-8"))
    lookup = {}
    for image in annotations["images"]:
        if int(image.get("video_id", -1)) != int(video_id):
            continue
        lookup[(int(video_id), int(image["view_id"]), int(image["frame_id"]))] = image
    if not lookup:
        raise RuntimeError(f"no annotation images found for video {video_id}")
    return lookup


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=PILOT_ROOT)
    parser.add_argument("--video-id", type=int, default=8)
    parser.add_argument(
        "--seed",
        type=int,
        default=POLICY_SEED,
        help="policy-seed metadata only; never used for association RNG",
    )
    parser.add_argument("--max-frame", type=int, default=None)
    parser.add_argument(
        "--device",
        default="cpu",
        help="formal replay device, e.g. cuda:2; the device is recorded in the manifest",
    )
    parser.add_argument(
        "--compare-trace",
        type=Path,
        help=(
            "optional legacy/current trace used only as an action-layout "
            "comparison; the generated trace always follows the current replay"
        ),
    )
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    trace_path = output_root / f"trace_video_{int(args.video_id):02d}.jsonl"
    if trace_path.exists():
        raise RuntimeError(f"refusing to overwrite existing v3 trace: {trace_path}")

    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "third_party" / "CenterNet2"))
    sys.path.insert(0, str(ROOT / "reproduction_tools"))
    import run_early_pilot_tracking as pilot
    from gtr.modeling.jev_runtime import DecisionTraceWriter

    # ``run_method`` resolves the cache/video coordinates through these
    # module-level values. Set them explicitly so this generator cannot
    # silently emit a video08 trace when asked to audit another partition.
    pilot.VIDEO_ID = int(args.video_id)
    pilot.TRACE = trace_path

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
    cache = FrozenPerceptionCache(pilot.CACHE)
    image_lookup = image_lookup_for_video(pilot.ANNOTATIONS, args.video_id)
    output_root.mkdir(parents=True, exist_ok=False)
    comparison_by_key = {}
    comparison_trace_sha256 = None
    comparison_trace_events = 0
    if args.compare_trace is not None:
        comparison_trace = args.compare_trace.resolve()
        if not comparison_trace.is_file():
            raise FileNotFoundError(comparison_trace)
        # Reuse the production parser only for the optional action-layout
        # comparison.  /dev/null keeps the parser from requiring policy
        # records; the generated trace itself is still driven by an empty
        # record map and the current formal replay.
        previous_trace = pilot.TRACE
        previous_records = pilot.RECORDS
        try:
            pilot.TRACE = comparison_trace
            pilot.RECORDS = Path("/dev/null")
            _annotations, _subset, _lookup, comparison_by_key, _records = pilot.load_inputs()
        finally:
            pilot.TRACE = previous_trace
            pilot.RECORDS = previous_records
        comparison_trace_sha256 = "sha256:" + sha256(comparison_trace)
        comparison_trace_events = sum(
            1 for line in comparison_trace.open(encoding="utf-8") if line.strip()
        )
    parity_report = {
        "schema_version": "jev_runtime_state_contract_v3",
        "source_trace": str(trace_path),
        "tolerance": 1e-6,
        "feature_names": list(feature_names(64)),
        "expected_record_count": 0,
        "compared_records": 0,
        "finite_runtime_records": 0,
        "max_abs_error": 0.0,
        "sum_abs_error": 0.0,
        "per_feature_max_abs_error": [0.0] * 64,
        "per_feature_sum_abs_error": [0.0] * 64,
        "per_feature_count": [0] * 64,
        "seen_record_keys": set(),
    }
    previous_pilot_root = pilot.PILOT
    pilot.PILOT = output_root
    try:
        with DecisionTraceWriter(trace_path) as trace_writer:
            result = pilot.run_method(
                "gmt_off",
                None,
                image_lookup=image_lookup,
                by_key=comparison_by_key,
                records={},
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
                parity_report=parity_report,
                device=str(args.device),
                max_frame=args.max_frame,
                trace_writer=trace_writer,
            )
    finally:
        pilot.PILOT = previous_pilot_root

    event_count = sum(1 for line in trace_path.open(encoding="utf-8") if line.strip())
    expected_event_count = sum(
        int(value)
        for key, value in result["counts"].items()
        if key.endswith("_DECISION")
    )
    if event_count != expected_event_count:
        raise RuntimeError("trace event count does not match runtime decision count")
    manifest = {
        "status": "COMPLETE",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "video_id": int(args.video_id),
        "trace": str(trace_path),
        "trace_sha256": "sha256:" + sha256(trace_path),
        "trace_events": event_count,
        "max_frame": args.max_frame,
        "source_branch": subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "checkpoint": str(pilot.CHECKPOINT),
        "checkpoint_sha256": "sha256:" + sha256(pilot.CHECKPOINT),
        "device": str(args.device),
        "perception_cache": str(pilot.CACHE),
        "perception_cache_index_sha256": "sha256:" + sha256(pilot.CACHE / "index.jsonl"),
        "association_backend": "formal_gmt_transformer",
        "policy_seed": int(args.seed),
        "trajectory_slot_rng": {
            "mode": "branch_local_explicit_python_random_v1",
            "master_seed": 20261006,
            "video_seed": 20261006 + int(args.video_id),
            "state_initialized_per_video": True,
            "state_cloned_per_counterfactual_branch": True,
            "recorded_in_every_event_context": True,
            "replay_requirement": "restore state.getstate() into a fresh branch-local RNG before each formal proposal",
        },
        "proposal_reused_across_legal_actions": True,
        "reassociate_reuses_score_matrix": True,
        "second_transformer_call_for_reassociate": False,
        "transformer_sha256": "sha256:" + sha256(
            ROOT / "gtr" / "modeling" / "roi_heads" / "transformer.py"
        ),
        "counterfactual_engine_sha256": "sha256:" + sha256(
            ROOT / "reproduction_tools" / "jev_counterfactual_v2.py"
        ),
        "adapter_sha256": "sha256:" + sha256(
            ROOT / "reproduction_tools" / "jev_gmt_association_adapter.py"
        ),
        "runtime_state_source": "gtr/modeling/jev_state.py",
        "runtime_counts": result["counts"],
        "comparison_trace": str(args.compare_trace.resolve())
        if args.compare_trace is not None
        else None,
        "comparison_trace_sha256": comparison_trace_sha256,
        "comparison_trace_events": comparison_trace_events,
        "comparison_trace_is_reference_only": args.compare_trace is not None,
        "comparison_off_action_mismatches": int(
            result["counts"].get("off_action_mismatches", 0)
        ),
        "classification": "SCREENING_ONLY_NOT_FOR_FINAL_SELECTION_NOT_FOR_PAPER_RESULT",
    }
    (output_root / "TRACE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
