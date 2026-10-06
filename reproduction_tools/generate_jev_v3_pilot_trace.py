"""Generate a provenance-carrying OFF trace from the frozen video08 cache.

This is a bounded v3 pilot artifact.  It does not touch the live Full H=8
builders or their output directory.  The replay uses the formal GMT
association transformer, the canonical runtime state builder, and an
explicit Python trajectory-slot seed.  Every event records that seed so a
fresh process can reproduce the same association RNG stream.
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
PILOT_ROOT = Path("/home/liuyeqiang/WWW_jev_full_h8_runtime/pilot_v3")
SEED = 20261003


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
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-frame", type=int, default=None)
    args = parser.parse_args()
    if int(args.video_id) != 8:
        raise ValueError("the v3 pilot currently supports only held-out video08")
    output_root = args.output_root.resolve()
    trace_path = output_root / "trace_video_08.jsonl"
    if trace_path.exists():
        raise RuntimeError(f"refusing to overwrite existing v3 trace: {trace_path}")

    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "third_party" / "CenterNet2"))
    sys.path.insert(0, str(ROOT / "reproduction_tools"))
    import run_early_pilot_tracking as pilot
    from gtr.modeling.jev_runtime import DecisionTraceWriter

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
    previous_seed = pilot.SEED
    pilot.PILOT = output_root
    pilot.SEED = int(args.seed)
    try:
        with DecisionTraceWriter(trace_path) as trace_writer:
            result = pilot.run_method(
                "gmt_off",
                None,
                image_lookup=image_lookup,
                by_key={},
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
                max_frame=args.max_frame,
                trace_writer=trace_writer,
            )
    finally:
        pilot.PILOT = previous_pilot_root
        pilot.SEED = previous_seed

    event_count = sum(1 for line in trace_path.open(encoding="utf-8") if line.strip())
    if event_count != int(result["counts"]["MATCH_DECISION"] + result["counts"]["MEMORY_DECISION"]):
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
        "perception_cache": str(pilot.CACHE),
        "perception_cache_index_sha256": "sha256:" + sha256(pilot.CACHE / "index.jsonl"),
        "association_backend": "formal_gmt_transformer",
        "trajectory_slot_rng": {
            "mode": "process_global_python_random_seeded",
            "seed": int(args.seed),
            "recorded_in_every_event_context": True,
            "replay_requirement": "seed before model construction and before the first association proposal",
        },
        "runtime_state_source": "gtr/modeling/jev_state.py",
        "runtime_counts": result["counts"],
        "classification": "SCREENING_ONLY_NOT_FOR_FINAL_SELECTION_NOT_FOR_PAPER_RESULT",
    }
    (output_root / "TRACE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
