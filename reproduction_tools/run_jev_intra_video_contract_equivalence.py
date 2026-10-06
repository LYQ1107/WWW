"""Run a bounded CPU contract equivalence probe for chunk orchestration.

The probe uses the deterministic cosine contract backend and a bounded prefix
of an existing video. It validates snapshot boundaries, range ownership, and
exact record merging without occupying a GPU or touching the active small gate.
Formal-GMT chunk authorization still requires the same verifier on a bounded
formal replay before the 24-video rebuild.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition-manifest", type=Path, required=True)
    parser.add_argument("--video-id", type=int, default=7)
    parser.add_argument("--max-trace-lines", type=int, default=240)
    parser.add_argument("--records-per-chunk", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import sys

    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "reproduction_tools"))
    sys.path.insert(0, str(ROOT / "third_party/CenterNet2"))
    from build_jev_counterfactual_dataset import load_gt, normalize_events
    from build_jev_counterfactual_v2 import advance_off_state_for_key, event_maps, build_v2_records
    from gtr.modeling.jev_perception_cache import FrozenPerceptionCache
    from jev_counterfactual_v2 import CachedPerceptionMutableAssociationV2, MutableGMTState
    from jev_intra_video_chunking import compare_records_exact, plan_chunks

    partition = json.loads(args.partition_manifest.read_text(encoding="utf-8"))
    video_id = int(args.video_id)
    source_trace = Path(partition["trace_by_video"]) / f"video_{video_id:02d}.jsonl"
    source_order = Path(partition["trace_by_video"]) / f"video_{video_id:02d}.orders.jsonl"
    trace_lines = [line for line in source_trace.read_text(encoding="utf-8").splitlines(True) if line.strip()]
    order_lines = [line for line in source_order.read_text(encoding="utf-8").splitlines(True) if line.strip()]
    count = min(int(args.max_trace_lines), len(trace_lines), len(order_lines))
    if count < 1:
        raise ValueError("bounded trace prefix is empty")
    with tempfile.TemporaryDirectory(prefix="jev_chunk_contract_") as temporary:
        root = Path(temporary)
        trace = root / "trace.jsonl"
        order = root / "trace.orders.jsonl"
        trace.write_text("".join(trace_lines[:count]), encoding="utf-8")
        order.write_text("".join(order_lines[:count]), encoding="utf-8")
        events = normalize_events(trace, order_index=order, minimal=True)[video_id]
        actions, memories, by_key = event_maps(events)
        cache_keys = json.loads(
            (args.partition_manifest.parent / "cache_keys_by_video.json").read_text(encoding="utf-8")
        )[str(video_id)]
        keys = [tuple(int(value) for value in key) for key in cache_keys]
        keys.sort(key=lambda key: (key[1], key[2]))
        chunks = plan_chunks(
            video_id=video_id,
            ordered_keys=keys,
            events_by_key=by_key,
            target_records=args.records_per_chunk,
        )
        cache = FrozenPerceptionCache(partition["cache"])
        gt_bundle = load_gt(Path("/data/DATASETS/TRACKING/JDE/VisionTrack/annotations/train.json"))
        single, single_stats, single_skipped = build_v2_records(
            trace=trace,
            cache_root=Path(partition["cache"]),
            annotations=Path("/data/DATASETS/TRACKING/JDE/VisionTrack/annotations/train.json"),
            checkpoint_hash="contract-equivalence",
            horizon=8,
            association_backend="cosine_contract",
            cache_obj=cache,
            gt_bundle=gt_bundle,
            cache_keys_by_video={video_id: keys},
            order_index=order,
            minimal_events=True,
            video_ids=[video_id],
        )

        chunk_records = []
        state = MutableGMTState().initialize_trajectory_rng(video_id)
        cursor = 0
        for chunk in chunks:
            for key_index in range(cursor, int(chunk["key_start"])):
                key = keys[key_index]
                advance_off_state_for_key(
                    payload=cache.load(*key),
                    key=key,
                    by_key=by_key,
                    actions=actions,
                    memories=memories,
                    engine=CachedPerceptionMutableAssociationV2(),
                    state=state,
                )
            initial = state.clone()
            chunk_result, _stats, _skipped = build_v2_records(
                trace=trace,
                cache_root=Path(partition["cache"]),
                annotations=Path("/data/DATASETS/TRACKING/JDE/VisionTrack/annotations/train.json"),
                checkpoint_hash="contract-equivalence",
                horizon=8,
                association_backend="cosine_contract",
                cache_obj=cache,
                gt_bundle=gt_bundle,
                cache_keys_by_video={video_id: keys},
                order_index=order,
                minimal_events=True,
                video_ids=[video_id],
                initial_state=initial,
                key_start_index=int(chunk["key_start"]),
                key_end_index=int(chunk["key_end"]),
                selected_key_range=(int(chunk["key_start"]), int(chunk["key_end"])),
            )
            chunk_records.extend(chunk_result)
            for key_index in range(int(chunk["key_start"]), int(chunk["key_end"])):
                key = keys[key_index]
                advance_off_state_for_key(
                    payload=cache.load(*key),
                    key=key,
                    by_key=by_key,
                    actions=actions,
                    memories=memories,
                    engine=CachedPerceptionMutableAssociationV2(),
                    state=state,
                )
            cursor = int(chunk["key_end"])
        comparison = compare_records_exact(single, chunk_records)
        report = {
            "schema_version": "jev_deterministic_intra_video_chunk_v1",
            "status": comparison["status"],
            "classification": "BOUNDED_CPU_CONTRACT_PROBE_NOT_FORMAL_GMT_AUTHORIZATION",
            "video_id": video_id,
            "trace_prefix_lines": count,
            "records_per_chunk_target": int(args.records_per_chunk),
            "chunk_count": len(chunks),
            "single_stats": single_stats,
            "single_skipped": int(single_skipped),
            "comparison": comparison,
            "formal_gmt_equivalence_required_before_canonical": True,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if comparison["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
