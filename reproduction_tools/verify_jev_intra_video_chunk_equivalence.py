"""Verify exact single-worker vs merged deterministic chunk records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jev_intra_video_chunking import compare_records_exact


def read_records(path: Path):
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"record is not an object: {path}")
                records.append(value)
    return records


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def compare_chunk_ranges(plan, manifests, chunked_count):
    chunks = sorted(plan.get("chunks", []), key=lambda item: int(item["chunk_index"]))
    manifest_by_index = {int(item.get("chunk_index", -1)): item for item in manifests}
    checks = {
        "plan_chunk_count": len(chunks),
        "manifest_chunk_count": len(manifests),
        "manifest_count_matches_plan": len(chunks) == len(manifests),
        "manifest_indices_unique": len(manifest_by_index) == len(manifests),
        "ranges_contiguous": True,
        "record_ranges_contiguous": True,
        "record_count_matches_plan": sum(int(item.get("decision_count", 0)) for item in chunks)
        == int(chunked_count),
        "manifest_counts_match_plan": True,
    }
    previous_key_end = None
    previous_record_end = None
    for chunk in chunks:
        key_start = int(chunk["key_start"])
        key_end = int(chunk["key_end"])
        record_start = int(chunk["record_start"])
        record_end = int(chunk["record_end"])
        if previous_key_end is not None and key_start != previous_key_end:
            checks["ranges_contiguous"] = False
        if previous_record_end is not None and record_start != previous_record_end:
            checks["record_ranges_contiguous"] = False
        previous_key_end = key_end
        previous_record_end = record_end
        manifest = manifest_by_index.get(int(chunk["chunk_index"]))
        if manifest is None:
            checks["manifest_counts_match_plan"] = False
            continue
        checks["manifest_counts_match_plan"] = checks["manifest_counts_match_plan"] and int(
            manifest.get("records", -1)
        ) == int(chunk["decision_count"])
    checks["pass"] = all(
        value is True
        for key, value in checks.items()
        if key not in {"plan_chunk_count", "manifest_chunk_count"}
    )
    return checks


def compare_provenance(single_manifest, chunk_manifests, warmup_manifest):
    def normalized(manifest):
        return {
            "source_commit": manifest.get("source_commit"),
            "association_backend": manifest.get("association_backend"),
            "counterfactual_engine": manifest.get("counterfactual_engine"),
            "checkpoint_sha256": manifest.get("gmt_checkpoint_sha256", manifest.get("checkpoint_sha256")),
            # Single-worker manifests retain both the full-trace hash and the
            # per-video partition hash.  Chunk manifests necessarily carry
            # the latter; compare like-for-like provenance here.
            "trace_sha256": manifest.get(
                "trace_partition_sha256",
                manifest.get("trace_sha256", manifest.get("source_trace_sha256")),
            ),
            "order_index_sha256": manifest.get("source_order_index_sha256", manifest.get("order_index_sha256")),
            "cache_index_sha256": manifest.get("cache_index_sha256"),
            "horizon": manifest.get("horizon"),
            "state_schema_version": manifest.get("state_schema_version"),
            "record_schema_version": manifest.get("record_schema_version"),
            "feature_schema_version": manifest.get("feature_schema_version"),
            "utility_definition": manifest.get("utility_definition"),
            "trajectory_rng_policy": manifest.get("trajectory_rng_policy"),
            "trajectory_rng_master_seed": manifest.get("trajectory_rng_master_seed"),
            "trajectory_rng_video_seed": manifest.get("trajectory_rng_video_seed"),
        }

    normalized_single = normalized(single_manifest)
    normalized_chunks = [normalized(item) for item in chunk_manifests]
    shared_fields = {}
    for field in normalized_single:
        values = [normalized_single[field]] + [item[field] for item in normalized_chunks]
        shared_fields[field] = {
            "values": values,
            "equal": len(set(values)) == 1,
        }
    expected_seed = single_manifest.get("trajectory_rng_video_seed")
    snapshots = (warmup_manifest or {}).get("snapshots", [])
    snapshot_rng = [
        {
            "chunk_index": item.get("chunk", {}).get("chunk_index"),
            "trajectory_rng_seed": item.get("metadata", {}).get("trajectory_rng_seed"),
            "trajectory_rng_calls": item.get("metadata", {}).get("trajectory_rng_calls"),
            "snapshot_sha256": item.get("sha256"),
        }
        for item in snapshots
    ]
    rng_seed_exact = bool(snapshot_rng) and all(
        item["trajectory_rng_seed"] == expected_seed for item in snapshot_rng
    )
    rng_calls_monotonic = all(
        left["trajectory_rng_calls"] <= right["trajectory_rng_calls"]
        for left, right in zip(snapshot_rng, snapshot_rng[1:])
    )
    return {
        "shared_fields": shared_fields,
        "source_commit_single": single_manifest.get("source_commit"),
        "source_commits_chunks": [item.get("source_commit") for item in chunk_manifests],
        "source_commit_equal": all(
            item.get("source_commit") == single_manifest.get("source_commit")
            for item in chunk_manifests
        ),
        "warmup_source_commit": (warmup_manifest or {}).get("source_commit"),
        "warmup_source_commit_equal": bool(
            (warmup_manifest or {}).get("source_commit")
            and (warmup_manifest or {}).get("source_commit") == single_manifest.get("source_commit")
            and all(
                (warmup_manifest or {}).get("source_commit") == item.get("source_commit")
                for item in chunk_manifests
            )
        ),
        "trajectory_rng_policy": single_manifest.get("trajectory_rng_policy"),
        "trajectory_rng_master_seed": single_manifest.get("trajectory_rng_master_seed"),
        "trajectory_rng_video_seed": expected_seed,
        "snapshot_rng": snapshot_rng,
        "rng_seed_exact": rng_seed_exact,
        "rng_calls_monotonic": rng_calls_monotonic,
        "pass": bool(
            all(item["equal"] for item in shared_fields.values())
            and bool(single_manifest.get("source_commit"))
            and rng_seed_exact
            and rng_calls_monotonic
            and bool((warmup_manifest or {}).get("source_commit"))
            and (warmup_manifest or {}).get("source_commit") == single_manifest.get("source_commit")
            and all(
                (warmup_manifest or {}).get("source_commit") == item.get("source_commit")
                for item in chunk_manifests
            )
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-records", type=Path, required=True)
    parser.add_argument("--chunk-records", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--single-manifest", type=Path)
    parser.add_argument("--chunk-manifests", nargs="+", type=Path)
    parser.add_argument("--chunk-plan", type=Path)
    parser.add_argument("--warmup-manifest", type=Path)
    parser.add_argument(
        "--expected-source-commit",
        help="exact source-commit value recorded in all manifests (usually a short SHA)",
    )
    parser.add_argument(
        "--expected-source-commit-full",
        help="full SHA corresponding to --expected-source-commit, recorded for audit",
    )
    args = parser.parse_args()
    single = read_records(args.single_records)
    chunked = []
    for path in args.chunk_records:
        chunked.extend(read_records(path))
    report = compare_records_exact(single, chunked)
    report.update(
        {
            "single_records_path": str(args.single_records.resolve()),
            "chunk_records_paths": [str(path.resolve()) for path in args.chunk_records],
            "canonical_authority": "PASS_ONLY_IF_EXACT_RECORD_MATCHES_AND_SNAPSHOT_PROVENANCE_IS_VALID",
        }
    )
    if args.chunk_plan is not None:
        if args.chunk_manifests is None or args.single_manifest is None or args.warmup_manifest is None:
            raise SystemExit("--chunk-plan requires --single-manifest, --chunk-manifests, and --warmup-manifest")
        plan = read_json(args.chunk_plan)
        single_manifest = read_json(args.single_manifest)
        chunk_manifests = [read_json(path) for path in args.chunk_manifests]
        warmup_manifest = read_json(args.warmup_manifest)
        report["chunk_range_gate"] = compare_chunk_ranges(
            plan, chunk_manifests, len(chunked)
        )
        report["rng_provenance_gate"] = compare_provenance(
            single_manifest, chunk_manifests, warmup_manifest
        )
        source_values = [single_manifest.get("source_commit")]
        source_values.extend(item.get("source_commit") for item in chunk_manifests)
        source_values.append(warmup_manifest.get("source_commit"))
        source_commit_exact = True
        if args.expected_source_commit is not None:
            source_commit_exact = all(
                value == args.expected_source_commit for value in source_values
            )
            report["source_commit_expectation"] = {
                "expected_recorded_value": args.expected_source_commit,
                "expected_full_sha": args.expected_source_commit_full,
                "observed_values": source_values,
                "exact": source_commit_exact,
            }
        report["status"] = "PASS" if (
            report["status"] == "PASS"
            and all(
                report["field_comparisons"][name]["equal_within_1e-6"]
                for name in ("best_actions", "target_probs", "utilities", "state_features")
            )
            and report["chunk_range_gate"]["pass"]
            and report["rng_provenance_gate"]["pass"]
            and source_commit_exact
        ) else "FAIL"
        report["final_gate"] = {
            "raw_canonical_records_exact": report["record_mismatches"] == 0
            and report["semantic_keys_exact"]
            and report["duplicate_free"],
            "best_actions_exact": report["field_comparisons"]["best_actions"]["equal_within_1e-6"],
            "target_probs_within_1e-6": report["field_comparisons"]["target_probs"]["equal_within_1e-6"],
            "utilities_within_1e-6": report["field_comparisons"]["utilities"]["equal_within_1e-6"],
            "state_features_within_1e-6": report["field_comparisons"]["state_features"]["equal_within_1e-6"],
            "ranges_no_gap_overlap_duplicate": report["chunk_range_gate"]["pass"],
            "rng_provenance_exact": report["rng_provenance_gate"]["pass"],
            "canonical_sha_identical": report["canonical_sha_identical"],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
