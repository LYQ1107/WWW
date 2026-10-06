"""Compare preserved pre-RNG H=8 shards with the new controlled shards."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from statistics import mean


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_records(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield line_number, json.loads(line)
            except json.JSONDecodeError:
                # A graceful SIGTERM may leave only the final JSONL line
                # incomplete.  Preserve that fact in the audit instead of
                # making the whole old screening shard unreadable.
                yield line_number, None


def key(record: dict, video_id: int) -> tuple:
    context = record.get("state", {}).get("online_context", {})
    return (
        int(video_id),
        str(record.get("sequence")),
        int(record.get("frame", context.get("frame", -1))),
        int(record.get("view", context.get("view", -1))),
        str(record.get("question_type")),
        int(context.get("detection_index", -1)),
        int(context.get("event_order", -1)),
    )


def old_path_for(snapshot: dict, video_id: int) -> Path | None:
    for item in snapshot.get("workers", []):
        if int(item.get("video_id", -1)) != int(video_id):
            continue
        root = Path(str(item.get("output_root", "")))
        # ``manifest`` is metadata, not a JSONL shard.  Prefer the official
        # records file even when the pause snapshot also contains a stale
        # temporary path; otherwise the audit silently compares zero records
        # from ``manifest.json`` and can report a false PASS.
        candidates = [root / "records.jsonl"] + sorted(root.glob("records.jsonl.tmp.*"))
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        for candidate in (item.get("tmp_records_path"), item.get("manifest")):
            if candidate and Path(candidate).is_file() and Path(candidate).suffix != ".json":
                return Path(candidate)
    return None


def read_shard(path: Path, video_id: int) -> tuple[dict[tuple, dict], dict]:
    records = {}
    invalid = 0
    questions = Counter()
    for line_number, record in iter_records(path):
        if record is None:
            invalid += 1
            continue
        try:
            record_key = key(record, video_id)
            if record_key in records:
                raise ValueError("duplicate semantic key")
            records[record_key] = record
            questions[str(record.get("question_type"))] += 1
        except Exception:
            invalid += 1
    return records, {
        "path": str(path),
        "sha256": "sha256:" + sha256(path),
        "records": len(records),
        "invalid_lines": invalid,
        "question_counts": dict(questions),
    }


def p95(values: list[float]) -> float | None:
    if not values:
        return None
    values = sorted(values)
    index = min(len(values) - 1, max(0, int(0.95 * len(values) + 0.999999) - 1))
    return float(values[index])


def action_outcome(record: dict, action: str) -> dict:
    return dict(record.get("action_outcomes", {}).get(action, {}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-runtime", type=Path, required=True)
    parser.add_argument("--new-runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video-ids", type=int, nargs="+", required=True)
    args = parser.parse_args()
    old_runtime = args.old_runtime.resolve()
    new_runtime = args.new_runtime.resolve()
    snapshot = json.loads((old_runtime / "RNG_UNCONTROLLED_PAUSE_20261006.json").read_text())
    queue = json.loads((old_runtime / "queue_state.json").read_text())
    compared = 0
    best_equal = 0
    utility_delta: list[float] = []
    target_l1: list[float] = []
    action_outcome_delta: list[float] = []
    question_disagreement = Counter()
    action_disagreement = Counter()
    per_video = {}
    old_paths = {}
    new_paths = {}
    for video_id in sorted({int(value) for value in args.video_ids}):
        old_path = old_path_for(snapshot, video_id)
        new_path = new_runtime / f"video_{video_id:02d}" / "records.jsonl"
        if old_path is None or not new_path.is_file():
            per_video[str(video_id)] = {
                "status": "MISSING_SIDE",
                "old_path": None if old_path is None else str(old_path),
                "new_path": str(new_path),
            }
            continue
        old_paths[str(video_id)] = str(old_path)
        new_paths[str(video_id)] = str(new_path)
        old_records, old_meta = read_shard(old_path, video_id)
        new_records, new_meta = read_shard(new_path, video_id)
        keys = sorted(set(old_records) & set(new_records), key=str)
        video_compared = 0
        video_best_equal = 0
        video_utility_delta = []
        for record_key in keys:
            old = old_records[record_key]
            new = new_records[record_key]
            video_compared += 1
            compared += 1
            same_best = set(old.get("best_actions", ())) == set(new.get("best_actions", ()))
            best_equal += int(same_best)
            video_best_equal += int(same_best)
            if not same_best:
                question_disagreement[str(record_key[4])] += 1
                for action in sorted(
                    set(old.get("best_actions", ())) | set(new.get("best_actions", ()))
                ):
                    action_disagreement[action] += 1
            old_probs = list(old.get("target_probs", ()))
            new_probs = list(new.get("target_probs", ()))
            if len(old_probs) == len(new_probs):
                target_l1.append(sum(abs(float(a) - float(b)) for a, b in zip(old_probs, new_probs)))
            for action in sorted(set(old.get("action_outcomes", {})) & set(new.get("action_outcomes", {}))):
                old_outcome = action_outcome(old, action)
                new_outcome = action_outcome(new, action)
                if isinstance(old_outcome.get("utility"), (int, float)) and isinstance(new_outcome.get("utility"), (int, float)):
                    delta = float(new_outcome["utility"]) - float(old_outcome["utility"])
                    utility_delta.append(delta)
                    video_utility_delta.append(delta)
                fields = sorted(set(old_outcome) & set(new_outcome))
                for field in fields:
                    if isinstance(old_outcome.get(field), (int, float)) and isinstance(new_outcome.get(field), (int, float)):
                        action_outcome_delta.append(abs(float(new_outcome[field]) - float(old_outcome[field])))
        per_video[str(video_id)] = {
            "status": "COMPARED",
            "old": old_meta,
            "new": new_meta,
            "old_records": len(old_records),
            "new_records": len(new_records),
            "records_compared": video_compared,
            "records_only_old": len(set(old_records) - set(new_records)),
            "records_only_new": len(set(new_records) - set(old_records)),
            "best_action_agreement_rate": video_best_equal / max(1, video_compared),
            "mean_utility_delta_new_minus_old": mean(video_utility_delta) if video_utility_delta else None,
        }
    new_manifests = {}
    for video_id in sorted({int(value) for value in args.video_ids}):
        path = new_runtime / f"video_{video_id:02d}" / "manifest.json"
        if path.is_file():
            new_manifests[str(video_id)] = json.loads(path.read_text(encoding="utf-8"))
    old_queue_classification = {
        "queue_status": queue.get("status"),
        "dataset_classification": queue.get("dataset_classification"),
        "canonical_authority": queue.get("canonical_authority"),
        "source_commit": snapshot.get("source_commit"),
        "old_artifacts_preserved": bool(snapshot.get("classification", {}).get("old_artifacts_preserved")),
    }
    rng_ok = all(
        manifest.get("trajectory_rng_policy") == "branch_local_explicit_python_random_v1"
        and int(manifest.get("trajectory_rng_master_seed", -1)) == 20261006
        and int(manifest.get("trajectory_rng_video_seed", -1)) == 20261006 + int(video_id)
        and manifest.get("trajectory_rng_state_cloned_per_counterfactual_branch") is True
        and manifest.get("proposal_reused_across_legal_actions") is True
        and manifest.get("reassociate_reuses_score_matrix") is True
        and manifest.get("second_transformer_call_for_reassociate") is False
        for video_id, manifest in new_manifests.items()
    ) and len(new_manifests) == len({int(value) for value in args.video_ids})
    report = {
        "status": "PASS" if rng_ok and all(item.get("status") == "COMPARED" for item in per_video.values()) else "INCOMPLETE",
        "classification": "OLD_PRE_RNG_CONTROL_SCREENING_ONLY_NEW_SMALL_GATE",
        "old_dataset": old_queue_classification,
        "old_trace_rng_provenance": "BLOCKED_BY_MISSING_LEGACY_RNG_PROVENANCE",
        "new_dataset_classification": "CANONICAL_RNG_CONTROLLED_H8_SMALL_GATE",
        "video_ids": sorted({int(value) for value in args.video_ids}),
        "records_compared": compared,
        "best_action_agreement_rate": best_equal / max(1, compared),
        "best_action_disagreement_rate": 1.0 - best_equal / max(1, compared),
        "mean_utility_delta_new_minus_old": mean(utility_delta) if utility_delta else None,
        "p95_utility_delta_absolute": p95([abs(value) for value in utility_delta]),
        "max_utility_delta_absolute": max((abs(value) for value in utility_delta), default=None),
        "target_prob_mean_l1": mean(target_l1) if target_l1 else None,
        "target_prob_max_l1": max(target_l1, default=None),
        "action_outcome_mean_absolute_delta": mean(action_outcome_delta) if action_outcome_delta else None,
        "question_wise_best_action_disagreement": dict(question_disagreement),
        "action_wise_best_action_disagreement": dict(action_disagreement),
        "old_paths": old_paths,
        "new_paths": new_paths,
        "new_manifests_rng_provenance_pass": rng_ok,
        "per_video": per_video,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "records_compared": compared}, indent=2))


if __name__ == "__main__":
    main()
