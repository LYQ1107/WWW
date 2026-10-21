"""Build the locked full-TRAIN listwise dataset from real causal GMT calls.

This join happens after inference and after Ω feature extraction.  Labels come
from the immutable post-audit event rows; model inputs contain only the native
score matrix, causal track history, and frozen Ω ROI vectors.  Unknown, mixed,
ambiguous, missing-match and NULL rows are counted and excluded from the
supervised groups.  The resulting files are ragged listwise candidate groups,
so no GT-built track or future observation is introduced.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "manifests/omega_full24_training_protocol_v1.json"
AUDIT_ROOT = Path("/data1/liuyeqiang/WWW_omega_fulltrain_runtime_20261016/full_train_logit_audit_v1")
FEATURE_ROOT = Path("/data1/liuyeqiang/WWW_omega_fulltrain_runtime_20261016/omega512_features_v1")
FEATURE_DIM = 10


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _audit_dir(scene: int) -> Path:
    path = AUDIT_ROOT / f"scene{scene}_full_logit_v1"
    if not path.is_dir():
        raise FileNotFoundError(f"missing native-score audit for scene {scene}: {path}")
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("status") != "COMPLETE_FULL_CAUSAL_AUDIT_WITH_NATIVE_SCORE_MATRICES":
        raise ValueError(f"audit {path} is not a complete score-matrix audit: {manifest.get('status')}")
    if not manifest.get("prefix_raw_prediction_exact"):
        raise ValueError(f"audit {path} lacks OFF-prefix parity")
    return path


def _feature_store(scene: int, feature_root: Path, backbone: str):
    if backbone == "omega":
        folder_name, expected_dim, expected_status = f"scene{scene}_omega512_v1", 2048, "COMPLETE_OMEGA512_FEATURE_EXTRACTION"
    elif backbone == "dino":
        folder_name, expected_dim, expected_status = f"scene{scene}_dinov2_v1", 1024, "COMPLETE_DINOV2_FEATURE_EXTRACTION"
    else:
        raise ValueError(backbone)
    path = feature_root / folder_name / "features.npz"
    manifest_path = path.with_name("manifest.json")
    if not path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"missing Ω feature store for scene {scene}: {path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != expected_status:
        raise ValueError(f"Ω feature store is not complete: {manifest_path}")
    archive = np.load(path, mmap_mode="r")
    keys = np.asarray(archive["source_keys"], dtype=np.int64)
    vectors = np.asarray(archive["features"], dtype=np.float32)
    if keys.ndim != 2 or keys.shape[1] != 2 or vectors.shape[0] != keys.shape[0] or vectors.shape[1] != expected_dim:
        raise ValueError(f"invalid Ω archive shapes for scene {scene}: {keys.shape}, {vectors.shape}")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if not np.isfinite(vectors).all() or np.any(norms <= 1e-8):
        raise ValueError(f"non-finite/zero Ω feature in scene {scene}")
    vectors = vectors / norms
    index = {tuple(key): row for row, key in enumerate(keys.tolist())}
    if len(index) != len(keys):
        raise ValueError(f"duplicate Ω source key in scene {scene}")
    return path, manifest_path, index, vectors


def _read_events(path: Path):
    events = {}
    counters = collections.Counter()
    with gzip.open(path / "events.jsonl.gz", "rt") as stream:
        for line in stream:
            row = json.loads(line)
            key = tuple(int(value) for value in row["key"][1:])
            if key in events:
                raise ValueError(f"duplicate event source {key} in {path}")
            events[key] = row
            counters["events"] += 1
            if row.get("query_gt_identity") is None:
                counters["unknown_query"] += 1
            if any(code in row.get("types", []) for code in ("E", "F")):
                counters["mixed_or_ambiguous"] += 1
    return events, counters


def _read_calls(path: Path):
    decisions = []
    with gzip.open(path / "actual_decisions.jsonl.gz", "rt") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("stage") != "run_memory_tracker":
                decisions.append(row)
    rankings = []
    with gzip.open(path / "rankings.jsonl.gz", "rt") as stream:
        for line in stream:
            row = json.loads(line)
            if "track_ids" not in row or "track_scores" not in row:
                raise ValueError(f"ranking row lacks native scores in {path}")
            rankings.append(row)
    if len(decisions) != len(rankings):
        raise ValueError(f"decision/ranking count mismatch in {path}: {len(decisions)} != {len(rankings)}")
    return decisions, rankings


def _make_candidate(query_vector, refs, image_meta, feature_index, feature_vectors, native_score, rank, candidate_count, margin, query_camera):
    source_keys = [tuple(int(value) for value in ref["source"]) for ref in refs]
    missing = [source for source in source_keys if source not in feature_index]
    if missing:
        raise ValueError(f"causal reference lacks Ω feature: {missing[:3]}")
    ref_values = feature_vectors[[feature_index[source] for source in source_keys]]
    similarity = ref_values @ query_vector
    cameras = np.asarray([int(image_meta[source[0]]["view_id"]) for source in source_keys])
    # Feature order is fixed in the manifest and shared by all models:
    # native score, normalized native rank, frozen evidence max/mean/top-2 similarity,
    # log reference count, cross-camera fraction, same-camera fraction,
    # current native margin, and candidate count.
    top_count = min(2, len(similarity))
    return np.asarray([
        float(native_score),
        float(rank / max(candidate_count - 1, 1)),
        float(np.max(similarity)),
        float(np.mean(similarity)),
        float(np.mean(np.sort(similarity)[-top_count:])),
        float(np.log1p(len(refs))),
        float(np.mean(cameras != query_camera)),
        float(np.mean(cameras == query_camera)),
        float(margin),
        float(candidate_count),
    ], dtype=np.float32)


def _build_scene(scene: int, feature_root: Path, backbone: str):
    audit = _audit_dir(scene)
    feature_path, feature_manifest, feature_index, feature_vectors = _feature_store(scene, feature_root, backbone)
    feature_manifest_data = json.loads(feature_manifest.read_text())
    feature_audit_root = Path(feature_manifest_data.get("audit_root", ""))
    current_detector_path = audit / "detector_provenance.jsonl.gz"
    if feature_audit_root.is_dir() and (feature_audit_root / "detector_provenance.jsonl.gz").is_file():
        def read_boxes(path):
            values = {}
            with gzip.open(path, "rt") as stream:
                for line in stream:
                    row = json.loads(line)
                    values[tuple(int(value) for value in row["source"])] = np.asarray(row["detection"]["bbox"], dtype=np.float64)
            return values
        feature_boxes = read_boxes(feature_audit_root / "detector_provenance.jsonl.gz")
        current_boxes = read_boxes(current_detector_path)
        if feature_boxes.keys() != current_boxes.keys() or any(not np.allclose(feature_boxes[key], current_boxes[key], atol=1e-5, rtol=0.0) for key in current_boxes):
            raise ValueError(f"feature and native-score audits have different detector provenance in scene {scene}")
    detector_keys = set()
    with gzip.open(audit / "detector_provenance.jsonl.gz", "rt") as stream:
        for line in stream:
            detector_keys.add(tuple(int(value) for value in json.loads(line)["source"]))
    if detector_keys != set(feature_index):
        raise ValueError(f"native-score audit and Ω feature source sets differ in scene {scene}: audit={len(detector_keys)} features={len(feature_index)}")
    events, event_counts = _read_events(audit)
    decisions, rankings = _read_calls(audit)
    # The annotations include the image view for the causal camera shortcut
    # control.  It is read-only dataset metadata, not GT.
    data = json.loads(Path("/data/DATASETS/TRACKING/JDE/VisionTrack/annotations/train.json").read_text())
    image_meta = {int(row["id"]): row for row in data["images"] if int(row["video_id"]) == scene}
    groups = []
    stats = collections.Counter(event_counts)
    for call_index, (decision, ranking) in enumerate(zip(decisions, rankings)):
        if ranking["queries"] != decision["queries"]:
            raise ValueError(f"query order mismatch at scene {scene} call {call_index}")
        track_ids = [int(value) for value in ranking["track_ids"]]
        score_matrix = np.asarray(ranking["track_scores"], dtype=np.float32)
        expected_shape = (len(decision["queries"]), len(track_ids))
        # JSON has no representation for a zero-column matrix: an empty
        # track-score list is decoded as shape (0,) even when this call has
        # live queries.  Treat it as the valid (query_count, 0) matrix; the
        # per-query no-candidate mask below will exclude those rows.
        if score_matrix.size == 0 and expected_shape[0] * expected_shape[1] == 0:
            score_matrix = np.empty(expected_shape, dtype=np.float32)
        if score_matrix.shape != expected_shape:
            raise ValueError(f"native score shape mismatch scene {scene} call {call_index}: {score_matrix.shape}")
        refs_by_track = collections.defaultdict(list)
        for ref in decision["references"]:
            refs_by_track[int(ref["track"])].append(ref)
        for query_index, source_raw in enumerate(decision["queries"]):
            source = tuple(int(value) for value in source_raw)
            row = events.get(source)
            if row is None:
                raise ValueError(f"missing event row for causal query {scene, *source}")
            stats["causal_queries"] += 1
            identity = row.get("query_gt_identity")
            if identity is None or row.get("query_label_reason") != "known_one_to_one_IoU_ge_0.5":
                stats["masked_unknown_or_unmatched"] += 1
                continue
            types = set(row.get("types", []))
            if types.intersection({"E", "F"}):
                stats["masked_mixed_or_ambiguous"] += 1
                continue
            positives = {int(value) for value in row.get("positive_tracks", [])}
            if not positives:
                stats["masked_no_pure_positive"] += 1
                continue
            query_key = source
            if query_key not in feature_index:
                raise ValueError(f"causal query lacks Ω feature: scene {scene} source {source}")
            query_vector = feature_vectors[feature_index[query_key]]
            n_candidates = len(track_ids)
            if n_candidates == 0:
                stats["masked_no_candidate"] += 1
                continue
            candidates = []
            labels = []
            for rank, track in enumerate(track_ids):
                refs = refs_by_track.get(track, [])
                if not refs:
                    raise ValueError(f"track {track} has no source history at scene {scene} call {call_index}")
                candidates.append(_make_candidate(query_vector, refs, image_meta, feature_index, feature_vectors,
                                                  score_matrix[query_index, rank], rank, n_candidates,
                                                  float(ranking["relative_margin"][query_index]), int(row["camera"])))
                labels.append(int(track in positives))
            labels = np.asarray(labels, dtype=np.uint8)
            if labels.sum() == 0:
                stats["masked_positive_not_in_legal_matrix"] += 1
                continue
            stats["legal_groups"] += 1
            if labels.sum() < len(labels):
                stats["trainable_groups"] += 1
            else:
                stats["all_candidates_positive"] += 1
            groups.append({
                "scene": scene,
                "source": source,
                "gt": int(identity),
                "features": np.asarray(candidates, dtype=np.float32),
                "labels": labels,
                "track_ids": np.asarray(track_ids, dtype=np.int64),
                "correctable": int(bool(row.get("correctable_known_wrong"))),
                "native_top_positive": int(labels[0] > 0),
                "native_rank": int(np.argmax(labels)) if labels.any() else -1,
                "call_index": call_index,
            })
    for key in ("causal_queries", "masked_unknown_or_unmatched", "masked_mixed_or_ambiguous", "masked_no_pure_positive",
                "masked_no_candidate", "masked_positive_not_in_legal_matrix", "legal_groups", "trainable_groups", "all_candidates_positive"):
        stats[key] += 0
    return groups, stats, {"audit_manifest": _sha256(audit / "manifest.json"), "feature_manifest": _sha256(feature_manifest), "feature_archive": _sha256(feature_path)}


def _write_split(path: Path, groups):
    path.parent.mkdir(parents=True, exist_ok=True)
    offsets = [0]
    features = []
    labels = []
    track_ids = []
    sources = []
    scenes = []
    gt = []
    correctable = []
    native_top_positive = []
    native_rank = []
    call_indices = []
    for group in groups:
        features.append(group["features"])
        labels.append(group["labels"])
        track_ids.append(group["track_ids"])
        offsets.append(offsets[-1] + len(group["labels"]))
        sources.append(group["source"])
        scenes.append(group["scene"])
        gt.append(group["gt"])
        correctable.append(group["correctable"])
        native_top_positive.append(group["native_top_positive"])
        native_rank.append(group["native_rank"])
        call_indices.append(group["call_index"])
    np.savez(
        path,
        offsets=np.asarray(offsets, dtype=np.int64),
        candidate_features=np.concatenate(features, axis=0).astype(np.float32),
        labels=np.concatenate(labels, axis=0).astype(np.uint8),
        track_ids=np.concatenate(track_ids, axis=0).astype(np.int64),
        query_sources=np.asarray(sources, dtype=np.int64),
        scene_ids=np.asarray(scenes, dtype=np.int16),
        query_gt=np.asarray(gt, dtype=np.int32),
        correctable=np.asarray(correctable, dtype=np.uint8),
        native_top_positive=np.asarray(native_top_positive, dtype=np.uint8),
        native_rank=np.asarray(native_rank, dtype=np.int16),
        call_indices=np.asarray(call_indices, dtype=np.int32),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene-ids", nargs="+", type=int)
    parser.add_argument("--backbone", choices=["omega", "dino"], default="omega")
    parser.add_argument("--feature-root", type=Path)
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text())
    fit = protocol["scene_split"]["fit_video_ids"]
    validation = protocol["scene_split"]["validation_video_ids"]
    scenes = args.scene_ids or sorted(fit + validation)
    if sorted(scenes) != sorted(set(scenes)) or set(scenes) != set(fit + validation):
        raise ValueError("dataset builder requires exactly the locked 24 scenes")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    started = time.perf_counter()
    feature_root = args.feature_root or (FEATURE_ROOT if args.backbone == "omega" else Path("/data1/liuyeqiang/WWW_omega_fulltrain_runtime_20261016/dinov2_features_v1"))
    all_groups = []
    scene_stats = {}
    source_hashes = {}
    for scene in scenes:
        groups, stats, hashes = _build_scene(scene, feature_root, args.backbone)
        all_groups.extend(groups)
        scene_stats[str(scene)] = dict(stats)
        source_hashes[str(scene)] = hashes
        print(json.dumps({"scene": scene, "groups": len(groups), "stats": dict(stats)}), flush=True)
    fit_groups = [group for group in all_groups if group["scene"] in fit]
    validation_groups = [group for group in all_groups if group["scene"] in validation]
    _write_split(args.output / "fit.npz", fit_groups)
    _write_split(args.output / "validation.npz", validation_groups)
    _write_split(args.output / "all24.npz", all_groups)
    manifest = {
        "status": "COMPLETE_FULL24_CAUSAL_SUPERVISED_DATASET",
        "protocol": str(PROTOCOL),
        "protocol_sha256": _sha256(PROTOCOL),
        "scene_ids": scenes,
        "fit_scene_ids": fit,
        "validation_scene_ids": validation,
        "evidence_backbone": args.backbone,
        "feature_definition": "native score matrix from actual causal GMT calls plus frozen per-candidate evidence; no labels in inputs",
        "candidate_feature_names": ["native_score", "native_rank_norm", "evidence_max", "evidence_mean", "evidence_top2_mean", "log_reference_count", "cross_camera_fraction", "same_camera_fraction", "native_relative_margin", "candidate_count"],
        "groups": {"all24": len(all_groups), "fit": len(fit_groups), "validation": len(validation_groups)},
        "candidate_rows": {"all24": int(sum(len(group["labels"]) for group in all_groups)), "fit": int(sum(len(group["labels"]) for group in fit_groups)), "validation": int(sum(len(group["labels"]) for group in validation_groups))},
        "scene_stats": scene_stats,
        "source_hashes": source_hashes,
        "files": {name: _sha256(args.output / name) for name in ("fit.npz", "validation.npz", "all24.npz")},
        "wall_seconds": time.perf_counter() - started,
        "ground_truth_used_only_for_mask_and_target": True,
        "future_information": False,
        "gt_built_tracks": False,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": manifest["status"], "groups": manifest["groups"], "candidate_rows": manifest["candidate_rows"], "seconds": manifest["wall_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
