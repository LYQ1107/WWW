#!/usr/bin/env python3
"""Verify delivered full-TRAIN control outputs without opening any GT labels."""
import argparse
from contextlib import ExitStack
import hashlib
import json
import math
from pathlib import Path
import sys
import time

DEFAULT_METHODS = ("RULE_BASED_WORLD_OBJECT_V3",
                   "LEGACY_PAIR_THEN_MERGE_ROI_RESTORED_CONTROL",
                   "LEGACY_PAIR_THEN_MERGE_FEATURE_OFF_CONTROL")


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def current_candidates(row):
    result = {}
    for view, objects in row["tracks_2d_by_camera"].items():
        view = int(view)
        if view not in (1, 2):
            raise ValueError("UNEXPECTED_CAMERA")
        ids = set()
        for obj in objects:
            key = (view, obj["detection_id"])
            bbox = obj["bbox_xywh"]
            if key in result or obj["global_id"] in ids:
                raise ValueError("DUPLICATE_DETECTION_OR_WITHIN_CAMERA_ID")
            if (len(bbox) != 4 or not all(math.isfinite(x) for x in bbox)
                    or min(bbox[2:]) <= 0 or obj["global_id"] <= 0
                    or obj.get("visible") is not True
                    or obj.get("evaluation_eligible") is not True):
                raise ValueError("INVALID_OR_PREDICTION_ONLY_2D_OUTPUT")
            result[key] = (bbox, obj["score"])
            ids.add(obj["global_id"])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    args = parser.parse_args()
    root = args.run_root
    metadata = root / "metadata_only_dataset/annotations/train.json"
    if read(metadata).get("annotations", []) != []:
        raise ValueError("PHYSICALLY_ANNOTATION_FREE_METADATA_REQUIRED")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from worldmot.data.visiontrack import VisionTrack
    from worldmot.runtime.experiment_authorization import source_fingerprint
    ds = VisionTrack(root / "metadata_only_dataset", split="train", load_annotations=False)
    if ds.scene_ids() != list(range(1, 25)):
        raise ValueError("TRAIN_METADATA_SCOPE_MISMATCH")
    source = source_fingerprint()
    methods = args.methods
    if len(set(methods)) != len(methods):
        raise ValueError("DUPLICATE_METHOD")
    proofs = {method: [] for method in methods}
    jobs = {}
    for method in methods:
        status = root / "jobs" / ("train_replay_" + method + ".status.json")
        job = read(status)
        if job["status"] != "COMPLETE" or job["exit_code"] != 0 or job["source_sha256"] != source:
            raise ValueError("SUCCESSFUL_SOURCE_BOUND_REPLAY_REQUIRED: " + method)
        jobs[method] = {"job_status_sha256": sha(status), "source_sha256": job["source_sha256"],
                        "worker_pid": job["worker_pid"], "exit_code": job["exit_code"]}
        scenes = {p.name for p in (root / "train_replay" / method).glob("scene_*")}
        if scenes != {f"scene_{sid:02d}" for sid in range(1, 25)}:
            raise ValueError("ACTUAL_REPLAY_SCENE_SCOPE_MISMATCH")
    for sid in range(1, 25):
        timestamps = ds.frames(sid)
        cache_path = root / "cache_train_v2" / f"scene_{sid:02d}" / "manifest.json"
        cache = read(cache_path)
        if cache["status"] != "COMPLETE" or cache["timestamps"] != len(timestamps):
            raise ValueError("CACHE_METADATA_TIMESTAMP_COUNT_MISMATCH")
        manifests = {}
        digests = {method: hashlib.sha256() for method in methods}
        detections = {method: 0 for method in methods}
        with ExitStack() as stack:
            streams = {}
            for method in methods:
                directory = root / "train_replay" / method / f"scene_{sid:02d}"
                manifest = read(directory / "manifest.json")
                if (manifest["status"] != "COMPLETE" or manifest["method_id"] != method
                        or manifest["GT_in_inference"] is not False
                        or manifest["timestamps"] != len(timestamps)):
                    raise ValueError("REPLAY_MANIFEST_MISMATCH")
                manifests[method] = manifest
                streams[method] = stack.enter_context((directory / "tracks.jsonl").open("rb"))
            for timestamp in timestamps:
                reference = None
                for method in methods:
                    line = streams[method].readline()
                    if not line:
                        raise ValueError("MISSING_DELIVERED_TIMESTAMP")
                    digests[method].update(line)
                    row = json.loads(line)
                    if (row["timestamp"] != timestamp or row["scene_id"] != sid
                            or row["method_id"] != method or row["GT_in_inference"] is not False):
                        raise ValueError("DELIVERED_OUTPUT_METADATA_MISMATCH")
                    candidates = current_candidates(row)
                    if reference is not None and candidates != reference:
                        raise ValueError("CROSS_METHOD_CURRENT_BOX_OR_SCORE_DIFFERENCE")
                    reference = candidates
                    detections[method] += len(candidates)
            if any(stream.readline() for stream in streams.values()):
                raise ValueError("EXTRA_DELIVERED_TIMESTAMPS")
        for method in methods:
            digest = digests[method].hexdigest()
            if digest != manifests[method]["tracks_sha256"]:
                raise ValueError("DELIVERED_TRACKS_CHECKSUM_MISMATCH")
            if detections[method] != cache["counts"]["observations"]:
                raise ValueError("DELIVERED_CANDIDATE_COUNT_MISMATCH")
            proofs[method].append({"scene": sid, "timestamps": len(timestamps),
                "current_detections": detections[method], "tracks_sha256": digest,
                "cache_manifest_sha256": sha(cache_path)})
        print(json.dumps({"verified_full_TRAIN_scene": sid, "methods": len(methods)}), flush=True)
    for method in methods:
        if sum(row["timestamps"] for row in proofs[method]) != 28754:
            raise ValueError("ACTUAL_TRAIN_TIMESTAMP_COVERAGE_MISMATCH")
    report = {"status": "FULL_TRAIN_CONTROL_OUTPUTS_VERIFIED", "observed_at_unix": time.time(),
        "verification_script_sha256": sha(Path(__file__)), "source_sha256": source,
        "metadata_sha256": sha(metadata), "GT_labels_opened": False,
        "current_boxes_scores_and_keys_equal_across_methods": len(methods) > 1,
        "checks": ["successful source-bound worker exit", "all24 actual scene directories",
                   "every delivered timestamp against annotation-free metadata",
                   "every delivered row parsed and checksummed", "unique detection and within-camera ID keys",
                   "finite positive current visible output boxes", "cross-method exact candidate keys/boxes/scores",
                   "actual per-scene detection totals against frozen perception manifests"],
        "methods": {method: {"scene_count": len(proofs[method]), "timestamps": 28754,
            "current_detections": sum(row["current_detections"] for row in proofs[method]),
            "job": jobs[method], "scenes": proofs[method]} for method in methods},
        "neural_training": False, "official_metric": False,
        "speed_claim": "None; this verifies cached causal TRAIN controls, not live RGB-to-ID speed."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
