#!/usr/bin/env python3
"""One preregistered TRAIN6 snapshot audit; never a final model or TEST route."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import time

import numpy as np
import torch

from worldmot.data.visiontrack import VisionTrack
from worldmot.data.worldobject_cache import WorldObjectCache, sha256
from worldmot.models.gmt_detection_cache import ImmutableGMTDetectionCache
from worldmot.models.world_object_membership_head import WorldObjectMembershipHead
from worldmot.models.world_object_v5_geometry import v5_observations
from worldmot.runtime.world_object_v5_contract import (
    atomic_json, require_v5_scope, state_tensor_sha256, v5_source_fingerprint,
    verify_scene_scope)
from worldmot.runtime.world_object_v5_core import ObservationAnchoredWorldObjectCore
from worldmot.tracking.world_object_v5_config import registered_config
from worldmot.training.world_object_identity_targets import match_train_identities, identity_key
from attribute_worldobject_v5_frozen_test import DeliveredIdentityAudit
from worldobject_v5_identity_summary import rates

SCENES = [4, 8, 12, 16, 20, 24]
TRAIN_SHA = "9093c36204bae482c2f74997f531bfe2d8a46464f4541093703ef22db0068c85"
INITIAL = "3fb3f3d6336e8aa32d635ecff3c05929b0d4b8f265f4a0ccf56771814d13ed23"


def validate_registration(registration, actual_hashes):
    if (registration.get("status") != "PREREGISTERED_FINITE_TRAIN_SNAPSHOT_AUDIT"
            or registration.get("variant") != "B3" or registration.get("split") != "train"
            or registration.get("scenes") != SCENES
            or registration.get("expected_paired_times") != 5860
            or registration.get("expected_candidates") != 172760
            or registration.get("device") != "cpu" or registration.get("threads") != 1
            or registration.get("TEST_read") is not False
            or registration.get("optimizer_created") is not False
            or registration.get("model_or_budget_selection_uses_this_audit") is not False
            or registration.get("formal_metric") is not False
            or registration.get("TRAIN_annotations_sha256") != TRAIN_SHA
            or registration.get("hashes") != actual_hashes):
        raise PermissionError("EXACT_PREREGISTERED_TRAIN6_CPU_SNAPSHOT_SCOPE_REQUIRED")


def snapshot_payload(path, expected_sha):
    if sha256(path) != expected_sha:
        raise ValueError("SNAPSHOT_SHA_MISMATCH_BEFORE_MODEL_LOAD")
    # This file is our own registered, SHA-checked training recovery payload.
    raw = Path(path).read_bytes()
    import hashlib
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError("SNAPSHOT_CHANGED_DURING_READ")
    return torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)


def validate_publication(receipt, registration_sha, registration):
    if (receipt.get("status") != "GITHUB_REF_TREE_AND_PUBLIC_FETCH_VERIFIED_FOR_TRAIN_SNAPSHOT_AUDIT"
            or receipt.get("registration_sha256") != registration_sha
            or receipt.get("snapshot_sha256") != registration["snapshot_sha256"]
            or receipt.get("driver_sha256") != registration["hashes"]["driver_sha256"]
            or len(receipt.get("remote_commit", "")) != 40):
        raise PermissionError("PUBLISHED_TRAIN_AUDIT_REGISTRATION_REQUIRED_BEFORE_MODEL_LOAD")


def cross_view_counts(counts, observations, labels, assigned):
    views = [[i for i, o in enumerate(observations) if o.camera_id == v and labels[i] is not None]
             for v in (1, 2)]
    if not all(views):
        return
    first, second = views
    same = np.asarray([labels[i] for i in first], object)[:, None] == np.asarray(
        [labels[i] for i in second], object)[None, :]
    left = np.asarray([assigned[observations[i].key] if assigned[observations[i].key] is not None else -1
                       for i in first])
    right = np.asarray([assigned[observations[i].key] if assigned[observations[i].key] is not None else -2
                        for i in second])
    valid = (left[:, None] >= 0) & (right[None, :] >= 0)
    shared = left[:, None] == right[None, :]
    counts.update(same_person_cross_view_pairs=int(same.sum()),
                  different_person_cross_view_pairs=int((~same).sum()),
                  correct_same_person_shared_ID=int((valid & shared & same).sum()),
                  same_person_ID_split=int((valid & ~shared & same).sum()),
                  false_different_person_shared_ID=int((valid & shared & ~same).sum()),
                  correct_different_person_separate=int((valid & ~shared & ~same).sum()),
                  missing_output_cross_view_pair_opportunities=int((~valid).sum()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--registration", type=Path, required=True)
    args = parser.parse_args()
    registration = json.loads(args.registration.read_text())
    analysis = Path(__file__).resolve().parent
    hashes = {"v5_source_sha256": v5_source_fingerprint(), "driver_sha256": sha256(__file__),
              "event_audit_sha256": sha256(analysis / "attribute_worldobject_v5_frozen_test.py"),
              "scope_sha256": sha256(registration["scope_lock"]),
              "authorization_sha256": sha256(registration["authorization"]),
              "calibration_sha256": sha256(registration["calibration"])}
    validate_registration(registration, hashes)
    registration_sha = sha256(args.registration)
    publication = json.loads(Path(registration["publication_receipt"]).read_text())
    validate_publication(publication, registration_sha, registration)
    scope, approval = require_v5_scope(registration["scope_lock"], registration["authorization"],
                                       "diagnostics", "train", "B3")
    calibration = json.loads(Path(registration["calibration"]).read_text())
    config = registered_config("B3", calibration)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    payload = snapshot_payload(registration["snapshot"], registration["snapshot_sha256"])
    progress = payload["progress"]
    if (payload.get("tracking_config") != config.to_dict()
            or payload.get("initial_state_tensor_sha256") != INITIAL
            or len(progress["history"]) != registration["snapshot_completed_epochs"]
            or progress["epoch"] != registration["snapshot_partial_epoch"]
            or [x["scene"] for x in progress["epoch_scenes"]] != registration["snapshot_partial_epoch_scenes"]):
        raise ValueError("REGISTERED_SNAPSHOT_CONFIGURATION_OR_TRAIN_PROGRESS_CHANGED")
    model = WorldObjectMembershipHead(config.feature_dim, config.hidden_dim, config.geometry_dim,
                                      config.pair_chunk, config.feature_dropout)
    model.load_state_dict(payload["model"], strict=True)
    del payload
    model.eval().requires_grad_(False)
    tensor_sha = state_tensor_sha256(model)
    if tensor_sha != registration["model_state_tensor_sha256"] or sum(p.numel() for p in model.parameters()) != 606468:
        raise ValueError("EXACT_FROZEN_SNAPSHOT_TENSORS_REQUIRED")
    output = Path(registration["output"])
    output.mkdir(parents=True, exist_ok=False)
    raw = ImmutableGMTDetectionCache(scope["splits"]["train"]["raw_cache"])
    dataset = VisionTrack(scope["dataset_root"], split="train", load_annotations=True)
    if sha256(dataset.annotation_path) != TRAIN_SHA:
        raise ValueError("ORIGINAL_TRAIN_ANNOTATIONS_CHANGED")
    totals, per_scene, examples, sample_counts = Counter(), {}, [], Counter()
    started = time.perf_counter()
    with torch.inference_mode():
        for scene in SCENES:
            core = ObservationAnchoredWorldObjectCore(config, model, "cpu")
            core.reset(scene)
            audit, visible, seen = DeliveredIdentityAudit(), Counter(), []
            timeline = {p.frame_id: p for p in dataset.synchronized_pairs(scene)}
            cache = WorldObjectCache(scope["splits"]["train"]["support_cache"], scene, raw_cache=raw)
            expected = verify_scene_scope(scope, "train", scene, cache)
            folder = output / ("scene_%02d" % scene)
            folder.mkdir()
            log_path = folder / "decisions.jsonl"
            with log_path.open("x") as log:
                for step, frame in enumerate(cache.frames()):
                    observations, context = v5_observations(frame, True)
                    observations = sorted(observations, key=lambda o: o.key)
                    timestamp = frame["timestamp"]
                    row = core.step(observations, timestamp, context)
                    # Labels are fetched only after the causal decision has been delivered.
                    gt = dataset.ground_truth_pair(timeline[timestamp])
                    labels, join = match_train_identities(observations, gt)
                    for camera, annotations in gt.items():
                        for label, count in Counter(identity_key(x) for x in annotations).items():
                            if label is not None and count == 1:
                                visible[int(camera), label] += 1
                    events, assigned = audit.observe(row, observations, labels, step, len(timeline), visible)
                    audit.counts.update({"GT_join/" + k: n for k, n in join.items()})
                    cross_view_counts(audit.counts, observations, labels, assigned)
                    for (kind, key), correct in events.items():
                        if not correct and kind in ("wrong_historical_ID_reuse_avoided", "mixed_history_reuse_avoided",
                                "false_birth_proxy_avoided", "duplicate_birth_avoided", "LOST_recovery_correct") and sample_counts[kind] < 8:
                            examples.append({"scene": scene, "timestamp": timestamp, "kind": kind, "key": key})
                            sample_counts[kind] += 1
                    log.write(json.dumps({"timestamp": timestamp, "keys": [o.key for o in observations],
                        "delivered_IDs": [assigned[o.key] for o in observations],
                        "GT_labels_post_decision_only": labels,
                        "decision_stages": [d.get("stage") for d in row["decisions"]],
                        "bank_counts": row["bank"],
                        "geometry_reprojection_status": context.reprojection_status}, allow_nan=False) + "\n")
                    seen.append(timestamp)
                    if len(seen) % 128 == 0:
                        log.flush()
                        atomic_json(output / "progress.json", {"scene": scene, "paired_times": len(seen),
                            "scene_expected_times": expected["timestamps"], "complete_scenes": len(per_scene),
                            "TRAIN_only": True, "snapshot_sha256": registration["snapshot_sha256"]})
            if (seen != list(timeline) or len(seen) != expected["timestamps"]
                    or audit.counts["frozen_candidates"] != expected["candidates"]
                    or core.scene_reset_count != 1 or v5_source_fingerprint() != hashes["v5_source_sha256"]):
                raise ValueError("COMPLETE_UNCHANGED_CAUSAL_TRAIN_SCENE_REQUIRED")
            record = {"scene": scene, "paired_times": len(seen), "counts": dict(audit.counts),
                      "decision_log_sha256": sha256(log_path), "scene_origin_resets": 1}
            atomic_json(folder / "report.json", record)
            per_scene[str(scene)] = record
            totals.update(audit.counts)
            print(json.dumps({"event": "TRAIN_SNAPSHOT_SCENE_COMPLETE", "scene": scene,
                              "paired_times": len(seen), "complete_scenes": len(per_scene)}), flush=True)
    if (totals["paired_timestamps"] != 5860 or totals["frozen_candidates"] != 172760
            or state_tensor_sha256(model) != tensor_sha or sha256(args.registration) != registration_sha
            or sha256(__file__) != hashes["driver_sha256"]
            or sha256(analysis / "attribute_worldobject_v5_frozen_test.py") != hashes["event_audit_sha256"]
            or sha256(registration["snapshot"]) != registration["snapshot_sha256"]
            or sha256(dataset.annotation_path) != TRAIN_SHA):
        raise ValueError("COMPLETE_FIXED_TRAIN6_AND_UNMODIFIED_SNAPSHOT_REQUIRED")
    report = {"status": "COMPLETE_TRAIN6_FROZEN_B3_SNAPSHOT_IDENTITY_MONITOR", "UTC": datetime.now(timezone.utc).isoformat(),
        "registration_sha256": registration_sha, "registration_publication_commit": publication["remote_commit"],
        "approval": approval, "hashes": hashes,
        "snapshot_sha256": registration["snapshot_sha256"], "model_state_tensor_sha256": tensor_sha,
        "snapshot_completed_epochs": registration["snapshot_completed_epochs"],
        "snapshot_partial_epoch": registration["snapshot_partial_epoch"],
        "snapshot_partial_epoch_scenes": registration["snapshot_partial_epoch_scenes"],
        "scenes": SCENES, "counts": dict(totals), "rates": rates(totals), "per_scene": per_scene, "examples": examples,
        "TRAIN_seen_by_snapshot": True, "GT_role": "POST_DECISION_DIAGNOSTIC_ONLY", "GT_repairs_history": False,
        "TEST_read": False, "formal_TrackEval_or_CVIDF1": False, "optimizer_created": False,
        "model_parameters_modified": False, "model_or_budget_selection_uses_this_audit": False,
        "no_method_comparison_or_geometry_benefit_claim": True, "training_steps_retroactively_reconstructed": False,
        "device": "cpu", "threads": 1, "wall_s_including_GT_audit_and_output": time.perf_counter() - started,
        "timing_is_not_a_registered_inference_FPS_comparison": True}
    atomic_json(output / "report.json", report)
    print(json.dumps({"status": report["status"], "paired_times": totals["paired_timestamps"]}), flush=True)


if __name__ == "__main__":
    main()
