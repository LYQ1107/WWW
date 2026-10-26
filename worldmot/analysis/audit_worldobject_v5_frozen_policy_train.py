#!/usr/bin/env python3
"""Registered TRAIN6 autonomous C1/B1 loops, without training or TEST access.

Unlike the conditional birth diagnostic, each method maintains its own genuine
predicted history. This audits the complete frozen B1 policy separation, not a
birth-only causal effect or a held-out benchmark of the frozen R2 identity head.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import torch

from worldmot.runtime.world_object_v5_contract import require_v5_scope, verify_scene_scope, atomic_json, v5_source_fingerprint
from worldmot.runtime.world_object_v4_test_contract import observation_signature
from worldmot.runtime.world_object_tracking_core import WorldObjectTrackingCore
from worldmot.runtime.world_object_v5_core import ObservationAnchoredWorldObjectCore
from worldmot.tracking.world_object_v4_variants import state_configs
from worldmot.tracking.world_object_v5_config import registered_config
from worldmot.models.world_object_v4_checkpoint import load_frozen_r2_head
from worldmot.models.world_object_v5_geometry import v5_observations
from worldmot.models.gmt_detection_cache import ImmutableGMTDetectionCache
from worldmot.data.worldobject_cache import WorldObjectCache, sha256
from worldmot.data.worldobject_observations import replay_observations
from worldmot.data.visiontrack import VisionTrack
from worldmot.training.world_object_identity_targets import match_train_identities, identity_key
from attribute_worldobject_v5_frozen_test import DeliveredIdentityAudit, matrix_contrast
from worldobject_v5_audit_common import compare_events


def main():
    p = argparse.ArgumentParser()
    for key in ("scope-lock", "authorization", "calibration", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--scene", type=int, choices=(4, 8, 12, 16, 20, 24), required=True)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    scope, approval = require_v5_scope(a.scope_lock, a.authorization, "diagnostics", "train", "B1")
    a.output.mkdir(parents=True, exist_ok=False)
    source, script = v5_source_fingerprint(), sha256(__file__)
    torch.set_num_threads(1)
    raw = ImmutableGMTDetectionCache(scope["splits"]["train"]["raw_cache"])
    gate = Path(scope["splits"]["train"]["support_cache"]).parent / "train/FULL_WORLD_OBJECT_V3/train_gate.json"
    model, checkpoint = load_frozen_r2_head(gate, raw.lock, a.device)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("FROZEN_IDENTITY_HEAD_REQUIRED")
    calibration = json.loads(a.calibration.read_text())
    cfg = registered_config("B1", calibration)
    cores = {"C1": WorldObjectTrackingCore(state_configs(1024)["C1"], model, a.device),
             "B1": ObservationAnchoredWorldObjectCore(cfg, model, a.device)}
    for core in cores.values():
        core.reset(a.scene)
    dataset = VisionTrack(scope["dataset_root"], split="train", load_annotations=True)
    if sha256(dataset.annotation_path) != calibration["TRAIN_annotations_sha256"]:
        raise ValueError("ORIGINAL_TRAIN_ANNOTATIONS_CHANGED")
    timeline = {pair.frame_id: pair for pair in dataset.synchronized_pairs(a.scene)}
    cache = WorldObjectCache(scope["splits"]["train"]["support_cache"], a.scene, raw_cache=raw)
    expected = verify_scene_scope(scope, "train", a.scene, cache)
    audits = {key: DeliveredIdentityAudit() for key in cores}
    paired, gt_visible, example_counts = Counter(), Counter(), Counter()
    records, examples, seen = [], [], []
    started = time.perf_counter()
    with (a.output / "decisions.jsonl").open("x") as log:
        for step, frame in enumerate(cache.frames()):
            observations = sorted(replay_observations(frame), key=lambda o: o.key)
            current, context = v5_observations(frame)
            current = sorted(current, key=lambda o: o.key)
            if observation_signature(observations) != observation_signature(current):
                raise ValueError("C1_B1_FROZEN_INPUT_DIFFERENCE")
            t = frame["timestamp"]
            rows = {"C1": cores["C1"].step(observations, t), "B1": cores["B1"].step(current, t, context)}
            # All GT is fetched after BOTH independent causal decisions.
            gt = dataset.ground_truth_pair(timeline[t])
            labels, match = match_train_identities(observations, gt)
            for camera, annotations in gt.items():
                for label, count in Counter(identity_key(x) for x in annotations).items():
                    if label is not None and count == 1:
                        gt_visible[int(camera), label] += 1
            events, assigned = {}, {}
            for key, audit in audits.items():
                events[key], assigned[key] = audit.observe(rows[key], observations, labels, step, len(timeline), gt_visible)
                audit.counts.update({"GT_join/" + k: n for k, n in match.items()})
            for i, left in enumerate(observations):
                if labels[i] is None:
                    continue
                for j in range(i + 1, len(observations)):
                    right = observations[j]
                    if labels[j] is None or left.camera_id == right.camera_id:
                        continue
                    same = labels[i] == labels[j]
                    kind = "same_person_shared_ID" if same else "different_person_separate_ID"
                    for key in cores:
                        ga, gb = assigned[key][left.key], assigned[key][right.key]
                        good = ga is not None and gb is not None and (ga == gb) == same
                        events[key][kind, (left.key, right.key)] = good
                        audits[key].counts[kind + "/opportunities"] += 1
                        audits[key].counts[kind + "/correct"] += int(good)
                records.append((labels[i], assigned["C1"][left.key], assigned["B1"][left.key]))
            changed = compare_events(events["C1"], events["B1"], paired)
            for change in changed:
                sample = change["kind"], change["correction"]
                if example_counts[sample] < 8:
                    examples.append({"scene": a.scene, "timestamp": t, **change})
                    example_counts[sample] += 1
            log.write(json.dumps({"timestamp": t, "keys": [o.key for o in observations],
                "delivered_IDs": {key: [assigned[key][o.key] for o in observations] for key in cores},
                "labels_used_post_decision_only": labels, "geometry_reprojection_status": context.reprojection_status,
                "bank_counts": {key: row["bank"] for key, row in rows.items()}}, allow_nan=False) + "\n")
            seen.append(t)
            if len(seen) % 128 == 0:
                log.flush()
                atomic_json(a.output / "progress.json", {"scene": a.scene, "timestamps": len(seen), "expected": expected["timestamps"], "TRAIN_only": True})
    if seen != list(timeline) or len(seen) != expected["timestamps"] or source != v5_source_fingerprint() or script != sha256(__file__):
        raise ValueError("COMPLETE_UNCHANGED_TRAIN_CLOSED_LOOPS_REQUIRED")
    graphs, maps = {}, {}
    for key, audit in audits.items():
        graphs[key], maps[key] = audit.graph()
    good0 = np.asarray([gid is not None and maps["C1"].get(label) == gid for label, gid, _ in records])
    good1 = np.asarray([gid is not None and maps["B1"].get(label) == gid for label, _, gid in records])
    global_pairs = Counter()
    matrix_contrast(global_pairs, good0, good1, np.ones(good0.shape, bool))
    atomic_json(a.output / "report.json", {"status": "COMPLETE_TRAIN_ONLY_FROZEN_POLICY_CLOSED_LOOPS",
        "scene": a.scene, "timestamps": len(seen), "counts": {k: dict(v.counts) for k, v in audits.items()},
        "paired_B1_minus_C1": dict(paired), "global_one_to_one_paired": dict(global_pairs), "identity_graphs": graphs,
        "examples": examples, "v5_source_sha256": source, "analysis_sha256": script, "checkpoint": checkpoint,
        "scope_sha256": sha256(a.scope_lock), "approval": approval, "device": a.device, "precision": "float32",
        "decision_log_sha256": sha256(a.output / "decisions.jsonl"), "wall_s": time.perf_counter() - started,
        "TEST_read": False, "optimizer_created": False, "independent_predicted_histories": True,
        "GT_repairs_history": False, "birth_only_causal_effect": False, "official_TrackEval_or_CVIDF1": False,
        "frozen_R2_training_included_audited_scenes": True, "model_or_budget_selection_uses_this_audit": False})
    print(json.dumps({"status": "COMPLETE_TRAIN_ONLY_FROZEN_POLICY_CLOSED_LOOPS", "scene": a.scene, "timestamps": len(seen)}), flush=True)


if __name__ == "__main__":
    main()
