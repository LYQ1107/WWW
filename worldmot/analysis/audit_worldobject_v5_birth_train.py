#!/usr/bin/env python3
"""Finite TRAIN6 same-C1-history attribution; no optimizer or TEST inputs.

Only the original frozen C1 prediction advances the real bank. Counterfactual
decisions are discarded after audit. This isolates birth under fixed history,
and separately removal of birth competition from the legacy history solver;
it is not the online B1 result or a formal held-out tracking metric.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from worldmot.runtime.world_object_v5_contract import require_v5_scope, verify_scene_scope, v5_source_fingerprint, atomic_json
from worldmot.runtime.world_object_v4_test_contract import observation_signature
from worldmot.runtime.world_object_tracking_core import WorldObjectTrackingCore
from worldmot.tracking.world_object_v4_variants import state_configs
from worldmot.tracking.world_object_v5_config import registered_config
from worldmot.tracking.joint_world_object_assignment import JointWorldObjectAssignment
from worldmot.tracking.world_object_birth_manager import WorldObjectBirthManager
from worldmot.models.world_object_v4_checkpoint import load_frozen_r2_head
from worldmot.models.world_object_v5_geometry import v5_observations
from worldmot.models.gmt_detection_cache import ImmutableGMTDetectionCache
from worldmot.data.worldobject_cache import WorldObjectCache, sha256
from worldmot.data.worldobject_observations import replay_observations
from worldmot.data.visiontrack import VisionTrack
from worldmot.training.world_object_identity_targets import match_train_identities
from worldobject_v5_audit_common import decision_events, compare_events, plan_mapping


VAL = (4, 8, 12, 16, 20, 24)
METHODS = ("ORIGINAL_JOINT", "INDEPENDENT_BIRTH_FIXED_ORIGINAL_HISTORY", "LEGACY_HISTORY_ONLY_THEN_INDEPENDENT_BIRTH")


class HistoricalOptionsOnly(JointWorldObjectAssignment):
    def _pair_evidence(self, a, b, birth=False, learned_probability=None):
        if birth:
            return None
        return super()._pair_evidence(a, b, birth=False, learned_probability=learned_probability)


def numpy_evidence(outputs):
    if outputs is None:
        return None
    keys = {"assignment_logits", "null_logits", "existence_logits", "birth_logits", "birth_pair_logits"}
    return {k: v.detach().cpu().numpy() for k, v in outputs.items() if k in keys}


def independent_plan(manager, observations, historical_options, context, history_hints):
    assigned = {j: int(option.global_id) for option in historical_options for j in option.indices}
    unmatched = [j for j in range(len(observations)) if j not in assigned]
    birth = manager.form(observations, unmatched, assigned, context, history_hint=history_hints)
    mapping = plan_mapping(list(historical_options) + birth.selected, duplicates=birth.duplicates)
    if set(mapping) != set(range(len(observations))):
        raise ValueError("COUNTERFACTUAL_MISSING_DETECTION_DECISION")
    return mapping, birth


def verify_original_decisions(mapping, observations, actual, next_id):
    assigned = [actual[o.key] for o in observations]
    for j, token in mapping.items():
        if token[0] == "H" and assigned[j] != token[1]:
            raise ValueError("COUNTERFACTUAL_CHANGED_REAL_ORIGINAL_HISTORY")
        if token[0] == "N" and assigned[j] < next_id:
            raise ValueError("ORIGINAL_BIRTH_WAS_NOT_A_NEW_ID")
    for j in range(len(observations)):
        for k in range(j + 1, len(observations)):
            if (mapping[j] == mapping[k]) != (assigned[j] == assigned[k]):
                raise ValueError("ORIGINAL_PARTITION_NOT_REPRODUCED")


def main():
    p = argparse.ArgumentParser()
    for key in ("scope-lock", "authorization", "calibration", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--scene", type=int, choices=VAL, required=True)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    scope, approval = require_v5_scope(a.scope_lock, a.authorization, "diagnostics", "train", "B1")
    a.output.mkdir(parents=True, exist_ok=False)
    source = v5_source_fingerprint()
    script_sha = sha256(__file__)
    torch.set_num_threads(1)
    raw = ImmutableGMTDetectionCache(scope["splits"]["train"]["raw_cache"])
    gate = Path(scope["splits"]["train"]["support_cache"]).parent / "train/FULL_WORLD_OBJECT_V3/train_gate.json"
    model, weight = load_frozen_r2_head(gate, raw.lock, a.device)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("FROZEN_C1_REQUIRED")
    core = WorldObjectTrackingCore(state_configs(1024)["C1"], model, a.device)
    core.reset(a.scene)
    cfg = registered_config("B1", json.loads(a.calibration.read_text()))
    manager = WorldObjectBirthManager(cfg)
    conditional = HistoricalOptionsOnly(core.config)
    dataset = VisionTrack(scope["dataset_root"], split="train", load_annotations=True)
    calibration = json.loads(a.calibration.read_text())
    if sha256(dataset.annotation_path) != calibration["TRAIN_annotations_sha256"]:
        raise ValueError("TRAIN_ANNOTATION_BYTES_CHANGED")
    pairs = {pair.frame_id: pair for pair in dataset.synchronized_pairs(a.scene)}
    cache = WorldObjectCache(scope["splits"]["train"]["support_cache"], a.scene, raw_cache=raw)
    expected = verify_scene_scope(scope, "train", a.scene, cache)
    people = defaultdict(set)
    method_counts = {name: Counter() for name in METHODS}
    contrasts = {name: Counter() for name in METHODS[1:]}
    examples, seen, general = [], [], Counter()
    example_counts = Counter()
    started = time.perf_counter()
    with (a.output / "decisions.jsonl").open("x") as stream:
        for frame in cache.frames():
            observations = sorted(replay_observations(frame), key=lambda o: o.key)
            birth_observations, context = v5_observations(frame, True)
            birth_observations = sorted(birth_observations, key=lambda o: o.key)
            if observation_signature(observations) != observation_signature(birth_observations):
                raise ValueError("FROZEN_DETECTION_OR_ROI_CHANGED_IN_BIRTH_ADAPTER")
            timestamp = frame["timestamp"]
            evidence = core.forward_evidence(observations, timestamp)
            learned = numpy_evidence(evidence)
            history = core.bank.active()
            active = {q.global_id for q in history}
            next_id = core.bank.next_id
            original = core.bank.assigner.assign(history, observations, learned)
            original_mapping = plan_mapping(original.selected, original.unmatched)
            history_only = conditional.assign(history, observations, learned)
            if any(option.global_id is None for option in history_only.selected):
                raise ValueError("BIRTH_OPTION_ENTERED_HISTORY_ONLY_OPTIMIZER")
            hints = {}
            if learned is not None and history:
                margins = learned["assignment_logits"] - learned["null_logits"][:, None]
                for j in range(len(observations)):
                    order = np.argsort(-margins[:, j], kind="stable")
                    best = int(order[0])
                    competitor = float(margins[order[1], j]) if len(order) > 1 else 0.
                    if margins[best, j] >= cfg.identity_write_min_margin and margins[best, j] - competitor >= cfg.identity_write_competition_margin:
                        hints[j] = history[best].global_id
            fixed_history = [option for option in original.selected if option.global_id is not None]
            fixed_mapping, fixed_birth = independent_plan(manager, birth_observations, fixed_history, context, hints)
            historical_mapping, historical_birth = independent_plan(manager, birth_observations, history_only.selected, context, hints)
            # Advance only the ORIGINAL model's genuine closed-loop decision.
            core.step(observations, timestamp, evidence=evidence)
            verify_original_decisions(original_mapping, observations, core.bank.last_assignments, next_id)
            # TRAIN GT is first requested after all real/counterfactual decisions.
            labels, target_audit = match_train_identities(observations, dataset.ground_truth_pair(pairs[timestamp]))
            general.update(target_audit)
            mappings = dict(zip(METHODS, (original_mapping, fixed_mapping, historical_mapping)))
            events = {}
            for name, mapping in mappings.items():
                events[name], counts = decision_events(observations, labels, mapping, people, active)
                method_counts[name].update(counts)
                method_counts[name]["output_candidates"] += sum(token is not None for token in mapping.values())
            for name in METHODS[1:]:
                changed = compare_events(events[METHODS[0]], events[name], contrasts[name])
                for change in changed:
                    key = (name, change["kind"], change["correction"])
                    if example_counts[key] < 5:
                        examples.append({"scene": a.scene, "timestamp": timestamp, "method": name, **change})
                        example_counts[key] += 1
            for o, label in zip(observations, labels):
                if label is not None:
                    people[core.bank.last_assignments[o.key]].add(label)
            general.update(timestamps=1, candidates=len(observations),
                           original_exact_partition_checks=1,
                           original_search_fallback=int("fallback" in original.diagnostics),
                           conditional_search_fallback=int("fallback" in history_only.diagnostics),
                           missing_world_camera_transform=int(context.reprojection_status == "WORLD_FROM_OMEGA_NOT_RECORDED"))
            stream.write(json.dumps({"timestamp": timestamp, "keys": [o.key for o in observations],
                "labels_used_after_decisions_only": labels,
                "methods": {name: [mapping[j] for j in range(len(observations))] for name, mapping in mappings.items()},
                "independent_birth_fixed_history": {k: v for k, v in fixed_birth.diagnostics.items() if k != "birth_pair_audit"},
                "independent_birth_history_only": {k: v for k, v in historical_birth.diagnostics.items() if k != "birth_pair_audit"}}, allow_nan=False) + "\n")
            seen.append(timestamp)
            if len(seen) % 128 == 0:
                stream.flush()
                atomic_json(a.output / "progress.json", {"scene": a.scene, "timestamps": len(seen), "expected": expected["timestamps"], "TRAIN_only": True})
    if seen != list(pairs) or len(seen) != expected["timestamps"] or source != v5_source_fingerprint():
        raise ValueError("COMPLETE_IMMUTABLE_TRAIN_SCENE_REQUIRED")
    report = {"status": "COMPLETE_TRAIN_ONLY_CONDITIONAL_BIRTH_DIAGNOSTIC", "scene": a.scene,
        "timestamps": len(seen), "counts": dict(general), "methods": {k: dict(v) for k, v in method_counts.items()},
        "versus_original": {k: dict(v) for k, v in contrasts.items()}, "examples": examples,
        "checkpoint": weight, "v5_source_sha256": source, "analysis_sha256": script_sha,
        "analysis_helper_sha256": sha256(Path(__file__).with_name("worldobject_v5_audit_common.py")),
        "raw_annotations_sha256": sha256(dataset.annotation_path), "support_manifest_sha256": sha256(cache.path / "manifest.json"),
        "decision_log_sha256": sha256(a.output / "decisions.jsonl"), "approval": approval, "wall_s": time.perf_counter() - started,
        "TEST_read": False, "optimizer_created": False, "original_history_advanced_only": True,
        "counterfactual_history_updates": False, "online_B1_result": False,
        "formal_TrackEval_or_CVIDF1": False, "heldout_from_original_R2_training": False,
        "legacy_history_only_uses_legacy_cap_behavior": True,
        "supervision_scope": "one-to-one clean original TRAIN box matches; ambiguous/missing labels excluded"}
    atomic_json(a.output / "report.json", report)
    print(json.dumps({"status": report["status"], "scene": a.scene, "timestamps": len(seen), "report_sha256": sha256(a.output / "report.json")}), flush=True)


if __name__ == "__main__":
    main()
