#!/usr/bin/env python3
"""Read-only full22 identity attribution after all five V5 outputs freeze.

Diagnostics use the unchanged clean IoU=.5 annotation join. Native TrackEval
and MATLAB remain the formal metrics. Dropped detections stay in the common
event denominator and cannot become successful identity decisions.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np

from worldmot.runtime.world_object_v5_contract import require_v5_scope, atomic_json, LEGAL_TEST, v5_source_fingerprint
from worldmot.data.worldobject_cache import sha256
from worldmot.runtime.world_object_v4_test_contract import output_candidates
from worldmot.training.world_object_identity_targets import match_evaluation_identities, identity_key
from attribute_worldobject_v3_completed_predictions import identity_graph
from worldobject_v5_audit_common import compare_events

VARIANTS = ("B0", "B1", "B2", "B3", "B4")
CONTRASTS = (("B0", "B1"), ("B1", "B2"), ("B2", "B3"), ("B4", "B3"), ("B0", "B3"))
ANNOTATIONS = "7a1e735bf25026a56148deaeb432e0520c6ad5dad97d5362aa58739255389593"


class DeliveredIdentityAudit:
    def __init__(self):
        self.active = {}
        self.people = defaultdict(Counter)
        self.person_ids = defaultdict(Counter)
        self.seen = set()
        self.previous = {}
        self.counts = Counter()
        self.quarters = [Counter() for _ in range(4)]

    def observe(self, row, observations, labels, step, total, gt_visible):
        counts = self.counts
        boxes = output_candidates(row)
        decisions = {(int(x["camera_id"]), int(x["detection_id"])): x for x in row["decisions"]}
        keys = {o.key for o in observations}
        if set(decisions) != keys or not set(boxes).issubset(keys):
            raise ValueError("COMMON_FROZEN_CANDIDATE_DECISIONS_REQUIRED")
        assigned = {}
        for o in observations:
            item = boxes.get(o.key)
            if item is None:
                if decisions[o.key].get("stage") != "TRACK_AWARE_DUPLICATE":
                    raise ValueError("MISSING_OUTPUT_WITHOUT_DUPLICATE_AUDIT")
                assigned[o.key] = None
            else:
                if item["bbox_xywh"] != o.bbox_xywh.tolist() or item["score"] != o.score:
                    raise ValueError("FROZEN_CURRENT_BOX_OR_SCORE_CHANGED")
                assigned[o.key] = int(item["global_id"])
        current_objects = {int(x["global_id"]): x for x in row["tracks_3d"]}
        pure_prior = defaultdict(list)
        for gid in self.active:
            if len(self.people[gid]) == 1:
                pure_prior[next(iter(self.people[gid]))].append(gid)
        events = {}
        grouped = defaultdict(Counter)
        new_by_person = defaultdict(set)
        null_checked = set()
        counts.update(paired_timestamps=1, frozen_candidates=len(observations), output_candidates=len(boxes),
                      audited_duplicate_suppression=len(observations) - len(boxes))
        for o, label in zip(observations, labels):
            if label is None:
                continue
            counts["clean_labelled_frozen_candidates"] += 1
            gid = assigned[o.key]
            present = gid is not None
            counts["clean_labelled_candidates_retained" if present else "clean_labelled_candidates_dropped"] += 1
            events["output_coverage", o.key] = present
            if not present:
                continue
            old = self.people[gid]
            born = gid not in self.seen
            counts["new_ID_clean_observations" if born else "historical_ID_clean_observations"] += 1
            wrong = bool(old) and label not in old
            counts["wrong_historical_ID_reuse"] += int(wrong)
            counts["reuse_of_cumulatively_mixed_ID"] += int(len(old) > 1)
            events["wrong_historical_ID_reuse_avoided", o.key] = not wrong
            events["mixed_history_reuse_avoided", o.key] = len(old) <= 1
            unique = len(pure_prior[label]) == 1
            counts["false_birth_unique_active_history_observations"] += int(born and unique)
            events["false_birth_proxy_avoided", o.key] = not (born and unique)
            if unique:
                events["false_birth_common_unique_history_avoided", o.key] = not born
                prior_gid = pure_prior[label][0]
                if label not in null_checked:
                    null_checked.add(label)
                    current = current_objects.get(prior_gid)
                    if current is None:
                        counts["unique_prior_history_state_missing_now"] += 1
                    else:
                        is_null = current.get("missed", 0) > 0
                        events["NULL_despite_current_person_avoided", label] = not is_null
                        counts["NULL_unique_active_history_opportunities"] += 1
                        counts["NULL_despite_current_person"] += int(is_null)
            if gid in self.active and self.active[gid].get("status") == "LOST":
                counts["LOST_recovery_clean_observations"] += 1
                correct = len(old) == 1 and label in old
                counts["LOST_recovery_correct_pure_identity"] += int(correct)
                events["LOST_recovery_correct", o.key] = correct
            key = o.camera_id, label
            previous = self.previous.get(key)
            if previous:
                old_gid, old_step, old_gt_count = previous
                gap = step - old_step
                correct = old_gid == gid
                counts["clean_per_camera_temporal_links"] += 1
                counts["diagnostic_per_camera_ID_changes"] += int(not correct)
                events["temporal_ID_retained", key] = correct
                if gap > 1:
                    suffix = "2to30" if gap <= 30 else "over30"
                    counts["detection_gap_returns_" + suffix] += 1
                    counts["detection_gap_retained_ID_" + suffix] += int(correct)
                    events["detection_gap_ID_retained_" + suffix, key] = correct
                    absent = gap - (gt_visible[key] - old_gt_count)
                    kind = "GT_absent_intervals" if absent > 0 else "GT_continuously_annotated"
                    counts["detection_gap_returns_" + kind] += 1
                    counts["detection_gap_retained_ID_" + kind] += int(correct)
                    events["detection_gap_ID_retained_" + kind, key] = correct
            self.previous[key] = gid, step, gt_visible[key]
            if born:
                new_by_person[label].add(gid)
            grouped[gid][label] += 1
            self.person_ids[label][gid] += 1
            state = current_objects.get(gid, {})
            counts["identity_write_timestamp_available_observations"] += int("last_identity_write" in state)
            counts["spatial_write_timestamp_available_observations"] += int("last_trusted_geometry_timestamp" in state)
            if wrong:
                counts["wrong_person_identity_write_observations"] += int(state.get("last_identity_write") == row["timestamp"])
                counts["wrong_person_current_spatial_write_observations"] += int(state.get("last_trusted_geometry_timestamp") == row["timestamp"])
        counts["within_timestamp_extra_birth_IDs_per_person"] += sum(max(0, len(ids) - 1) for ids in new_by_person.values())
        for gid, values in grouped.items():
            was_mixed = len(self.people[gid]) > 1
            self.people[gid].update(values)
            if len(self.people[gid]) > 1:
                counts["clean_observations_into_cumulatively_mixed_ID"] += sum(values.values())
                counts["identity_contamination_onsets"] += int(not was_mixed)
        self.seen.update(gid for gid in assigned.values() if gid is not None)
        self.active = current_objects
        counts["active_mixed_object_steps"] += sum(len(self.people[gid]) > 1 for gid in self.active)
        return events, assigned

    def graph(self):
        graph = identity_graph(self.person_ids)
        graph.update(mixed_output_IDs=sum(len(v) > 1 for v in self.people.values()),
                     minority_observations_in_mixed_IDs=sum(sum(v.values()) - max(v.values()) for v in self.people.values() if v))
        from scipy.optimize import linear_sum_assignment
        labels = sorted(self.person_ids)
        gids = sorted({g for ids in self.person_ids.values() for g in ids})
        index = {g: j for j, g in enumerate(gids)}
        matrix = np.zeros((len(labels), len(gids)), np.int64)
        for i, label in enumerate(labels):
            for gid, n in self.person_ids[label].items():
                matrix[i, index[gid]] = n
        rows, columns = linear_sum_assignment(-matrix)
        return graph, {labels[i]: gids[j] for i, j in zip(rows, columns)}


def matrix_contrast(counter, left, right, mask):
    a, b = left[mask], right[mask]
    counter["opportunities"] += int(a.size)
    counter["corrections"] += int((~a & b).sum())
    counter["regressions"] += int((a & ~b).sum())
    counter["net"] += int(b.sum()) - int(a.sum())


def global_retention_flags(records, variant_index, mapping):
    return np.asarray([gids[variant_index] is not None and mapping.get(label) == gids[variant_index]
                       for label, gids in records], dtype=bool)


def main():
    p = argparse.ArgumentParser()
    for key in ("scope-lock", "authorization", "output-root", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError("Keep completed posthoc reports immutable")
    scope, approval = require_v5_scope(a.scope_lock, a.authorization, "diagnostics", "test", "B3")
    source, analysis_sha = v5_source_fingerprint(), sha256(__file__)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from run_worldobject_v5_test import completed
    from worldobject_v5_reference_scope import require_canonical_or_adopted
    proofs = {v: [] for v in VARIANTS}
    # No annotations are opened until all 110 complete outputs pass checksums.
    for v in VARIANTS:
        for scene in range(1, 23):
            folder = a.output_root / f"predictions/{v}/scene_{scene:02d}"
            m = completed(folder, scope, v, scene)
            require_canonical_or_adopted(a.output_root, a.scope_lock, v, scene, m)
            proofs[v].append({"scene": scene, "timestamps": m["timestamps"],
                              "candidates": m["candidate_count"], "outputs": m["output_count"],
                              "tracks_sha256": m["output_sha256"]["tracks.jsonl"],
                              "original_scope_sha256": m["scope_sha256"]})
        if sum(x["timestamps"] for x in proofs[v]) != 29019 or sum(x["candidates"] for x in proofs[v]) != 548583:
            raise ValueError("ALL22_FROZEN_INPUTS_REQUIRED_FOR_EACH_VARIANT")
    from worldmot.data.visiontrack import VisionTrack
    annotation = Path(scope["dataset_root"]) / "annotations/test.json"
    if sha256(annotation) != ANNOTATIONS:
        raise ValueError("ORIGINAL_TEST_ANNOTATIONS_CHANGED")
    dataset = VisionTrack(scope["dataset_root"], split="test", load_annotations=True)
    started = time.perf_counter()
    totals = {label: {v: Counter() for v in VARIANTS} for label in ("legal14", "invalid8_diagnostic")}
    paired = {label: defaultdict(Counter) for label in totals}
    strata = defaultdict(Counter)
    scenes, examples, sample_counts = {}, [], Counter()
    for scene in range(1, 23):
        legal = scene in LEGAL_TEST
        group = "legal14" if legal else "invalid8_diagnostic"
        audits = {v: DeliveredIdentityAudit() for v in VARIANTS}
        scene_paired = defaultdict(Counter)
        global_records = []
        gt_visible = Counter()
        timeline = list(dataset.synchronized_pairs(scene))
        with ExitStack() as stack:
            streams = {v: stack.enter_context((a.output_root / f"predictions/{v}/scene_{scene:02d}/tracks.jsonl").open()) for v in VARIANTS}
            for step, pair in enumerate(timeline):
                rows = {v: json.loads(next(streams[v])) for v in VARIANTS}
                if any(row["timestamp"] != pair.frame_id for row in rows.values()):
                    raise ValueError("CAUSAL_TIMESTAMP_JOIN_DIFFERENCE")
                original = output_candidates(rows["B0"])
                keys = sorted(original)
                obs = [SimpleNamespace(key=k, camera_id=k[0], detection_id=k[1],
                         bbox_xywh=np.asarray(original[k]["bbox_xywh"], float), score=original[k]["score"]) for k in keys]
                gt = dataset.ground_truth_pair(pair)
                labels, match = match_evaluation_identities(obs, gt)
                for camera, annotations in gt.items():
                    for label, count in Counter(identity_key(x) for x in annotations).items():
                        if label is not None and count == 1:
                            gt_visible[int(camera), label] += 1
                events, assigned = {}, {}
                before = {v: audits[v].counts.copy() for v in VARIANTS}
                for v in VARIANTS:
                    events[v], assigned[v] = audits[v].observe(rows[v], obs, labels, step, len(timeline), gt_visible)
                    audits[v].counts.update({"GT_join/" + k: n for k, n in match.items()})
                # Use the identical original geometry inputs, including the
                # B3 context's registered alignment check, even for B4 strata.
                geometry = {(int(x["camera_id"]), int(x["detection_id"])): x for x in rows["B3"]["decisions"]}
                context = rows["B3"]["geometry_context"]
                config = scope["variant_configs"]["B3"]
                aligned = context["alignment_success"] and context["alignment_confidence"] >= config["alignment_min_inlier_ratio"]
                reliable = {k: bool(aligned and geometry[k]["cross_view_fusion_eligible"] and geometry[k]["geometry_reliability"] >= config["reliable_geometry_min_quality"]) for k in keys}
                if context["reprojection_status"] == "WORLD_FROM_OMEGA_NOT_RECORDED":
                    for audit in audits.values():
                        audit.counts["shared_input_missing_world_camera_transform_times"] += 1
                for left, right in CONTRASTS:
                    contrast = right + "_minus_" + left
                    changed = compare_events(events[left], events[right], scene_paired[contrast + "/detection_temporal_events"])
                    for change in changed:
                        sample = contrast, change["kind"], change["correction"], legal
                        if sample_counts[sample] < 5:
                            examples.append({"scene": scene, "timestamp": pair.frame_id, "original_GT_legal": legal,
                                             "contrast": contrast, **change})
                            sample_counts[sample] += 1
                indices = {v: [i for i, o in enumerate(obs) if o.camera_id == v and labels[i] is not None] for v in (1, 2)}
                i1, i2 = indices[1], indices[2]
                if i1 and i2:
                    same = np.asarray([labels[i] for i in i1], object)[:, None] == np.asarray([labels[i] for i in i2], object)[None, :]
                    valid, shared, good = {}, {}, {}
                    for v in VARIANTS:
                        first = np.asarray([assigned[v][keys[i]] if assigned[v][keys[i]] is not None else -1 for i in i1])
                        second = np.asarray([assigned[v][keys[i]] if assigned[v][keys[i]] is not None else -2 for i in i2])
                        valid[v] = (first[:, None] >= 0) & (second[None, :] >= 0)
                        shared[v] = first[:, None] == second[None, :]
                        good[v] = valid[v] & (shared[v] == same)
                        counts = audits[v].counts
                        counts.update(same_person_cross_view_pairs=int(same.sum()), different_person_cross_view_pairs=int((~same).sum()),
                                      correct_same_person_shared_ID=int((good[v] & same).sum()),
                                      same_person_ID_split=int((valid[v] & ~shared[v] & same).sum()),
                                      false_different_person_shared_ID=int((valid[v] & shared[v] & ~same).sum()),
                                      correct_different_person_separate=int((good[v] & ~same).sum()),
                                      missing_output_cross_view_pair_opportunities=int((~valid[v]).sum()))
                    reliable_pairs = np.asarray([reliable[keys[i]] for i in i1])[:, None] & np.asarray([reliable[keys[i]] for i in i2])[None, :]
                    for left, right in CONTRASTS:
                        contrast = right + "_minus_" + left
                        for kind, mask in (("same_person_shared_ID", same), ("different_person_separate_ID", ~same)):
                            matrix_contrast(scene_paired[contrast + "/" + kind], good[left], good[right], mask)
                            if legal:
                                for name, selection in (("geometry_reliable", reliable_pairs), ("geometry_unreliable", ~reliable_pairs)):
                                    matrix_contrast(strata[contrast + "/" + name + "/" + kind], good[left], good[right], mask & selection)
                                matrix_contrast(strata[contrast + "/quarter" + str(min(3, 4 * step // len(timeline))) + "/" + kind], good[left], good[right], mask)
                            for correction, changes in ((True, ~good[left] & good[right] & mask), (False, good[left] & ~good[right] & mask)):
                                sample = contrast, kind, correction, legal
                                for x, y in np.argwhere(changes)[:max(0, 5 - sample_counts[sample])]:
                                    actors = [i1[int(x)], i2[int(y)]]
                                    examples.append({"scene": scene, "timestamp": pair.frame_id, "original_GT_legal": legal,
                                        "contrast": contrast, "kind": kind, "correction": correction,
                                        "actors": [{"camera": keys[i][0], "detection_id": keys[i][1], "GT_posthoc": labels[i],
                                                    "before_ID": assigned[left][keys[i]], "after_ID": assigned[right][keys[i]],
                                                    "bbox_xywh": original[keys[i]]["bbox_xywh"], "shared_geometry_reliable": reliable[keys[i]]} for i in actors]})
                                    sample_counts[sample] += 1
                for i, label in enumerate(labels):
                    if label is not None:
                        global_records.append((label, tuple(assigned[v][keys[i]] for v in VARIANTS)))
                quarter = min(3, 4 * step // max(1, len(timeline)))
                for v in VARIANTS:
                    for key, value in audits[v].counts.items():
                        delta = value - before[v].get(key, 0)
                        if delta < 0:
                            raise ValueError("NONMONOTONE_V5_AUDIT_COUNT")
                        audits[v].quarters[quarter][key] += delta
            if any(stream.readline() for stream in streams.values()):
                raise ValueError("EXTRA_FROZEN_PREDICTION_ROWS")
        graphs, mappings = {}, {}
        for v in VARIANTS:
            graphs[v], mappings[v] = audits[v].graph()
            totals[group][v].update(audits[v].counts)
        for left, right in CONTRASTS:
            li, ri = VARIANTS.index(left), VARIANTS.index(right)
            a0 = global_retention_flags(global_records, li, mappings[left])
            b0 = global_retention_flags(global_records, ri, mappings[right])
            matrix_contrast(scene_paired[right + "_minus_" + left + "/global_one_to_one_retention"], a0, b0, np.ones(a0.shape, bool))
        for kind, counter in scene_paired.items():
            paired[group][kind].update(counter)
        scenes[str(scene)] = {"original_GT_legal": legal, "timestamps": len(timeline),
                             "counts": {v: dict(audits[v].counts) for v in VARIANTS}, "identity_graph": graphs,
                             "paired": {k: dict(v) for k, v in scene_paired.items()},
                             "relative_video_quarters": {v: [dict(c) for c in audits[v].quarters] for v in VARIANTS}}
        atomic_json(a.output.with_suffix(".progress.json"), {"status": "RUNNING_POSTHOC", "complete_scenes": scene, "total_scenes": 22})
        print(json.dumps({"posthoc_scene_complete": scene, "original_GT_legal": legal}), flush=True)
    if sha256(annotation) != ANNOTATIONS or sha256(__file__) != analysis_sha or v5_source_fingerprint() != source:
        raise ValueError("POSTHOC_SOURCE_OR_GT_CHANGED")
    report = {"status": "COMPLETE_FROZEN_ALL5_FULL22_IDENTITY_ATTRIBUTION", "approval": approval,
        "proofs": proofs, "v5_source_sha256": source, "analysis_sha256": analysis_sha,
        "original_GT_sha256": ANNOTATIONS, "official_metric": False, "TEST_parameter_selection": False,
        "new_inference_or_training_executed": False, "GT_role": "POSTHOC_ONLY_AFTER_ALL110_OUTPUTS_FROZEN",
        "legal14_scenes": LEGAL_TEST, "totals": {s: {v: dict(c) for v, c in values.items()} for s, values in totals.items()},
        "paired": {s: {k: dict(c) for k, c in values.items()} for s, values in paired.items()},
        "legal14_shared_input_strata": {k: dict(c) for k, c in strata.items()}, "scenes": scenes, "examples": examples,
        "definitions": {"clean_join": "unchanged greedy one-to-one original annotation IoU .5 match; not native official MC-MOT",
            "False_Birth": "new delivered ID despite exactly one pure active matching historical ID; an outcome proxy, not causal proof of Birth calibration",
            "NULL": "a unique pure active historical ID remains serialized missed>0 despite the same current clean person",
            "gap": "gap between retained clean observations; GT absence may be occlusion, out-of-view or missing annotation",
            "LOST_recovery": "current clean observation reuses a prior serialized LOST ID; correctness requires pure cumulative prior identity",
            "mixed_ID": "cumulative clean GT associations; this is an identity contamination audit, not a direct embedding measurement",
            "identity_write_coincidence": "a wrong-reused ID has a current identity-write timestamp; this does not identify the camera/observation that originated that write",
            "geometry_strata": "same B3 frozen current input support eligibility/quality>=.6 and alignment confidence>=.25 for every method, including B4",
            "global_graph": "posthoc scene-wise Hungarian one-to-one clean overlap; dropped common candidates count as failed retention, never official CVIDF1",
            "dropped_outputs": "retained in raw candidate coverage and cross-view/global event denominators; historical event pairs use common delivered opportunities"},
        "created_UTC": datetime.now(timezone.utc).isoformat(), "wall_s": time.perf_counter() - started}
    atomic_json(a.output, report)


if __name__ == "__main__":
    main()
