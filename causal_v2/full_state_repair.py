"""Target-centric complete internal-state repair for Causal Audit v2.

The helper deliberately separates public tracker output from the copied
history consumed by GMT.  It is called only on a restored event branch.  A
repair either applies atomically or returns an explicit skip reason.
"""
from __future__ import annotations

import copy
from collections import Counter

import torch
from detectron2.structures import Instances

from gtr.modeling.meta_arch.gtr_rcnn import old_reids, poss_ids


def _clone(value):
    return copy.deepcopy(value)


def _row_id(inst, det: int) -> int:
    return int(inst.track_ids[int(det)].item())


def _feature_index(history, row):
    """Return one exact feature match in an Instances history, or None."""
    if history is None or not history.has("reid_features"):
        return None
    target = row.reid_features.detach()
    for j in range(len(history)):
        if torch.equal(history.reid_features[j].detach(), target[0].detach()):
            return int(j)
    return None


def _proxy_for(engine, pid: int):
    """Rebuild the released memory-bank proxy from repaired history."""
    hist = engine.id_reid_dict.get(int(pid))
    if hist is None or len(hist) < int(engine.model.bank_size):
        return None
    proxy = _clone(hist[:1])
    proxy.track_ids = torch.full_like(proxy.track_ids, int(pid))
    mean = hist[-int(engine.model.bank_size):].reid_features.mean(dim=0, keepdim=True)
    proxy.reid_features = mean.to(device=proxy.reid_features.device,
                                  dtype=proxy.reid_features.dtype)
    return proxy


def _old_reid_ids():
    ids = set()
    for container in old_reids.old_reids:
        if not container.has("track_ids"):
            continue
        ids.update(int(x.item()) for x in container.track_ids)
    return ids


def _rebuild_relevant_memory_bank(engine, relevant):
    """Replace only old_reids rows for relevant IDs with repaired proxies."""
    if not old_reids.old_reids:
        return {"status": "not_present", "ids": []}
    rebuilt = []
    touched = set()
    for container in old_reids.old_reids:
        if not container.has("track_ids"):
            rebuilt.append(container)
            continue
        rows = []
        for i in range(len(container)):
            pid = int(container.track_ids[i].item())
            if pid not in relevant:
                rows.append(container[i:i + 1])
                continue
            proxy = _proxy_for(engine, pid)
            if proxy is None:
                return {
                    "status": "C1c-bankoff",
                    "ids": sorted(touched | {pid}),
                    "reason": f"cannot rebuild old_reids proxy for id {pid}",
                }
            rows.append(proxy)
            touched.add(pid)
        if rows:
            rebuilt.append(Instances.cat(rows))
    old_reids.old_reids = rebuilt
    return {"status": "rebuilt", "ids": sorted(touched)}


def _window_rows(engine, event, window=20):
    """Find wrong-ID observations in the preceding W internal frames."""
    event_frame = int(event["frame"]) - 1
    lo = max(0, event_frame - int(window))
    moved = []
    for frame in range(lo, event_frame):
        for view in range(engine.view_num):
            seq = frame * engine.view_num + view
            if seq >= len(engine.instances):
                continue
            inst = engine.instances[seq]
            if not inst.has("track_ids"):
                continue
            gt_ids = engine._gt_ids(frame, view)
            for det, gid in enumerate(gt_ids):
                if gid != int(event["gt_id"]):
                    continue
                pid = engine._effective_public_or_override_id(seq, det)
                if pid == int(event["p_baseline"]):
                    moved.append({
                        "sequence": int(seq),
                        "frame": int(frame),
                        "view": int(view),
                        "detection_index": int(det),
                        "gt_id": int(gid),
                        "row": _clone(inst[det:det + 1]),
                    })
    return moved


def _assert_unique_instances(engine, instances):
    failures = []
    for seq, inst in enumerate(instances):
        if not inst.has("track_ids"):
            continue
        ids = [int(x.item()) for x in inst.track_ids]
        if len(ids) != len(set(ids)):
            failures.append({"sequence": int(seq), "ids": ids})
    return failures


def _state_summary(engine, pids):
    old_entries = []
    for container in old_reids.old_reids:
        if not container.has("track_ids"):
            continue
        old_entries.extend(int(x.item()) for x in container.track_ids
                           if int(x.item()) in pids)
    return {
        "id_count_dict": {str(pid): int(engine.id_count_dict.get(pid, 0)) for pid in pids},
        "id_reid_dict_len": {str(pid): int(len(engine.id_reid_dict.get(pid, []))) for pid in pids},
        "poss_ids": sorted(int(x) for x in poss_ids.poss_ids if int(x) in pids),
        "old_reids_ids": old_entries,
    }


def _public_track_ids(engine):
    return [
        [int(x.item()) for x in inst.track_ids] if inst.has("track_ids") else None
        for inst in engine.instances
    ]


def apply_full_state_repair(engine, event, *, window=20):
    """Apply an atomic C1c repair to the restored branch.

    Returns a report with ``executable`` and explicit skip reasons.  No public
    ``engine.instances`` track ID is changed; future history copies use the
    returned ``engine.history_id_overrides`` mapping.
    """
    wrong = int(event["p_baseline"])
    correct = int(event["p_correct"])
    gid = int(event["gt_id"])
    public_before = _public_track_ids(engine)
    state_before = _state_summary(engine, {wrong, correct})
    moved = _window_rows(engine, event, window=window)
    overrides = dict(engine.history_id_overrides)

    conflicts = []
    by_seq = {}
    for item in moved:
        seq = item["sequence"]
        det = item["detection_index"]
        if (seq, det) in overrides and int(overrides[(seq, det)]) != correct:
            conflicts.append({"sequence": seq, "detection_index": det,
                              "reason": "existing_override_mismatch"})
        by_seq.setdefault(seq, []).append(det)
        inst = engine.instances[seq]
        for other, value in enumerate(inst.track_ids):
            if other != det and engine._effective_public_or_override_id(seq, other) == correct:
                conflicts.append({"sequence": seq, "detection_index": det,
                                  "occupied_by": other,
                                  "reason": "current_target_occupied"})
        overrides[(seq, det)] = correct
    for seq, dets in by_seq.items():
        if len(dets) != len(set(dets)):
            conflicts.append({"sequence": seq, "reason": "duplicate_detection"})
        proposed = [_row_id(engine.instances[seq], i) for i in range(len(engine.instances[seq]))]
        for det in dets:
            proposed[det] = correct
        if len(proposed) != len(set(proposed)):
            conflicts.append({"sequence": seq, "reason": "post_repair_id_collision"})
    if conflicts:
        return {
            "executable": False,
            "skip_reason": "repair_conflict",
            "conflicts": conflicts,
            "moved": [],
            "assertions": {},
        }

    # Preflight exact feature matches before mutating any state.
    remove_indices = []
    for item in moved:
        hist = engine.id_reid_dict.get(wrong)
        index = _feature_index(hist, item["row"])
        if index is None:
            return {
                "executable": False,
                "skip_reason": "repair_id_reid_mismatch",
                "conflicts": [{"sequence": item["sequence"],
                               "detection_index": item["detection_index"],
                               "reason": "wrong-id feature not found"}],
                "moved": [],
                "assertions": {},
            }
        remove_indices.append(index)

    # Apply history overrides only after all collision and feature checks pass.
    engine.history_id_overrides = overrides
    if wrong in engine.id_reid_dict and remove_indices:
        hist = engine.id_reid_dict[wrong]
        remove = set(remove_indices)
        keep = [i for i in range(len(hist)) if i not in remove]
        if keep:
            engine.id_reid_dict[wrong] = hist[keep]
        else:
            engine.id_reid_dict.pop(wrong, None)
    for item in moved:
        row = _clone(item["row"])
        row.track_ids = torch.full_like(row.track_ids, correct)
        if correct in engine.id_reid_dict:
            engine.id_reid_dict[correct] = Instances.cat([engine.id_reid_dict[correct], row])
        else:
            engine.id_reid_dict[correct] = row

    engine.id_count_dict[wrong] = int(engine.id_count_dict.get(wrong, 0)) - len(moved)
    if engine.id_count_dict[wrong] <= 0:
        engine.id_count_dict.pop(wrong, None)
    engine.id_count_dict[correct] = int(engine.id_count_dict.get(correct, 0)) + len(moved)

    if moved:
        if engine.gt_history[wrong][gid] < len(moved):
            return {
                "executable": False,
                "skip_reason": "repair_gt_history_mismatch",
                "conflicts": [{"reason": "gt_history_underflow"}],
                "moved": [],
                "assertions": {},
            }
        engine.gt_history[wrong][gid] -= len(moved)
        if engine.gt_history[wrong][gid] <= 0:
            del engine.gt_history[wrong][gid]
        engine.gt_history[correct][gid] += len(moved)

    bank_report = _rebuild_relevant_memory_bank(engine, {wrong, correct})
    if bank_report["status"] == "C1c-bankoff":
        return {
            "executable": False,
            "skip_reason": "C1c-bankoff",
            "conflicts": [bank_report],
            "moved": [],
            "assertions": {},
        }

    # Recalculate active candidate eligibility for the repaired IDs while
    # preserving unrelated GMT module-global state.
    active_old = _old_reid_ids()
    for pid in (wrong, correct):
        poss_ids.poss_ids.discard(pid)
        if int(engine.id_count_dict.get(pid, 0)) > int(engine.model.bank_size) and pid not in active_old:
            poss_ids.poss_ids.add(pid)

    # Assertions are emitted as data so the sanity report can show the exact
    # before/after state instead of relying on a silent boolean.
    internal_counts = engine.internal_history_counts((wrong, correct), event, window=window)
    count_assertions = {}
    for pid in (wrong, correct):
        count_assertions[str(pid)] = {
            "id_count_dict": int(engine.id_count_dict.get(pid, 0)),
            "id_reid_dict_len": int(len(engine.id_reid_dict.get(pid, []))),
            "equal": int(engine.id_count_dict.get(pid, 0)) == int(len(engine.id_reid_dict.get(pid, []))),
        }
    unique_failures = _assert_unique_instances(engine, engine.instances)
    assertions = {
        "state_before": state_before,
        "state_after": _state_summary(engine, {wrong, correct}),
        "internal_history_counts": internal_counts,
        "id_count_vs_id_reid": count_assertions,
        "frame_view_unique": not bool(unique_failures),
        "frame_view_failures": unique_failures,
        "old_reids": bank_report,
        "public_history_unchanged": public_before == _public_track_ids(engine),
    }
    if not all(x["equal"] for x in count_assertions.values()) or unique_failures:
        return {
            "executable": False,
            "skip_reason": "repair_consistency_assertion",
            "conflicts": [],
            "moved": [],
            "assertions": assertions,
        }
    return {
        "executable": True,
        "skip_reason": None,
        "conflicts": [],
        "moved": [{k: v for k, v in x.items() if k != "row"} for x in moved],
        "assertions": assertions,
    }
