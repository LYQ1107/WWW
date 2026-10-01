#!/usr/bin/env python
"""Event-isolated counterfactual replay over a frozen GMT observation cache.

The engine runs the released association methods on cached observations.  It
performs one baseline pass per scene and, at each selected event, branches
SHAM and treatment from the same serialized pre-event state.  Branches run
only through the requested +20-frame horizon and are restored before the
baseline pass continues.
"""
from __future__ import annotations

import argparse
import collections
import csv
import copy
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.utils.logger import setup_logger

from centernet.config import add_centernet_config
from gtr.config import add_gtr_config
from gtr.audit.association_replay import AssociationReplayReader
from gtr.audit.state_snapshot import (
    restore_tracker_state,
    snapshot_tracker_state,
    state_digest,
)
from gtr.modeling.meta_arch.gtr_rcnn import old_ids, old_reids, poss_ids


def iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ab = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / (aa + ab - inter) if aa + ab - inter > 0 else 0.0


def xywh_to_xyxy(box):
    x, y, w, h = [float(v) for v in box]
    return (x, y, x + w, y + h)


class ReplayEngine:
    def __init__(self, model, scene_payload, cache_root, gt_payload, events, mode):
        self.model = model
        self.device = next(model.parameters()).device
        self.scene = scene_payload["scene"]
        # Dataset/cache order is view-blocked: all frames of view 1 followed
        # by all frames of view 2.  The GMT indexing uses that order.
        self.images = sorted(
            scene_payload["images"],
            key=lambda x: (int(x.get("view_id", 0)), int(x["frame_id"])),
        )
        self.view_num = int(scene_payload["view_num"])
        self.view_frames = len(self.images) // self.view_num
        self.cache = AssociationReplayReader(cache_root, self.scene)
        self.records = self.cache.records
        self.gt_images = {int(x["id"]): x for x in gt_payload["images"]}
        self.anns = collections.defaultdict(list)
        for ann in gt_payload["annotations"]:
            self.anns[int(ann["image_id"])].append(ann)
        self.events = {str(x["event_key"]): x for x in events}
        self.mode = mode
        self.instances = []
        self.id_count = 0
        self.id_count_dict = {}
        self.id_reid_dict = {}
        self.quarantine = set()
        self.gt_history = collections.defaultdict(collections.Counter)
        self.branching = False
        self.current_context = None
        self.results = []
        self.snapshot_roundtrip = None
        self.last_frame = self.view_frames - 1
        self._reset_globals()

    def _reset_globals(self):
        poss_ids.poss_ids = set()
        old_ids.old_ids = set()
        old_reids.old_reids = []
        self.model._gmt_replay_quarantine = set()
        self.model._gmt_replay_quarantine_current = set()
        self.model._gmt_replay_action = None
        self.model._gmt_replay_action_skip = None

    def _item(self, frame, view):
        return self.images[frame + view * self.view_frames]

    def _record(self, frame, view):
        return self.records[(int(frame), int(view))]

    def context(self, frame, view):
        item = self._item(frame, view)
        return {
            "scene": self.scene,
            "video_id": int(item.get("video_id", -1)),
            "image_id": int(item.get("image_id", -1)),
            "frame": int(item.get("frame_id", frame + 1)),
            "view": int(item.get("view_id", view + 1)),
            "width": int(item.get("width", 0)),
            "height": int(item.get("height", 0)),
        }

    def _append_frame(self, frame):
        for view in range(self.view_num):
            self.instances.append(self.cache.get(frame, view, self.device))

    def _gt_ids(self, frame, view):
        row = self._record(frame, view)
        image_id = int(row["image_id"])
        anns = self.anns.get(image_id, [])
        boxes = row["pred_boxes"].numpy().copy()
        ih, iw = [float(x) for x in row["image_size"]]
        ow, oh = float(row["width"]), float(row["height"])
        boxes[:, [0, 2]] *= ow / max(iw, 1.0)
        boxes[:, [1, 3]] *= oh / max(ih, 1.0)
        gt_boxes = [xywh_to_xyxy(x["bbox"]) for x in anns]
        if not gt_boxes or len(boxes) == 0:
            return [None] * len(boxes)
        mat = np.asarray([[iou(p, g) for g in gt_boxes] for p in boxes])
        # The audit cache uses the same one-to-one IoU>=.5 matching as the
        # original C0 hook.
        from scipy.optimize import linear_sum_assignment
        rows, cols = linear_sum_assignment(-mat)
        out = [None] * len(boxes)
        for r, c in zip(rows, cols):
            if mat[r, c] >= 0.5:
                out[r] = int(anns[c].get("instance_id", -1))
        return out

    def _update_gt_history(self, frame):
        for view in range(self.view_num):
            seq = frame * self.view_num + view
            gt_ids = self._gt_ids(frame, view)
            inst = self.instances[seq]
            for i, gid in enumerate(gt_ids):
                if gid is None or (seq, i) in self.quarantine:
                    continue
                pid = int(inst.track_ids[i].item())
                if pid >= 0:
                    self.gt_history[pid][int(gid)] += 1

    def _canonical(self, pid):
        counts = self.gt_history.get(int(pid))
        if not counts:
            return None
        return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]

    def _frame_error(self, frame):
        wrong = total = 0
        for view in range(self.view_num):
            seq = frame * self.view_num + view
            inst = self.instances[seq]
            for i, gid in enumerate(self._gt_ids(frame, view)):
                if gid is None:
                    continue
                pid = int(inst.track_ids[i].item())
                can = self._canonical(pid)
                if can is None:
                    continue
                total += 1
                wrong += int(int(can) != int(gid))
        return wrong, total

    def _snapshot(self, frame, view, window_start=None, window_end=None):
        return snapshot_tracker_state(
            frame_index=frame,
            view_index=view,
            instances=self.instances,
            id_count=self.id_count,
            id_count_dict=self.id_count_dict,
            id_reid_dict=self.id_reid_dict,
            window_start=window_start,
            window_end=window_end,
            extra={"quarantine": set(self.quarantine), "gt_history": copy.deepcopy(self.gt_history)},
        )

    def _restore(self, snap):
        state = restore_tracker_state(snap)
        self.instances = state["instances"]
        self.id_count = state["id_count"]
        self.id_count_dict = state["id_count_dict"]
        self.id_reid_dict = state["id_reid_dict"]
        extra = state.get("extra", {})
        self.quarantine = set(extra.get("quarantine", set()))
        self.gt_history = extra.get("gt_history", collections.defaultdict(collections.Counter))
        if not isinstance(self.gt_history, collections.defaultdict):
            self.gt_history = collections.defaultdict(collections.Counter, self.gt_history)
        self.model._gmt_replay_quarantine = self.quarantine
        self.model._gmt_replay_quarantine_current = set()
        self.model._gmt_replay_action = None
        self.model._gmt_replay_action_skip = None

    def _action(self, event, kind):
        key = str(event["event_key"])
        target_key = "p_wrong" if kind == "C2b-INJECTION" else "p_correct"
        target = int(event[target_key])
        det = int(event["detection_index"])

        def callback(*, context, instances, track_ids, **kwargs):
            current_key = f"{context['scene']}|{context['frame']}|{context['view']}|{det}"
            if current_key != key:
                return None
            out = track_ids.clone()
            if det >= len(out):
                raise RuntimeError(f"event detection index out of range: {key}")
            if any(i != det and int(x.item()) == target for i, x in enumerate(out)):
                self.model._gmt_replay_action_skip = "current_target_occupied"
                return None
            out[det] = target
            return {"track_ids": out,
                    "quarantine_indices": [det] if kind == "C1b-B" else []}
        return callback

    def _repair_identity_state(self, event):
        """Oracle C1b-C repair of state history, without rewriting outputs."""
        wrong = int(event["p_baseline"])
        correct = int(event["p_correct"])
        gid = int(event["gt_id"])
        # Locate wrong-ID observations in the preceding 20 frame window.
        moved = []
        frame = int(event["frame"]) - 1
        lo = max(0, frame - 20)
        for f in range(lo, frame + 1):
            for v in range(self.view_num):
                seq = f * self.view_num + v
                if seq >= len(self.instances):
                    continue
                inst = self.instances[seq]
                if not inst.has("track_ids"):
                    continue
                gts = self._gt_ids(f, v)
                for i, g in enumerate(gts):
                    if g == gid and int(inst.track_ids[i].item()) == wrong:
                        moved.append((seq, i, inst[i]))
                        self.gt_history[wrong][gid] -= 1
                        if self.gt_history[wrong][gid] <= 0:
                            del self.gt_history[wrong][gid]
        # Do not silently merge two detections into one p_correct identity at
        # the same historical (frame, view). Such an oracle repair is marked
        # non-executable for this event.
        conflicts = []
        for seq, det, _ in moved:
            inst = self.instances[seq]
            for other in range(len(inst)):
                if other != det and int(inst.track_ids[other].item()) == correct:
                    conflicts.append({
                        "sequence": int(seq),
                        "detection_index": int(det),
                        "occupied_by": int(other),
                    })
        if conflicts:
            return {"executable": False, "skip_reason": "repair_conflict", "conflicts": conflicts}

        if wrong in self.id_reid_dict and moved:
            hist = self.id_reid_dict[wrong]
            # Match moved observations by exact fused feature and remove one
            # matching history row for each observation.
            remove = []
            for _, _, m in moved:
                mf = m.reid_features.detach()
                for j in range(len(hist)):
                    if j in remove:
                        continue
                    if torch.equal(hist[j].reid_features.detach(), mf):
                        remove.append(j); break
            keep = [j for j in range(len(hist)) if j not in set(remove)]
            if keep:
                self.id_reid_dict[wrong] = hist[keep]
            else:
                self.id_reid_dict.pop(wrong, None)
            self.id_count_dict[wrong] = max(0, int(self.id_count_dict.get(wrong, 0)) - len(remove))
            if self.id_count_dict.get(wrong, 0) <= 0:
                self.id_count_dict.pop(wrong, None)
        if moved:
            for _, _, m in moved:
                m = copy.deepcopy(m)
                m.track_ids = torch.full_like(m.track_ids, correct)
                if correct in self.id_reid_dict:
                    from detectron2.structures import Instances
                    self.id_reid_dict[correct] = Instances.cat([self.id_reid_dict[correct], m])
                else:
                    self.id_reid_dict[correct] = m
                self.id_count_dict[correct] = int(self.id_count_dict.get(correct, 0)) + 1
                self.gt_history[correct][gid] += 1
        if wrong not in self.id_count_dict:
            poss_ids.poss_ids.discard(wrong)
        if correct in self.id_count_dict and self.id_count_dict[correct] > self.model.bank_size:
            poss_ids.poss_ids.add(correct)
        return {"executable": True, "skip_reason": None, "conflicts": []}

    def _run_tracker_once(self, frame, view, action=None):
        win_st = max(0, frame + 1 - self.model.test_len) * self.view_num
        win_ed = self.view_num * frame
        history = self.model._gmt_filter_replay_history(
            self.instances[win_st:win_ed], win_st)
        old = self.model._gmt_filter_replay_history(self.instances[:win_st], 0)
        kv = history + [self.instances[win_ed + view]]
        asso_output, pred_boxes, n_t, np_, query_inds = self.model.get_asso(
            kv, k=len(kv) - 1)
        self.current_context = self.context(frame, view)
        self.model._gmt_audit_context = self.current_context
        self.model._gmt_replay_action = action
        self.model._gmt_replay_quarantine = self.quarantine
        self.model._gmt_replay_quarantine_current = set()
        self.model._gmt_replay_action_skip = None
        kv, self.id_count, self.id_count_dict = self.model.run_global_tracker_plus(
            self.view_num, kv, [asso_output[0][:, :np_]], pred_boxes[:np_, :],
            len(kv) - 1, self.id_count, self.id_count_dict,
            self.id_reid_dict, old, view)
        self.instances[win_ed + view] = kv[-1]
        seq = frame * self.view_num + view
        for det in getattr(self.model, "_gmt_replay_quarantine_current", set()):
            self.quarantine.add((seq, int(det)))
        skip_reason = getattr(self.model, "_gmt_replay_action_skip", None)
        self.model._gmt_replay_action = None
        self.model._gmt_replay_quarantine = self.quarantine
        self.model._gmt_replay_action_skip = None
        return skip_reason

    def _initialise(self):
        self._append_frame(0)
        first_counts = [len(self.instances[i]) for i in range(self.view_num)]
        max_index = int(np.argsort(first_counts)[-1])
        self.instances[max_index].track_ids = torch.arange(
            1, len(self.instances[max_index]) + 1,
            device=self.device)
        self.id_count = len(self.instances[max_index])
        for i in range(1, self.id_count + 1):
            self.id_count_dict[i] = 1
            self.id_reid_dict[i] = self.instances[max_index][i - 1]
        other = sorted([i for i in range(self.view_num) if i != max_index])
        kv = [self.instances[max_index]]
        for v in other:
            kv.append(self.instances[v])
            asso_output, pred_boxes, n_t, np_, query_inds = self.model.get_asso(kv, k=len(kv)-1)
            kv, self.id_count, self.id_count_dict, self.id_reid_dict = self.model.run_first_tracker_plus(
                kv, [asso_output[0][:, :np_]], pred_boxes[:np_, :], len(kv)-1,
                self.id_count, self.id_count_dict, self.id_reid_dict)
            self.instances[v] = kv[-1]
        self._update_gt_history(0)

    def _run_future(self, event_frame, event_view, horizon=20):
        out = {}
        # Complete the remainder of the event frame first, without scoring
        # it as a future horizon.
        for v in range(event_view + 1, self.view_num):
            self._run_tracker_once(event_frame, v)
        self._update_gt_history(event_frame)
        for h in range(1, horizon + 1):
            f = event_frame + h
            if f > self.last_frame:
                out[h] = {"wrong": 0, "total": 0, "skip_reason": "scene_end"}
                continue
            self._append_frame(f)
            for v in range(self.view_num):
                self._run_tracker_once(f, v)
            wrong, total = self._frame_error(f)
            out[h] = {"wrong": int(wrong), "total": int(total),
                      "error_rate": (float(wrong) / total if total else None)}
            self._update_gt_history(f)
        return out

    def _run_branch(self, snap, frame, view, event, kind):
        self._restore(snap)
        repair = {"executable": True, "skip_reason": None}
        if kind == "C1b-C":
            repair = self._repair_identity_state(event)
            if not repair["executable"]:
                return {
                    h: {"wrong": 0, "total": 0,
                        "skip_reason": repair["skip_reason"]}
                    for h in range(1, 21)
                }
        # C1b-C repairs persistent history and then applies the current
        # p_correct assignment before committing it to the repaired state.
        action = None if kind == "SHAM" else self._action(event, kind)
        skip_reason = self._run_tracker_once(frame, view, action)
        if skip_reason:
            return {
                h: {"wrong": 0, "total": 0, "skip_reason": skip_reason}
                for h in range(1, 21)
            }
        return self._run_future(frame, view)

    def _evaluate_event(self, frame, view, event, pre_snapshot=None, baseline_after=None):
        win_st = max(0, frame + 1 - self.model.test_len) * self.view_num
        win_ed = self.view_num * frame
        # Several frozen detections can share one frame/view. Every one must
        # branch from exactly the same uncommitted state, rather than letting
        # the first event change the pre-state of the next event.
        pre = pre_snapshot if pre_snapshot is not None else self._snapshot(frame, view, win_st, win_ed)
        # A full tensor digest is useful for the dedicated round-trip gate;
        # avoid copying every large history to CPU for all 327 follow-up
        # events. Keep one representative digest per scene in this result.
        pre_digest = state_digest(pre) if not self.results else "omitted_for_runtime"
        # Baseline current decision, saved as the continuation point.
        if baseline_after is None:
            self._restore(pre)
            self._run_tracker_once(frame, view)
            baseline_after = self._snapshot(frame, view, win_st, win_ed)
        kinds = ["SHAM", "C1b-A", "C1b-B", "C1b-C"] if self.mode == "C1b" else ["SHAM", "C2b-INJECTION"]
        for kind in kinds:
            values = self._run_branch(pre, frame, view, event, kind)
            for h, stats in values.items():
                self.results.append({
                    "event_key": event["event_key"], "scene": self.scene,
                    "frame": int(frame), "view": int(view),
                    "gt_id": int(event["gt_id"]), "condition": kind,
                    "horizon": int(h), **stats,
                })
        self._restore(baseline_after)
        self.model._gmt_replay_action = None
        self.model._gmt_replay_quarantine = self.quarantine
        self.results.append({"event_key": event["event_key"], "scene": self.scene,
                             "frame": int(frame), "view": int(view),
                             "gt_id": int(event["gt_id"]), "condition": "_META",
                             "horizon": 0, "pre_snapshot_digest": pre_digest,
                             "post_snapshot_digest": (state_digest(baseline_after)
                                                       if pre_digest != "omitted_for_runtime"
                                                       else "omitted_for_runtime")})

    def run(self):
        self._initialise()
        for frame in range(1, self.view_frames):
            self._append_frame(frame)
            for view in range(self.view_num):
                key = f"{self.scene}|{int(self._item(frame, view).get('frame_id', frame + 1))}|{int(self._item(frame, view).get('view_id', view + 1))}|"
                # Event manifests use frame/view values parsed from filenames;
                # select by the exact cache context and detection suffix.
                events = [candidate for candidate in self.events.values()
                          if (candidate["scene"] == self.scene
                              and int(candidate["frame"]) == int(self._item(frame, view).get("frame_id", frame + 1))
                              and int(candidate["view"]) == int(self._item(frame, view).get("view_id", view + 1)))]
                if events and not self.branching:
                    win_st = max(0, frame + 1 - self.model.test_len) * self.view_num
                    win_ed = self.view_num * frame
                    shared_pre = self._snapshot(frame, view, win_st, win_ed)
                    self._restore(shared_pre)
                    self._run_tracker_once(frame, view)
                    shared_after = self._snapshot(frame, view, win_st, win_ed)
                    for event in events:
                        self._evaluate_event(
                            frame, view, event,
                            pre_snapshot=shared_pre,
                            baseline_after=shared_after,
                        )
                else:
                    self._run_tracker_once(frame, view)
            self._update_gt_history(frame)
        return self.results


def load_model(root, weight, device="cuda"):
    cfg = get_cfg(); add_centernet_config(cfg); add_gtr_config(cfg)
    cfg.merge_from_file(str(root / "configs/VISION_test.yaml"))
    cfg.MODEL.WEIGHTS = str(weight)
    cfg.INPUT.VIDEO.TEST_LEN = 40
    cfg.MODEL.ASSO_HEAD.WITH_BANK = True
    cfg.MODEL.ASSO_HEAD.BANK_SIZE = 10
    cfg.MODEL.DEVICE = device
    cfg.freeze()
    model = build_model(cfg)
    model.eval()
    model._gmt_replay_audit = True
    model._gmt_replay_reconciliations = []
    DetectionCheckpointer(model).resume_or_load(str(weight), resume=False)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["C1b", "C2b"], required=True)
    ap.add_argument("--events", type=Path, required=True)
    ap.add_argument("--cache-root", type=Path, required=True)
    ap.add_argument("--gt-json", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--weight", type=Path, required=True)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "4")
    os.environ["GMT_ASSOC_REPLAY_LOAD"] = "1"
    os.environ["GMT_ASSOC_REPLAY_DIR"] = str(args.cache_root)
    os.environ.pop("GMT_CAUSAL_AUDIT_MODE", None)
    model = load_model(args.root, args.weight)
    all_events = json.loads(args.events.read_text())["events"]
    gt = json.loads(args.gt_json.read_text())
    by_video = collections.defaultdict(list)
    for item in gt["images"]:
        by_video[int(item["video_id"])].append(item)
    results = []
    for video_id, images in sorted(by_video.items()):
        images = sorted(images, key=lambda x: (int(x.get("view_id", 0)), int(x.get("frame_id", 0))))
        view_num = len({int(x["view_id"]) for x in images})
        scene = re.search(r"([^/\\]+)_View\d+", str(images[0]["file_name"])).group(1)
        events = [x for x in all_events if x["scene"] == scene]
        payload = {"scene": scene, "images": images, "view_num": view_num}
        # The source JSON is view-blocked (all frames of view 1, then view 2).
        engine = ReplayEngine(model, payload, args.cache_root, gt, events, args.mode)
        with torch.no_grad():
            results.extend(engine.run())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in results for k in row})
    with args.output.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(results)
    meta = args.output.with_suffix(".json")
    meta.write_text(json.dumps({"mode": args.mode, "rows": len(results),
                                "event_count": len(all_events),
                                "cache_root": str(args.cache_root),
                                "status": "COMPLETED"}, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "rows": len(results), "events": len(all_events)}, indent=2))


if __name__ == "__main__":
    main()
