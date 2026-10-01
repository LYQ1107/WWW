"""Audit-only causal identity interventions for the released GMT tracker.

The module is deliberately inert unless ``GMT_CAUSAL_AUDIT_MODE`` is set.  It
keeps the GT sidecar, event selection, and outcome logging outside the normal
tracker tensors.  The tracker calls :func:`intervene` after its baseline
Hungarian (and optional memory-bank) decision but before it commits an
observation to ``id_count_dict``/``id_reid_dict``.
"""
from __future__ import annotations

import atexit
import collections
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


def _xywh_to_xyxy(box):
    x, y, w, h = [float(v) for v in box]
    return (x, y, x + w, y + h)


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ab = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = aa + ab - inter
    return inter / union if union > 0 else 0.0


class CausalIdentityAudit:
    """Stateful, deterministic audit hook owned by one inference process."""

    def __init__(self, mode: str, audit_root: str | os.PathLike[str]):
        self.mode = mode
        self.root = Path(audit_root)
        self.gt_path = Path(os.environ["GMT_CAUSAL_GT_JSON"])
        self.event_path = Path(os.environ.get(
            "GMT_CAUSAL_EVENT_MANIFEST",
            str(self.root / "causal" / "manifests" / "correction_events.json"),
        ))
        self.injection_event_path = Path(os.environ.get(
            "GMT_CAUSAL_INJECTION_MANIFEST",
            str(self.root / "causal" / "manifests" / "injection_events.json"),
        ))
        self.run_name = os.environ.get("GMT_CAUSAL_RUN", mode)
        self.run_root = self.root / "causal" / "runs" / self.run_name
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.log_path = self.run_root / "decisions.jsonl"
        self._log = self.log_path.open("a", encoding="utf-8", buffering=1)
        self._closed = False

        payload = json.loads(self.gt_path.read_text())
        self.images = {int(x["id"]): x for x in payload["images"]}
        self.videos = {int(x["id"]): x for x in payload["videos"]}
        self.anns_by_image: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
        for ann in payload["annotations"]:
            self.anns_by_image[int(ann["image_id"])].append(ann)

        # Counts are updated only after the current intervention decision.
        self.history: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
        self.last_correction: dict[int, int] = {}
        self.last_injection: dict[int, int] = {}
        self.current_scene: str | None = None
        # A sham must use the same frozen schedule as its paired intervention.
        # The default is the correction schedule for backwards compatibility;
        # injection sham runs opt into the injection schedule explicitly.
        sham_kind = os.environ.get("GMT_CAUSAL_SHAM_KIND", "correction")
        if mode == "sham" and sham_kind == "injection":
            self.events = self._load_events(self.injection_event_path)
        else:
            self.events = self._load_events(self.event_path) if mode in {"correction", "sham"} else {}
        self.events.update(self._load_events(self.injection_event_path) if mode == "injection" else {})
        atexit.register(self.close)

    @staticmethod
    def _key(scene: str, frame: int, view: int, det: int) -> str:
        return f"{scene}|{int(frame)}|{int(view)}|{int(det)}"

    @staticmethod
    def _load_events(path: Path) -> dict[str, dict[str, Any]]:
        if not path.exists():
            return {}
        payload = json.loads(path.read_text())
        rows = payload.get("events", payload if isinstance(payload, list) else [])
        return {str(row["event_key"]): row for row in rows}

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self._log.flush()
            self._log.close()
        except Exception:
            pass

    def _current_gt(self, image_id: int, boxes_xyxy: np.ndarray):
        image = self.images.get(int(image_id), {})
        anns = self.anns_by_image.get(int(image_id), [])
        gt_boxes = [_xywh_to_xyxy(a["bbox"]) for a in anns]
        if not gt_boxes or len(boxes_xyxy) == 0:
            return [None] * len(boxes_xyxy), [0.0] * len(boxes_xyxy)
        mat = np.asarray([[_iou(p, g) for g in gt_boxes] for p in boxes_xyxy], dtype=float)
        rows, cols = linear_sum_assignment(-mat)
        matched = [None] * len(boxes_xyxy)
        scores = [0.0] * len(boxes_xyxy)
        for r, c in zip(rows, cols):
            if mat[r, c] >= 0.5:
                matched[r] = int(anns[c].get("instance_id", -1))
                scores[r] = float(mat[r, c])
        return matched, scores

    def _ensure_scene(self, scene: str) -> None:
        if self.current_scene == scene:
            return
        self.current_scene = scene
        self.history = collections.defaultdict(collections.Counter)
        self.last_correction = {}
        self.last_injection = {}

    def seed_initial(self, contexts, instances) -> None:
        """Seed canonical IDs from frame zero before any intervention event."""
        if not contexts or not instances:
            return
        self._ensure_scene(str(contexts[0]["scene"]))
        for context, inst in zip(contexts, instances):
            boxes = inst.pred_boxes.tensor.detach().float().cpu().numpy()
            ih, iw = [float(x) for x in inst.image_size]
            ow, oh = float(context.get("width", iw)), float(context.get("height", ih))
            boxes[:, [0, 2]] *= ow / max(iw, 1.0)
            boxes[:, [1, 3]] *= oh / max(ih, 1.0)
            gt_ids, _ = self._current_gt(int(context["image_id"]), boxes)
            ids = inst.track_ids.detach().cpu().tolist()
            for pid, gid in zip(ids, gt_ids):
                if gid is not None and int(pid) >= 0:
                    self.history[int(pid)][int(gid)] += 1

    def _canonical(self, pid: int):
        counts = self.history.get(int(pid))
        if not counts:
            return None, 0, 0.0
        gt, support = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        total = sum(counts.values())
        return int(gt), int(support), float(support / total) if total else 0.0

    def _candidate_rows(self, unique_ids, traj_score, support, det_index: int):
        def array(x):
            if hasattr(x, "detach"):
                x = x.detach().cpu().numpy()
            return np.asarray(x)
        ids = [int(x) for x in array(unique_ids).reshape(-1).tolist()]
        scores = array(traj_score)
        if scores.ndim > 1:
            scores = scores[min(det_index, scores.shape[0] - 1)]
        supports = array(support).reshape(-1)
        rows = []
        for j, pid in enumerate(ids):
            can, sup, purity = self._canonical(pid)
            rows.append({
                "id": pid,
                "score": float(scores[j]) if j < len(scores) else 0.0,
                "support": int(round(float(supports[j]))) if j < len(supports) else sup,
                "past_support": sup,
                "canonical_gt": can,
                "purity": purity,
            })
        return rows

    def intervene(
        self,
        *,
        context: dict[str, Any],
        instances,
        track_ids,
        unique_ids,
        traj_score,
        support,
        id_count_dict=None,
    ):
        """Return the possibly replaced ``track_ids`` and commit audit history."""
        scene = str(context["scene"])
        self._ensure_scene(scene)
        frame = int(context["frame"])
        view = int(context["view"])
        image_id = int(context["image_id"])
        # Convert the model-space boxes back to annotation coordinates.  The
        # normal tracker tensors are never given GT boxes or IDs.
        inst = instances
        boxes = inst.pred_boxes.tensor.detach().float().cpu().numpy()
        ih, iw = [float(x) for x in inst.image_size]
        ow, oh = float(context.get("width", iw)), float(context.get("height", ih))
        scale_x, scale_y = ow / max(iw, 1.0), oh / max(ih, 1.0)
        boxes_orig = boxes.copy()
        boxes_orig[:, [0, 2]] *= scale_x
        boxes_orig[:, [1, 3]] *= scale_y
        gt_ids, gt_ious = self._current_gt(image_id, boxes_orig)
        baseline = [int(x) for x in track_ids.detach().cpu().tolist()]
        assigned = {p for p in baseline if p >= 0}
        out = track_ids.clone()
        scores = inst.scores.detach().float().cpu().tolist() if inst.has("scores") else [1.0] * len(inst)

        for i, p_base in enumerate(baseline):
            candidates = self._candidate_rows(unique_ids, traj_score, support, i)
            g = gt_ids[i] if i < len(gt_ids) else None
            cand = candidates
            correct = None
            wrong = None
            if g is not None:
                corrects = [x for x in cand if x["canonical_gt"] == g and x["id"] != p_base]
                corrects = [x for x in corrects if x["id"] not in (assigned - {p_base})]
                if corrects:
                    correct = sorted(corrects, key=lambda x: (-x["past_support"], -x["score"], x["id"]))[0]
                wrongs = [x for x in cand if x["canonical_gt"] not in (None, g)]
                wrongs = [x for x in wrongs if x["id"] not in (assigned - {p_base})]
                if wrongs:
                    wrong = sorted(wrongs, key=lambda x: (-x["score"], x["id"]))[0]
            base_can, base_sup, base_purity = self._canonical(p_base) if p_base >= 0 else (None, 0, 0.0)
            key = self._key(scene, frame, view, i)
            row = {
                "event_key": key,
                "scene": scene, "video_id": int(context.get("video_id", -1)),
                "frame": frame, "view": view, "image_id": image_id,
                "detection_index": i, "gt_id": g, "gt_iou": float(gt_ious[i]),
                "bbox_xyxy": [float(x) for x in boxes_orig[i]],
                "detector_score": float(scores[i]),
                "baseline_pred_id": p_base,
                "canonical_gt_baseline": base_can,
                "baseline_support": base_sup,
                "baseline_purity": base_purity,
                "candidate_ids": [x["id"] for x in cand],
                "candidate_scores": [x["score"] for x in cand],
                "candidate_support": [x["support"] for x in cand],
                "candidate_canonical_gt": [x["canonical_gt"] for x in cand],
                "candidate_purity": [x["purity"] for x in cand],
                "correct_existing_id": None if correct is None else correct["id"],
                "wrong_existing_id": None if wrong is None else wrong["id"],
                "selected_pred_id": p_base,
                "intervention_target_id": None,
                "intervention": "none",
            }
            event = self.events.get(key)
            if event and self.mode == "correction" and event.get("gt_id") == g:
                target = int(event["p_correct"])
                row["intervention_target_id"] = target
                occupied_elsewhere = any(j != i and int(v) == target for j, v in enumerate(out.tolist()))
                active = id_count_dict is None or target in {int(x) for x in id_count_dict}
                if occupied_elsewhere:
                    row["intervention"] = "correction_conflict"
                elif not active:
                    row["intervention"] = "correction_inactive"
                else:
                    out[i] = target
                    row["selected_pred_id"] = target
                    row["intervention"] = "correction"
                    self.last_correction[int(g)] = frame
            elif event and self.mode == "injection" and event.get("gt_id") == g:
                target = int(event["p_wrong"])
                row["intervention_target_id"] = target
                occupied_elsewhere = any(j != i and int(v) == target for j, v in enumerate(out.tolist()))
                active = id_count_dict is None or target in {int(x) for x in id_count_dict}
                if occupied_elsewhere:
                    row["intervention"] = "injection_conflict"
                elif not active:
                    row["intervention"] = "injection_inactive"
                else:
                    out[i] = target
                    row["selected_pred_id"] = target
                    row["intervention"] = "injection"
                    self.last_injection[int(g)] = frame
            elif event and self.mode == "sham" and event.get("gt_id") == g:
                row["intervention"] = "sham"
            self._log.write(json.dumps(row, separators=(",", ":")) + "\n")

        # Commit only after every current decision has been inspected and any
        # frozen replacement has been applied. This is the intervention point.
        for i, g in enumerate(gt_ids):
            pid = int(out[i].item())
            if g is not None and pid >= 0:
                self.history[pid][int(g)] += 1
        return out


_AUDIT_SINGLETON = None


def get_audit(mode: str, audit_root: str | os.PathLike[str]):
    global _AUDIT_SINGLETON
    if _AUDIT_SINGLETON is None or _AUDIT_SINGLETON.mode != mode:
        if _AUDIT_SINGLETON is not None:
            _AUDIT_SINGLETON.close()
        _AUDIT_SINGLETON = CausalIdentityAudit(mode, audit_root)
    return _AUDIT_SINGLETON
