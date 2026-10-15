"""Read-only normalized adapter for mined *native* GMT hard events.

This adapter is for authorized original-TRAIN screening, not training-time GT
inside the tracker. It cannot by itself certify native commit correctness,
produce a deployable correction, or measure full-video HOTA/FPS.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import torch

FROZEN_STAGE2_SHA256 = "cd72823824d16c86ed27c2dfc8323de610aa9f6c0c0b29249aa3de609deabce8"
FROZEN_CONFIG_SHA256 = "bdaeca71d875e8824c3eaf825967a7ba032514297642a6aabe7ad10a740be94a"
FIT_SCENES = frozenset((4, 5, 6, 7, 8))
DEVELOPMENT_SCENES = frozenset((12,))
ALLOWED_STATES = frozenset((
    "correct_link", "wrong_link", "missed_known_positive", "ambiguous_reference",
    "unknown_query", "unmatched", "unknown", "newborn", "candidate_missing",
))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def checked_json(path: Path) -> dict:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return obj


def source_key(value) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("A detector source key must be [image_id, row_id]")
    if any(isinstance(x, bool) or not isinstance(x, int) for x in value):
        raise ValueError("Source key components must be integers")
    return tuple(value)


def get_observation(observations: dict, key: tuple):
    if key not in observations:
        raise ValueError("Native observation is missing from verified bundle")
    obs = observations[key]
    if not isinstance(obs, dict) or "feature" not in obs:
        raise ValueError("Observation missing native feature")
    return obs


def safe_torch_load(path: Path):
    # Only locally generated, manifest-SHA-verified artifacts are accepted.
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise RuntimeError("PyTorch weights_only=True required; refusing unsafe pickle load") from exc


def load_trackvggt_bundle(folder: Path, *, expected_scope: str | None = None) -> dict:
    """Load a verified, frozen TrackVGGT native-event producer bundle."""
    folder = Path(folder)
    m = checked_json(folder / "manifest.json")
    if m.get("status") != "PASS_READONLY_EVENTS_POST_PARITY_LABELS":
        raise ValueError("Original-GMT OFF/parity collector has not passed")
    if not m.get("prefix40_raw_prediction_exact"):
        raise ValueError("Original GMT exact prefix is not certified")
    if m.get("gt_query_features") is not False or m.get("future_query_features") is not False:
        raise ValueError("Future/GT leakage flags are not explicitly false")
    if m.get("vggt_calls") != 0 or m.get("dinov2_calls") != 0:
        raise ValueError("Not the expected frozen GMT-only source")
    if m.get("scope") not in ("fit", "validation") or (expected_scope and m["scope"] != expected_scope):
        raise ValueError("Unexpected or unauthorized source split")
    if m.get("selection_uses_gt") is not (m["scope"] == "fit"):
        raise ValueError("Development selection depended on GT, or fit selection metadata changed")
    scene = m.get("video_id")
    if scene not in (FIT_SCENES if m["scope"] == "fit" else DEVELOPMENT_SCENES):
        raise ValueError("Wrong scene assigned to train/diagnostic scope")
    if m.get("native_reid_dim") != 1152:
        raise ValueError("Native GMT embedding not the declared 1152-dim feature")
    p = m.get("provenance", {})
    if p.get("checkpoint_sha256") != FROZEN_STAGE2_SHA256:
        raise ValueError("Frozen full GMT Stage2 checkpoint differs")
    if p.get("config_sha256") != FROZEN_CONFIG_SHA256:
        raise ValueError("Frozen GMT config differs")
    if not p.get("annotation_sha256"):
        raise ValueError("GT annotation source hash missing")
    artifacts = m.get("artifacts", {})
    if not isinstance(artifacts, dict) or not isinstance(artifacts.get("events.pt"), str):
        raise ValueError("No events.pt SHA in source manifest")
    file = folder / "events.pt"
    if not file.is_file() or sha256(file) != artifacts["events.pt"]:
        raise ValueError("Native event file checksum mismatch")
    body = safe_torch_load(file)
    if not isinstance(body, dict) or body.get("feature_dim") != 1152:
        raise ValueError("Payload must declare 1152-dim GMT native feature")
    if not isinstance(body.get("events"), list) or not isinstance(body.get("calls"), dict) or not isinstance(body.get("observations"), dict):
        raise ValueError("Malformed native event payload")
    if m.get("selected_events") != len(body["events"]):
        raise ValueError("Selected-event count does not match source manifest")
    return {"manifest": m, "payload": body, "artifact_sha256": artifacts["events.pt"], "folder": str(folder)}


def normalize_event(bundle: dict, index: int) -> dict:
    """Return *separated* deployable inputs and OFFLINE_ONLY supervision."""
    m, body = bundle["manifest"], bundle["payload"]
    e = body["events"][index]
    if not isinstance(e, dict):
        raise ValueError("Invalid event type")
    event_key = e.get("key")
    if not isinstance(event_key, (list, tuple)) or len(event_key) != 3 or int(event_key[0]) != m["video_id"]:
        raise ValueError("Native event key does not bind source scene")
    query_key = source_key(e.get("query_source"))
    if tuple(event_key[1:]) != query_key:
        raise ValueError("Native event and query physical source key disagree")
    ci = e.get("call_index")
    if not isinstance(ci, int) or ci not in body["calls"]:
        raise ValueError("Native association call not retained")
    call = body["calls"][ci]
    if not isinstance(call, dict):
        raise ValueError("Native call has invalid type")
    qrow = e.get("query_row")
    if not isinstance(qrow, int) or qrow < 0 or qrow >= len(call["queries"]):
        raise ValueError("Native call query row out of range")
    if source_key(call["queries"][qrow]) != query_key:
        raise ValueError("Native call query key differs from selected event")
    track_ids = e.get("candidate_tracks")
    columns = e.get("candidate_track_indices")
    original_scores = e.get("native_track_scores")
    if not (isinstance(track_ids, list) and isinstance(columns, list) and isinstance(original_scores, list)):
        raise ValueError("Candidate IDs/indices/scores must be parallel lists")
    if not (len(track_ids) == len(columns) == len(original_scores) and 1 <= len(track_ids) <= 8):
        raise ValueError("Invalid Top8 candidate dimensions")
    if len(set(track_ids)) != len(track_ids):
        raise ValueError("Duplicate candidate Global IDs cannot be inferred as independent")
    raw = call.get("track_scores")
    if not isinstance(raw, torch.Tensor) or raw.ndim != 2 or raw.shape[0] <= qrow or raw.shape[1] != len(call["track_ids"]):
        raise ValueError("Native score tensor does not match original call")
    for tid, col, val in zip(track_ids, columns, original_scores):
        if not isinstance(col, int) or not 0 <= col < len(call["track_ids"]) or call["track_ids"][col] != tid:
            raise ValueError("Selected candidate index or ID not equal to original GMT call")
        value = float(raw[qrow, col])
        if not math.isfinite(value) or not math.isfinite(float(val)) or abs(value-float(val)) > 2e-5:
            raise ValueError("Native selected score differs from original GMT call")
    observations = body["observations"]
    query = get_observation(observations, query_key)
    qt, qc = int(query["frame"]), int(query["camera"])
    if qc not in (1, 2) or qt < 0:
        raise ValueError("Invalid physical camera/frame")
    qfeat = query["feature"]
    if not isinstance(qfeat, torch.Tensor) or tuple(qfeat.shape) != (1152,) or not torch.isfinite(qfeat).all():
        raise ValueError("Bad native query embedding")
    histories = []
    missed_same_time = 0
    for tid in track_ids:
        hist = call.get("histories", {}).get(tid)
        if not isinstance(hist, dict) or not isinstance(hist.get("sources"), list):
            raise ValueError("The original candidate Global ID lacks native past sources")
        legal = []
        for src in hist["sources"]:
            key = source_key(src)
            obs = get_observation(observations, key)
            ht, hc = int(obs["frame"]), int(obs["camera"])
            if ht > qt:
                raise ValueError("Native history includes future scene-frame")
            if ht == qt:
                missed_same_time += 1
                continue
            emb = obs["feature"]
            if not isinstance(emb, torch.Tensor) or tuple(emb.shape) != (1152,) or not torch.isfinite(emb).all():
                raise ValueError("Invalid historical native feature")
            legal.append({"camera":hc, "lag":qt-ht, "feature":emb.float().detach().clone(), "cross_view":hc!=qc})
        histories.append(legal)
    if e.get("native_top1_state") not in ALLOWED_STATES or e.get("actual_primary_state") not in ALLOWED_STATES:
        raise ValueError("Unknown actual primary or ranking state")
    positives=e.get("positive_tracks")
    impure=e.get("ambiguous_tracks")
    if not isinstance(positives,list) or not isinstance(impure,list):
        raise ValueError("Offline GT positive/ambiguous labels missing")
    if any(t not in track_ids for t in positives+impure):
        raise ValueError("Offline label references a noncandidate identity")
    eligible=(len(positives)==1 and not any(t in impure for t in positives)
              and bool(e.get("top8_positive")) and e.get("query_gt_identity") is not None)
    target=track_ids.index(positives[0]) if eligible else None
    category=e["actual_primary_state"]
    actual_wrong=category in ("wrong_link", "missed_known_positive")
    causal_positive=bool(eligible and any(row["cross_view"] for row in histories[target]))
    return {
        "INPUT_ONLY": {"event_key":tuple(event_key),"video":m["video_id"],"frame":qt,"camera":qc,
                       "query_feature":qfeat.float().detach().clone(),
                       "native_candidate_ids":tuple(track_ids),
                       "native_scores":torch.tensor(original_scores,dtype=torch.float32),
                       "past_histories":histories},
        "OFFLINE_ONLY": {"GT_person":e["query_gt_identity"],"certified_target_index":target,
                         "actual_primary_state":category,"native_top1_state":e["native_top1_state"],
                         "correctable_primary_error":bool(actual_wrong and causal_positive),
                         "masked_no_unique_target":not bool(eligible),
                         "has_strict_past_cross_view_positive":causal_positive,
                         "same_time_sources_excluded":missed_same_time,
                         "split":m["scope"]},
    }


def cohort_audit(runtime: Path, cohort_path: Path) -> dict:
    """Read-only audit of locked TrackVGGT fit/diagnostic cohort; no model training."""
    c = checked_json(cohort_path)
    if c.get("status") != "LOCKED_SCENE_DISJOINT_REAL_EVENT_COHORT":
        raise ValueError("Cohort is not frozen")
    if c.get("native_feature_dim") != 1152 or c.get("used_teacher_results_for_selection") is not False:
        raise ValueError("Cohort source type or sample selection changed")
    if set(c.get("fit_scene_ids", [])) != FIT_SCENES or set(c.get("validation_scene_ids", [])) != DEVELOPMENT_SCENES:
        raise ValueError("Unexpected scene split")
    fit, val = c.get("fit"), c.get("validation")
    if not isinstance(fit,list) or not isinstance(val,list) or len(fit)!=1000 or len(val)!=300:
        raise ValueError("Current cohort does not match the frozen 1000/300 protocol")
    if len({tuple(e["key"]) for e in fit}) != len(fit) or {tuple(e["key"]) for e in fit} & {tuple(e["key"]) for e in val}:
        raise ValueError("Physical query appears twice or fit/diagnostic overlap")
    cache, result = {}, Counter()
    persons = {"fit":set(),"validation":set()}
    verified_source_files = set()
    for scope, rows in (("fit",fit),("validation",val)):
        for pointer in rows:
            source=Path(pointer["event_file"])
            folder=source.parent
            if not source.resolve().is_relative_to(runtime.resolve()):
                raise ValueError("Event file is outside the agreed read-only runtime")
            source_sha=c.get("source_artifacts", {}).get(str(source))
            if source_sha is None:
                raise ValueError("Selected source has no locked SHA")
            if str(source) not in verified_source_files:
                if sha256(source)!=source_sha:
                    raise ValueError("Selected source does not match locked cohort SHA")
                verified_source_files.add(str(source))
            key=(str(folder),scope)
            if key not in cache:
                cache[key]=load_trackvggt_bundle(folder,expected_scope=scope)
            obj=normalize_event(cache[key],int(pointer["event_index"]))
            if tuple(pointer["key"]) != obj["INPUT_ONLY"]["event_key"]:
                raise ValueError("Selected pointer mismatch")
            info=obj["OFFLINE_ONLY"]
            result[scope+"_events"]+=1
            result[scope+"_"+info["actual_primary_state"]]+=1
            result[scope+"_"+info["native_top1_state"]+"_top1"]+=1
            result[scope+"_correctable_primary_error"]+=info["correctable_primary_error"]
            result[scope+"_with_certified_positive"]+=info["certified_target_index"] is not None
            result[scope+"_strict_past_cross_view_positive"]+=info["has_strict_past_cross_view_positive"]
            result[scope+"_same_time_sources_excluded"]+=info["same_time_sources_excluded"]
            if info["GT_person"] is not None:
                persons[scope].add((obj["INPUT_ONLY"]["video"],info["GT_person"]))
    return dict(status="PASS_READONLY_NATIVE_EVENT_CONTRACT",counts=dict(result),
                unique_video_person_groups={k:len(v) for k,v in persons.items()},
                source_cohort_sha256=sha256(cohort_path),source_files=len(cache),
                scientific_success=False,full_online_HOTA=None)
