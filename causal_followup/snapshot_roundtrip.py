#!/usr/bin/env python
"""Run the 20-frame serializable GMT tracker-state round-trip gate."""
from __future__ import annotations

import collections
import json
import os
import random
from pathlib import Path

# PyTorch requires this to be present before its CUDA context is initialized
# for deterministic cuBLAS GEMMs used by the association transformer.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch

from replay_engine import ReplayEngine, load_model
from gtr.audit.state_snapshot import save_snapshot, state_digest


def advance(engine, frame):
    engine._append_frame(frame)
    for view in range(engine.view_num):
        engine._run_tracker_once(frame, view)
    engine._update_gt_history(frame)


def ids(engine, start, end):
    out = []
    for frame in range(start, end):
        for view in range(engine.view_num):
            out.append(engine.instances[frame * engine.view_num + view].track_ids.detach().cpu().tolist())
    return out


def main():
    root = Path(__file__).resolve().parents[1]
    cache = root / "causal_followup/association_replay"
    gt_path = Path("/data3/liuyeqiang/GMT_challenge_audit/audit/cache/sanity_subset_test.json")
    gt = json.loads(gt_path.read_text())
    weight = root / "audit/links/stage2_model_20000.pth"
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "4")
    model = load_model(root, weight)
    torch.set_grad_enabled(False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    checks = []
    snap_root = root / "causal_followup/snapshots"
    for video in gt["videos"]:
        images = [x for x in gt["images"] if int(x["video_id"]) == int(video["id"])]
        images = sorted(images, key=lambda x: (int(x.get("view_id", 0)), int(x.get("frame_id", 0))))
        scene = str(video["file_name"])
        payload = {"scene": scene, "images": images, "view_num": int(video["view_num"])}
        engine = ReplayEngine(model, payload, cache, gt, [], "C1b")
        engine._initialise()
        for f in range(1, min(10, engine.view_frames)):
            advance(engine, f)
        if engine.view_frames < 20:
            checks.append({"scene": scene, "status": "SKIP", "reason": "fewer_than_20_frames"})
            continue
        snap = engine._snapshot(10, 0)
        original_digest = state_digest(snap)
        path = snap_root / f"{scene}.pt"
        save_snapshot(path, snap)
        # Verify the Python/NumPy/Torch random streams are part of the saved state.
        py_a = random.random(); np_a = float(np.random.rand()); torch_a = torch.rand(3)
        loaded = torch.load(path, map_location=engine.device)
        loaded_digest = state_digest(loaded)
        engine._restore(loaded)
        py_b = random.random(); np_b = float(np.random.rand()); torch_b = torch.rand(3)
        rng_equal = (py_a == py_b and np_a == np_b and torch.equal(torch_a, torch_b))
        first = []
        for f in range(10, 20):
            advance(engine, f)
        first = ids(engine, 10, 20)
        # Restore the same serialized snapshot and replay the same twenty-frame
        # continuation. Exact list equality is the gate, not a tolerance.
        engine._restore(torch.load(path, map_location=engine.device))
        for f in range(10, 20):
            advance(engine, f)
        second = ids(engine, 10, 20)
        first_diff = None
        for di, (a, b) in enumerate(zip(first, second)):
            if a != b:
                first_diff = {"continuation_index": di, "first": a, "second": b}
                break
        checks.append({
            "scene": scene,
            "status": "PASS" if first == second and rng_equal and original_digest == loaded_digest else "FAIL",
            "snapshot": str(path),
            "snapshot_sha256": __import__("hashlib").sha256(path.read_bytes()).hexdigest(),
            "original_digest": original_digest,
            "loaded_digest": loaded_digest,
            "trace_exact": first == second,
            "first_diff": first_diff,
            "rng_exact": rng_equal,
            "frames_replayed": 10,
        })
    passed = all(x["status"] in ("PASS", "SKIP") for x in checks) and any(x["status"] == "PASS" for x in checks)
    payload = {"status": "PASS" if passed else "STOP", "checks": checks,
               "definition": "serializable pre-frame-10 tracker state; exact 10-frame continuation and RNG round-trip"}
    out = root / "causal_followup/SNAPSHOT_ROUNDTRIP.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
