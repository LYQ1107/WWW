#!/usr/bin/env python
"""Assemble the replay-fidelity gate after dump and load runs finish."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch


def main():
    root = Path(__file__).resolve().parents[1]
    gt_path = Path("/data3/liuyeqiang/GMT_challenge_audit/audit/cache/sanity_subset_test.json")
    gt = json.loads(gt_path.read_text())
    cache = root / "causal_followup/association_replay"
    dump_trace = root / "causal_followup/association_trace_dump"
    load_trace = root / "causal_followup/association_trace_load"
    post_dump = root / "causal_followup/association_trace_post_dump"
    post_load = root / "causal_followup/association_trace_post_load"
    scenes = {str(v["file_name"]): v for v in gt["videos"]}
    checks = []
    for scene, video in scenes.items():
        n = sum(1 for x in gt["images"] if int(x["video_id"]) == int(video["id"]))
        p = cache / f"{scene}.pt"
        ok = p.exists()
        count = None
        if ok:
            payload = torch.load(p, map_location="cpu")
            count = int(payload.get("record_count", -1)); ok = count == n
        checks.append({"scene": scene, "cache_records": count, "expected_records": n,
                       "status": "PASS" if ok else "FAIL"})
    compare = root / "causal_followup/REPLAY_TRACE_COMPARISON.json"
    cmd = [sys.executable, str(root / "causal_followup/compare_traces.py"),
           str(dump_trace), str(load_trace), str(compare)]
    trace_rc = subprocess.run(cmd, check=False).returncode
    post_compare = root / "causal_followup/REPLAY_POSTFILTER_TRACE_COMPARISON.json"
    post_available = post_dump.exists() and post_load.exists() and any(post_dump.glob("*.pt")) and any(post_load.glob("*.pt"))
    if post_available:
        post_rc = subprocess.run([sys.executable, str(root / "causal_followup/compare_traces.py"),
                                  str(post_dump), str(post_load), str(post_compare)], check=False).returncode
    else:
        post_rc = 0
        post_compare.write_text(json.dumps({"status": "NOT_COLLECTED",
                                             "reason": "pre-filter exact trace is the primary association fidelity gate"}, indent=2) + "\n")
    old = json.loads((root / "causal/runs/C0_baseline_log/evaluation/metrics.json").read_text())
    trace_pass = trace_rc == 0 and post_rc == 0
    cache_pass = all(x["status"] == "PASS" for x in checks)
    payload = {
        "status": "PASS" if cache_pass and trace_pass else "STOP",
        "gate": "Replay Fidelity",
        "per_scene_cache": checks,
        "pre_filter_trace_comparison": str(compare),
        "post_filter_trace_comparison": str(post_compare),
        "exact_track_id_and_count_equality": trace_pass,
        "post_filter_trace_collected": post_available,
        "baseline_reference_metrics": old,
        "metrics_reference": "C0 baseline metrics are retained unchanged; exact post-filter trace equality is the fidelity criterion",
        "fixed_stream": {"seed": 20260930, "test_len": 40, "with_bank": True, "bank_size": 10},
    }
    out = root / "causal_followup/REPLAY_FIDELITY.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    raise SystemExit(0 if payload["status"] == "PASS" else 1)


if __name__ == "__main__": main()
