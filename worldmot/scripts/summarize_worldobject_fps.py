#!/usr/bin/env python3
"""Summarize recorded timing logs; does not execute inference or benchmarks."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def timing_statistics(samples):
    samples=np.asarray(samples,dtype=float)
    if not len(samples) or not np.isfinite(samples).all() or np.any(samples<=0):
        raise ValueError("INVALID_OR_MISSING_WALL_CLOCK_SAMPLES")
    p50,p95=np.percentile(samples,[50,95])
    return {"samples":len(samples),"p50_s":float(p50),"p95_s":float(p95),
            "fps_from_p50_latency":float(1/p50),"fps_from_p95_latency":float(1/p95),
            "serial_throughput_pairs_per_second":float(len(samples)/samples.sum())}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--predictions",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args();groups={}
    for path in sorted(args.predictions.glob("scene_*/timings.jsonl")):
        manifest=json.loads(path.with_name("manifest.json").read_text())
        label=manifest["fps_definition"]
        for line in path.read_text().splitlines():
            row=json.loads(line)
            for key in ("cache_replay_s","live_rgb_to_id_s","rgb_read_s"):
                if key in row:
                    groups.setdefault((label,key),[]).append(row[key])
            for key,value in row.get("stages",{}).items():
                if value>0:
                    groups.setdefault((label,"stage_"+key),[]).append(value)
    records=[{"fps_definition":label,"stage":stage,**timing_statistics(values)}
             for (label,stage),values in sorted(groups.items())]
    if not records:
        raise ValueError("NO_EXECUTED_TIMING_LOGS_FOUND")
    with args.output.open("x") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)


if __name__=="__main__":
    main()
