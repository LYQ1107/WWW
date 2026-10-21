"""Fresh synchronized two-view causal GMT prefix latency, without Ω or caches."""
import argparse
import collections
import json
from pathlib import Path
import time

import numpy as np
import torch

from common import load_gmt, map_records, save_json, seed_all, sha256, train_data, video_records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=int, choices=[2, 8, 17], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    from gtr.geometry.online_runner import CausalGMT
    torch.set_num_threads(2)
    cfg, model = load_gmt(seed=20261021, geo_opts=dict(ENABLED=False))
    seed_all(20261021)
    records = video_records(train_data(), args.scene, max_frames=100)
    groups = collections.defaultdict(list)
    for r in records:
        groups[int(r['frame_id'])].append(r)
    tracker = CausalGMT(model)
    rows = []
    torch.cuda.reset_peak_memory_stats()
    for frame, group in sorted(groups.items()):
        torch.cuda.synchronize()
        start = time.perf_counter()
        mapped = map_records(cfg, sorted(group, key=lambda r: r['view_id']))
        outputs = tracker.step(mapped)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        rows.append(dict(frame=frame, seconds=elapsed, detections=sum(len(o['instances']) for o in outputs)))
    times = [r['seconds'] for r in rows[10:]]
    report = dict(status='COMPLETE_SHORT_CAUSAL_GMT_TIMING_PROBE', scene=args.scene, continuous_prefix_groups=len(rows),
                  excluded_warmup_groups=10, measured_groups=len(times), model_resident=True,
                  scope='read/decode/preprocess/two-view detection/causal association/final postprocess; full-video and Omega integration not measured',
                  synchronized_timestamp_fps=len(times) / sum(times), camera_image_fps=2 * len(times) / sum(times),
                  mean_ms=float(np.mean(times) * 1000), p50_ms=float(np.median(times) * 1000), p95_ms=float(np.quantile(times, .95) * 1000),
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(), rows=rows,
                  script_sha256=sha256(__file__), gt_used=False, omega_enabled=False)
    save_json(args.output, report)
    print(json.dumps({k: v for k, v in report.items() if k != 'rows'}), flush=True)


if __name__ == '__main__':
    main()
