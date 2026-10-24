#!/usr/bin/env python3
"""Compare recorded detector stages on identical held-out TRAIN images."""
import argparse
import json
from pathlib import Path
import statistics
import numpy as np
from gmt_point_report_common import NAS, artifact, read_json
from build_gmt_point_comparison import write_csv


def summary(values):
    return dict(timestamps=len(values), images=2 * len(values), total_stage_seconds=sum(values),
                mean_seconds_per_image=sum(values) / (2 * len(values)),
                dual_camera_stage_P50_s=statistics.median(values), dual_camera_stage_P95_s=float(np.quantile(values, .95)),
                detector_stage_dual_camera_FPS=len(values) / sum(values))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--nas', type=Path, default=NAS)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    all_values = {'GMT': [], 'RTMDet': []}
    scenes = []
    for sid in (4, 8, 12, 16, 20, 24):
        gdir = args.nas / 'gmt_detector_comparison_20261023/raw_train_v1' / f'scene_{sid:02d}'
        rdir = args.nas / 'fulltrain_20261022/train_cache' / f'scene_{sid:02d}'
        gm, rm = read_json(gdir / 'manifest.json'), read_json(rdir / 'manifest.json')
        g = {}
        for shard in gm['shards']:
            for row in read_json(gdir / Path(shard['file']).with_suffix('.json'))['records']:
                g[row['timestamp']] = row
        r = {}
        for shard in rm['shards']:
            for row in read_json(rdir / Path(shard['file']).with_suffix('.json'))['per_frame']:
                r[row['timestamp']] = row
        if gm['status'] != 'COMPLETE' or rm['status'] != 'COMPLETE' or set(g) != set(r):
            raise ValueError('Detector timing coverage differs')
        for timestamp in g:
            if g[timestamp]['image_ids'] != r[timestamp]['image_ids']:
                raise ValueError('Detector timings are not for the same RGB images')
        values = {'GMT': [g[t]['detector_timings']['infer_wall_s'] for t in sorted(g)],
                  'RTMDet': [r[t]['runtime']['detector_s'] for t in sorted(r)]}
        for detector, samples in values.items():
            all_values[detector].extend(samples)
            scenes.append(dict(scene=sid, detector=detector, **summary(samples)))
    combined = {detector: summary(samples) for detector, samples in all_values.items()}
    result = dict(status='PAIRED_COMPLETE_SIX_TRAIN_VIDEO_RECORDED_STAGE_TIMING', TEST_used=False,
                  new_GPU_inference=False, paired_image_IDs_verified=True, scenes=scenes, combined=combined,
                  quality=read_json(Path(__file__).resolve().parents[1] / 'results/gmt_detector_train_validation_quality.json')['combined'],
                  scope='GMT raw-export infer_wall_s includes both current cameras, preprocessing, proposals and frozen ROI descriptors. RTMDet detector_s is the sum of two real per-camera infer timers including its mask/ROI processing. Neither is tracking E2E FPS.',
                  limitations='These are archived shared-resource runs, not an exclusive controlled latency ranking. Raw visibility/duplicate-GT limitations of the six-scene quality diagnostic remain unchanged.')
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    write_csv(args.output.with_suffix('.csv'), [dict(scope='six TRAIN videos combined', detector=k, **v) for k, v in combined.items()] + scenes)
    print(json.dumps(combined), flush=True)


if __name__ == '__main__':
    main()
