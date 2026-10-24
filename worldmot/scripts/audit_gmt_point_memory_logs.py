#!/usr/bin/env python3
"""Read actual CUDA allocation peaks from saved VGGT forward diagnostics."""
import argparse
import json
from pathlib import Path
from gmt_point_report_common import NAS, artifact, read_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prefix', type=int, default=8)
    parser.add_argument('--nas', type=Path, default=NAS)
    args = parser.parse_args()
    methods = {}
    for method in ('temporal', 'joint'):
        scenes, all_peaks = [], []
        for sid in range(1, 23):
            directory = args.nas / 'vggt_point_association_20261023/test_v1' / method / f'scene_{sid:02d}'
            manifest = read_json(directory / 'manifest.json')
            peaks, timestamps = [], []
            with (directory / 'tracks.jsonl').open() as stream:
                for line in stream:
                    row = json.loads(line)
                    timestamps.append(row['timestamp'])
                    for call in row.get('point_forward_calls', []):
                        value = call.get('diagnostics', {}).get('peak_GPU_bytes')
                        if value is not None:
                            peaks.append(int(value))
                    if len(timestamps) >= args.prefix:
                        break
            scenes.append(dict(scene=sid, timestamps=timestamps, forward_records=len(peaks),
                               peak_allocated_VGGT_process_bytes=max(peaks) if peaks else None,
                               source_prediction_sha256=manifest['prediction_sha256'],
                               manifest=artifact(directory / 'manifest.json')))
            all_peaks.extend(peaks)
        methods[method] = dict(recorded_forward_count=len(all_peaks),
                               maximum_sampled_peak_allocated_VGGT_process_bytes=max(all_peaks), scenes=scenes)
    result = dict(status='MEASURED_SAVED_FORWARD_PEAKS_SAMPLED_PREFIX_ONLY', new_GPU_work=False,
                  prefix_timestamps_per_video=args.prefix, methods=methods,
                  whole_system_peak_GPU_memory_bytes='NOT_AVAILABLE',
                  scope='torch.cuda.max_memory_allocated from real saved VGGT forward calls; first N timestamps in each of 22 videos; includes resident VGGT allocations, excludes the separate GMT worker and CUDA reserved/driver memory.',
                  limitation='Sampled maximum is a lower bound on the full-video VGGT-process peak. Do not relabel it as a whole-system peak or full-video maximum.')
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: {x: y for x, y in v.items() if x != 'scenes'} for k, v in methods.items()}), flush=True)


if __name__ == '__main__':
    main()
