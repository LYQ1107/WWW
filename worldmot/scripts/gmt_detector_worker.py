#!/usr/bin/env python3
"""Persistent GMT-environment worker; stdout is a strict JSON protocol."""
import argparse
import contextlib
import json
import sys
import traceback
import numpy as np
from worldmot.models.gmt_detector_adapter import GMTDetectorAdapter, detection_rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--score-threshold', type=float)
    args = p.parse_args()
    import torch
    torch.set_num_threads(2)
    with contextlib.redirect_stdout(sys.stderr):
        detector = GMTDetectorAdapter(args.source, args.device, score_threshold=args.score_threshold)
    print(json.dumps({'status': 'READY', 'manifest': detector.manifest()}), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        if request.get('close'):
            break
        try:
            images = {int(v): np.load(path, allow_pickle=False) for v, path in request['images'].items()}
            with contextlib.redirect_stdout(sys.stderr):
                detections = detector.infer(images)
            print(json.dumps({'detections': detection_rows(detections), 'timings': detector.last_timings,
                              'diagnostics': detector.last_diagnostics}, allow_nan=False), flush=True)
        except Exception:
            print(json.dumps({'error': traceback.format_exc()}), flush=True)
            raise


if __name__ == '__main__':
    main()
