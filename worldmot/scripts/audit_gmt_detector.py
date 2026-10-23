#!/usr/bin/env python3
"""Real TRAIN component preflight: native raw candidate parity and causality."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
from worldmot.models.gmt_detector_adapter import GMTDetectorAdapter, detection_rows


def canonical(dets):
    # Exact float serialization compares candidates and independent descriptors.
    return json.dumps(detection_rows(dets), sort_keys=True, allow_nan=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--metadata', required=True)
    p.add_argument('--dataset-root', default='/data/DATASETS/TRACKING/JDE/VisionTrack')
    p.add_argument('--scene', type=int, default=8)
    p.add_argument('--frames', type=int, default=8)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    if a.scene not in [4, 8, 12, 16, 20, 24]:
        raise ValueError('Preflight must use preregistered TRAIN validation scenes')
    meta = json.loads(Path(a.metadata).read_text())
    if meta.get('annotations'):
        raise ValueError('GT-bearing inference metadata rejected')
    rows = [r for r in meta['images'] if r['video_id'] == a.scene]
    times = sorted({r['frame_id'] for r in rows})[:a.frames]
    rows = [r for r in rows if r['frame_id'] in times]
    import torch
    torch.set_num_threads(2)
    detector = GMTDetectorAdapter(device=a.device)
    torch.cuda.reset_peak_memory_stats()
    report = {'status': 'RUNNING', 'split': 'train', 'scene': a.scene,
              'metadata_sha256': hashlib.sha256(Path(a.metadata).read_bytes()).hexdigest(),
              'detector': detector.manifest(), 'timestamps': times, 'records': []}
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(out)
    out.write_text(json.dumps(report, indent=2))
    last_images = None
    last_signature = None
    for timestamp in times:
        current = sorted([r for r in rows if r['frame_id'] == timestamp], key=lambda r:r['view_id'])
        assert [r['view_id'] for r in current] == [1, 2]
        images = {r['view_id']: np.asarray(Image.open(Path(a.dataset_root)/'train'/r['file_name']).convert('RGB')) for r in current}
        dets = detector.infer(images)
        diagnostics = detector.last_diagnostics.copy()
        parity = []
        # Native current-image inference is exercised separately from all
        # tracking functions, which remain installed as hard-failure guards.
        from common import map_records
        mapped_pair = map_records(detector.cfg, [dict(r, image_id=r['id'], view_num=2,
            file_name=str(Path(a.dataset_root)/'train'/r['file_name'])) for r in current])
        for r in current:
            view = r['view_id']
            mapped = [m for m in mapped_pair if m['view_id'] == view]
            with torch.inference_mode():
                native = detector.model.inference(mapped, 2, [timestamp,timestamp,[timestamp]], view-1,
                                                  do_postprocess=True)[0]['instances']
            boxes = native.pred_boxes.tensor.cpu().numpy()
            scores = native.scores.cpu().numpy()
            ours = np.array([[d.bbox_xywh[0],d.bbox_xywh[1],d.bbox_xywh[0]+d.bbox_xywh[2],d.bbox_xywh[1]+d.bbox_xywh[3]] for d in dets[view]],np.float32).reshape(-1,4)
            assert boxes.shape == ours.shape, (boxes.shape, ours.shape)
            assert np.array_equal(boxes,ours), 'Native raw boxes differ'
            assert np.array_equal(scores,np.array([d.score for d in dets[view]],np.float32)), 'Native scores differ'
            assert not native.has('track_ids'), 'Native detection contains history identity'
            isolated = detector.infer({view: images[view]})
            assert canonical({view:dets[view]}) == canonical(isolated), 'Single-image/prefix mismatch'
            parity.append({'view':view,'native_count':len(boxes),'exact_box_score_parity':True,'single_image_prefix_exact':True})
        report['records'].append({'timestamp':timestamp,'parity':parity,'diagnostics':diagnostics})
        last_images, last_signature = images, canonical(dets)
        print(json.dumps({'timestamp':timestamp,'counts':{v:len(x) for v,x in dets.items()},'native_parity':True}),flush=True)
    # Perturb/order history by revisiting an earlier frame. The final raw
    # candidates and descriptors must remain identical for the same pixels.
    replay = detector.infer(last_images)
    assert canonical(replay) == last_signature
    assert detector.forbidden_calls == {}
    report.update(status='PASS_REAL_RAW_DETECTOR_PARITY_AND_PREFIX_CAUSALITY',
                  history_future_trajectory_ID_dependency=False,
                  GTA_calls=0, peak_GPU_bytes=torch.cuda.max_memory_allocated(),
                  hardware=torch.cuda.get_device_name(0), point_or_tracking_models_used=False)
    out.write_text(json.dumps(report,indent=2,allow_nan=False))


if __name__ == '__main__':
    main()
