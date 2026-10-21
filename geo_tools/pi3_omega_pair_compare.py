"""Pi3 (original) versus VGGT-Omega paired-view comparison.

This is a standalone research probe.  It never changes GMT's production
configuration or tracker.  The probe uses the same 54 frozen TRAIN pairs,
the same exact RGB resize, and a separate live GMT detector pass whose boxes
are frozen before either geometry model is evaluated.

The reconstruction metric is intentionally GT-free: both models are reduced
to dense world point maps and scored on cross-view point agreement at fixed
SIFT matches.  TRAIN identities are read only by evaluate() after all model
artifacts are complete, for the separate same-person ranking metric.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = Path('/data/DATASETS/TRACKING/JDE/VisionTrack')
PRIOR_PLAN = ROOT / 'manifests/omega_pair_geometry_fps_probe_v1.json'
PLAN = ROOT / 'manifests/pi3_omega_pair_comparison_v1.json'
RUNTIME = Path('/data1/liuyeqiang/WWW_pi3_omega_pair_comparison_runtime_20261022')
PI3_SOURCE = Path('/data1/liuyeqiang/Pi3_src_20261022')
PI3_WEIGHT = Path('/data1/liuyeqiang/Pi3_weights_20261022/model.safetensors')
OMEGA_SOURCE = Path('/data1/liuyeqiang/WWW_omega_runtime_20261016/official_vggt_omega')
OMEGA_WEIGHT = Path('/data1/liuyeqiang/WWW_omega_runtime_20261016/weights/vggt_omega_1b_512.pt')
OMEGA_WEIGHT_SHA = 'c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934'
GMT_CHECKPOINT = Path('/data1/liuyeqiang/WWW/outputs/stage2_single_gpu/model_20000.pth')
TARGET_W, TARGET_H = 560, 336
SEED = 20261022


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path: Path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def summary(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return dict(n=0, mean_ms=None, p50_ms=None, p95_ms=None)
    return dict(n=int(values.size), mean_ms=float(values.mean() * 1000),
                p50_ms=float(np.median(values) * 1000),
                p95_ms=float(np.quantile(values, .95) * 1000))


def load_plan():
    return json.loads(PLAN.read_text())


def prepare():
    """Freeze the exact pair list and the common image transform."""
    prior = json.loads(PRIOR_PLAN.read_text())
    samples = []
    for pair in prior['samples']:
        copied = json.loads(json.dumps(pair))
        copied['common_transform'] = dict(kind='bicubic_resize', width=TARGET_W,
                                          height=TARGET_H,
                                          x_scale=TARGET_W / pair['images'][0]['width'],
                                          y_scale=TARGET_H / pair['images'][0]['height'])
        samples.append(copied)
    plan = dict(
        status='LOCKED_BEFORE_INFERENCE', version='pi3_omega_pair_comparison_v1',
        created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), seed=SEED,
        source_plan=str(PRIOR_PLAN), source_plan_sha256=sha256(PRIOR_PLAN),
        sampling='exactly the prior 18 fit scenes x 3 temporal quantiles = 54 pairs; no output selection',
        purpose='same-input original Pi3 versus original VGGT-Omega paired-view comparison',
        samples=samples,
        common_input=dict(width=TARGET_W, height=TARGET_H, channels=3,
                          interpolation='PIL.Image.Resampling.BICUBIC',
                          pixel_range='[0,1]', pair_shape='B=1,S=2',
                          source_aspect_ratio='1920x1080 resized to 560x336; fixed mild aspect deformation'),
        models=dict(
            pi3=dict(source=str(PI3_SOURCE), source_commit='9fa3ddb3f8d53041f8b2738df404f62223bbaa7b',
                     weight=str(PI3_WEIGHT),
                     weight_sha256=sha256(PI3_WEIGHT) if PI3_WEIGHT.exists() else None,
                     variant='original Pi3, not Pi3X',
                     patch_size=14, precision='explicit fp16 AMP with fp32 output heads'),
            omega=dict(source=str(OMEGA_SOURCE), source_commit='48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6',
                       weight=str(OMEGA_WEIGHT), weight_sha256=OMEGA_WEIGHT_SHA,
                       variant='original VGGT-Omega 512 checkpoint', patch_size=16,
                       precision='explicit fp16 AMP with official fp32 camera/depth heads')),
        detector=dict(status='PENDING_LIVE_GMT_DETECTOR', checkpoint=str(GMT_CHECKPOINT),
                      cache=str(RUNTIME / 'detector_cache.json'),
                      scope='original GMT detector only; no GMT association/tracker state'),
        metrics=dict(
            reconstruction='GT-free fixed SIFT correspondence world-point agreement; lower normalized error is better',
            association='same-person ranking from live detector boxes, labels posthoc by mutual-best TRAIN GT IoU>=0.5; duplicate instance_id excluded',
            abstention='missing/invalid point samples, tied best score, or no positive-vs-negative eligible query',
            fps='model-only and RGB-read-to-assignment; composed serial FPS adds measured live detector latency; one unit is one synchronized two-camera timestamp'),
        gmt_main_flow_changed=False,
    )
    if not OMEGA_WEIGHT.exists() or sha256(OMEGA_WEIGHT) != OMEGA_WEIGHT_SHA:
        raise ValueError('Omega checkpoint missing or hash mismatch')
    if not GMT_CHECKPOINT.exists():
        raise FileNotFoundError(GMT_CHECKPOINT)
    save_json(PLAN, plan)
    print(json.dumps(dict(plan=str(PLAN), pairs=len(samples), sha256=sha256(PLAN))))


def _metadata_index():
    data = json.loads((DATA / 'annotations/train.json').read_text())
    return data, {int(x['id']): x for x in data['images']}


def _records_for_pair(pair, image_index):
    records = []
    for item in pair['images']:
        meta = dict(image_index[int(item['image_id'])])
        meta['image_id'] = meta['id']
        meta['file_name'] = str(DATA / 'train' / meta['file_name'])
        meta['view_num'] = 2
        records.append(meta)
    return sorted(records, key=lambda x: int(x['view_id']))


def detect():
    """Run the unchanged GMT detector once and freeze boxes/timing.

    This command is run with the GMT environment.  Only detector inference is
    called; no CausalGMT, geometry runtime, assignment, or tracker history is
    entered.
    """
    import torch
    from common import load_gmt, map_records, seed_all

    plan = load_plan()
    if plan['status'] != 'LOCKED_BEFORE_INFERENCE':
        raise ValueError('prepare must run before detect')
    if (RUNTIME / 'detector_cache.json').exists():
        raise FileExistsError(RUNTIME / 'detector_cache.json')
    _data, image_index = _metadata_index()
    torch.set_num_threads(2)
    seed_all(SEED)
    cfg, model = load_gmt(seed=SEED, geo_opts=dict(ENABLED=False))
    model.eval()

    first_records = _records_for_pair(plan['samples'][0], image_index)
    first_mapped = map_records(cfg, first_records)
    with torch.inference_mode():
        for _ in range(3):
            for v in range(2):
                model.inference([first_mapped[v]], 2,
                                [first_records[v]['frame_id']] * 3, v,
                                do_postprocess=True)
        torch.cuda.synchronize()

    rows = []
    for pair in plan['samples']:
        records = _records_for_pair(pair, image_index)
        torch.cuda.synchronize()
        row_start = time.perf_counter()
        mapped_start = time.perf_counter()
        mapped = map_records(cfg, records)
        read_seconds = time.perf_counter() - mapped_start
        outputs = []
        model_start = time.perf_counter()
        with torch.inference_mode():
            for v in range(2):
                outputs.extend(model.inference([mapped[v]], 2,
                                               [records[v]['frame_id']] * 3, v,
                                               do_postprocess=True))
            torch.cuda.synchronize()
        detector_seconds = time.perf_counter() - model_start
        row_seconds = time.perf_counter() - row_start
        boxes = []
        for out in outputs:
            inst = out['instances']
            boxes.append(inst.pred_boxes.tensor.detach().cpu().numpy().astype(float).tolist())
        rows.append(dict(pair_id=pair['pair_id'], scene=pair['scene'], frame=pair['frame'],
                         boxes_original=boxes, detections=[len(x) for x in boxes],
                         read_preprocess_seconds=read_seconds,
                         detector_model_seconds=detector_seconds,
                         pair_seconds=row_seconds))
        if len(rows) % 6 == 0:
            print(json.dumps(dict(stage='detector', completed=len(rows), total=len(plan['samples']),
                                  last_pair=pair['pair_id'])), flush=True)
    report = dict(status='COMPLETE_LIVE_GMT_DETECTOR_CACHE', plan_sha256=sha256(PLAN),
                  rows=rows, gpu=torch.cuda.get_device_name(),
                  cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  torch=torch.__version__, checkpoint=str(GMT_CHECKPOINT),
                  checkpoint_sha256=sha256(GMT_CHECKPOINT), model_resident=True,
                  detector_model=summary([r['detector_model_seconds'] for r in rows]),
                  read_preprocess=summary([r['read_preprocess_seconds'] for r in rows]),
                  pair_serial=summary([r['pair_seconds'] for r in rows]),
                  pair_timestamp_fps=float(len(rows) / sum(r['pair_seconds'] for r in rows)),
                  camera_image_fps=float(2 * len(rows) / sum(r['pair_seconds'] for r in rows)),
                  association_or_tracker_called=False, gt_used=False,
                  script_sha256=sha256(Path(__file__)))
    save_json(RUNTIME / 'detector_cache.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'rows'}), flush=True)


def _load_images(paths):
    import torch
    from torchvision.transforms.functional import pil_to_tensor
    tensors = []
    for path in paths:
        with Image.open(path) as image:
            image = image.convert('RGB').resize((TARGET_W, TARGET_H), Image.Resampling.BICUBIC)
            tensors.append(pil_to_tensor(image).float().div_(255.0))
    return torch.stack(tensors, dim=0).unsqueeze(0)


def _sample_map(point_map, uv):
    """Bilinear sample a [H,W,3] point map at floating image coordinates."""
    uv = np.asarray(uv, dtype=np.float32)
    out = []
    for c in range(3):
        out.append(cv2.remap(point_map[..., c].astype(np.float32, copy=False),
                             uv[:, 0:1], uv[:, 1:2], cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0).reshape(-1))
    return np.stack(out, axis=1)


def _bbox_points(point_map, box, source_w, source_h):
    x0, y0, x1, y1 = np.asarray(box, dtype=float)
    sx, sy = TARGET_W / source_w, TARGET_H / source_h
    x0, x1, y0, y1 = x0 * sx, x1 * sx, y0 * sy, y1 * sy
    xs = x0 + np.linspace(.2, .8, 5) * (x1 - x0)
    ys = y0 + np.linspace(.2, .9, 7) * (y1 - y0)
    gx, gy = np.meshgrid(xs, ys)
    uv = np.column_stack([gx.ravel(), gy.ravel()])
    pts = _sample_map(point_map, uv)
    valid = np.isfinite(pts).all(axis=1) & (np.linalg.norm(pts, axis=1) > 1e-7)
    return pts[valid]


def world_box_scores(pair, boxes, points):
    """Return symmetric point-cloud scores for every cross-view box pair."""
    clouds = []
    for view in range(2):
        clouds.append([_bbox_points(points[view], box,
                                     pair['images'][view]['width'], pair['images'][view]['height'])
                       for box in boxes[view]])
    valid_all = [p for view in clouds for p in view if len(p)]
    if not valid_all:
        return dict(world_chamfer=np.full((len(boxes[0]), len(boxes[1])), -1e6),
                    world_centroid=np.full((len(boxes[0]), len(boxes[1])), -1e6),
                    valid_points=0, scene_scale=0.0)
    scale = float(np.median(np.linalg.norm(np.concatenate(valid_all), axis=1)))
    scale = max(scale, 1e-6)
    chamfer = np.full((len(clouds[0]), len(clouds[1])), -1e6, dtype=float)
    centroid = np.full_like(chamfer, -1e6)
    for i, a in enumerate(clouds[0]):
        if not len(a):
            continue
        for j, b in enumerate(clouds[1]):
            if not len(b):
                continue
            d = np.linalg.norm(a[:, None] - b[None, :], axis=2)
            chamfer[i, j] = -float(.5 * (np.median(d.min(axis=1)) + np.median(d.min(axis=0))) / scale)
            centroid[i, j] = -float(np.linalg.norm(np.median(a, axis=0) - np.median(b, axis=0)) / scale)
    return dict(world_chamfer=chamfer, world_centroid=centroid,
                valid_points=int(sum(len(x) for x in valid_all)), scene_scale=scale)


def _sift_matches(pair):
    sift = cv2.SIFT_create(nfeatures=400)
    images = []
    for item in pair['images']:
        with Image.open(item['path']) as image:
            images.append(np.asarray(image.convert('L')))
    k0, d0 = sift.detectAndCompute(images[0], None)
    k1, d1 = sift.detectAndCompute(images[1], None)
    if d0 is None or d1 is None:
        return np.empty((0, 2)), np.empty((0, 2))
    matches = cv2.BFMatcher(cv2.NORM_L2).knnMatch(d0, d1, k=2)
    good = [m[0] for m in matches if len(m) == 2 and m[0].distance < .75 * m[1].distance]
    good.sort(key=lambda m: (m.distance, m.queryIdx, m.trainIdx))
    good = good[:128]
    return (np.asarray([k0[m.queryIdx].pt for m in good], dtype=float),
            np.asarray([k1[m.trainIdx].pt for m in good], dtype=float))


def reconstruction_proxy(pair, points):
    a, b = _sift_matches(pair)
    if len(a) == 0:
        return dict(matches=0, valid=0, median_normalized_error=None, p90_normalized_error=None)
    pa = _sample_map(points[0], a * np.asarray([TARGET_W / pair['images'][0]['width'], TARGET_H / pair['images'][0]['height']]))
    pb = _sample_map(points[1], b * np.asarray([TARGET_W / pair['images'][1]['width'], TARGET_H / pair['images'][1]['height']]))
    valid = np.isfinite(pa).all(axis=1) & np.isfinite(pb).all(axis=1)
    valid &= (np.linalg.norm(pa, axis=1) > 1e-7) & (np.linalg.norm(pb, axis=1) > 1e-7)
    if not valid.any():
        return dict(matches=int(len(a)), valid=0, median_normalized_error=None, p90_normalized_error=None)
    scale = max(float(np.median(np.linalg.norm(np.concatenate([pa[valid], pb[valid]]), axis=1))), 1e-6)
    errors = np.linalg.norm(pa[valid] - pb[valid], axis=1) / scale
    return dict(matches=int(len(a)), valid=int(valid.sum()),
                median_normalized_error=float(np.median(errors)),
                p90_normalized_error=float(np.quantile(errors, .9)))


def _load_model(kind, device):
    import torch
    torch.set_num_threads(2)
    if kind == 'omega':
        sys.path.insert(0, str(OMEGA_SOURCE))
        from vggt_omega.models import VGGTOmega
        from vggt_omega.utils.pose_enc import encoding_to_camera
        model = VGGTOmega(autocast=False).eval().to(device).requires_grad_(False)
        state = torch.load(OMEGA_WEIGHT, map_location='cpu', weights_only=True)
        state = state.get('model', state.get('state_dict', state))
        model.load_state_dict(state, strict=True)
        del state
        return model, encoding_to_camera
    if kind == 'pi3':
        sys.path.insert(0, str(PI3_SOURCE))
        from pi3.models.pi3 import Pi3
        from safetensors.torch import load_file
        model = Pi3().eval().to(device).requires_grad_(False)
        state = load_file(str(PI3_WEIGHT), device='cpu')
        model.load_state_dict(state, strict=True)
        del state
        return model, None
    raise ValueError(kind)


def _omega_world(predictions, encoding_to_camera, shape):
    import torch
    E, K = encoding_to_camera(predictions['pose_enc'], shape[-2:])
    depth = predictions['depth'][0, ..., 0].float()
    _B, H, W = depth.shape
    yy, xx = torch.meshgrid(torch.arange(H, device=depth.device, dtype=torch.float32) + .5,
                            torch.arange(W, device=depth.device, dtype=torch.float32) + .5,
                            indexing='ij')
    pix = torch.stack([xx, yy, torch.ones_like(xx)], dim=-1)
    rays = torch.einsum('bnij,hwj->bnhwi', torch.linalg.inv(K.float()), pix)
    camera = rays * depth[..., None]
    # encoding_to_camera returns OpenCV camera-from-world [R|t].
    world = torch.einsum('bnji,bnhwj->bnhwi', E[..., :3, :3].float(),
                         camera - E[..., None, None, :, 3].float())
    return world.cpu().numpy()[0].astype(np.float32), predictions['depth_conf'][0].float().cpu().numpy()


def _pi3_world(predictions):
    import torch
    points = predictions['points'][0].float().cpu().numpy().astype(np.float32)
    conf = torch.sigmoid(predictions['conf'][0, ..., 0].float()).cpu().numpy()
    return points, conf


def infer(kind: str):
    import torch
    plan = load_plan()
    if plan['status'] != 'LOCKED_BEFORE_INFERENCE':
        raise ValueError('prepare must run before infer')
    detector_path = RUNTIME / 'detector_cache.json'
    if not detector_path.exists():
        raise FileNotFoundError('run detect first: ' + str(detector_path))
    detector = json.loads(detector_path.read_text())
    if detector['plan_sha256'] != sha256(PLAN):
        raise ValueError('detector cache was made for a different plan')
    if kind == 'pi3' and (not PI3_WEIGHT.exists() or PI3_WEIGHT.stat().st_size < 1000000000):
        raise FileNotFoundError('Pi3 weight is missing or incomplete')
    if kind == 'omega' and sha256(OMEGA_WEIGHT) != OMEGA_WEIGHT_SHA:
        raise ValueError('Omega checkpoint hash mismatch')

    run_tag = os.environ.get('PI3_OMEGA_RUN_TAG', '').strip()
    out = RUNTIME / (f'{kind}_{run_tag}' if run_tag else kind)
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    save_json(out / 'status.json', dict(status='RUNNING', kind=kind, plan_sha256=sha256(PLAN)))
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device('cuda')
    process_start = time.perf_counter()
    model, decoder = _load_model(kind, device)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - process_start

    def model_forward(images):
        with torch.autocast('cuda', dtype=torch.float16):
            pred = model(images)
        torch.cuda.synchronize()
        return pred

    def decode_world(pred, shape):
        if kind == 'omega':
            points, conf = _omega_world(pred, decoder, shape)
        else:
            points, conf = _pi3_world(pred)
        return points, conf

    warm_images = _load_images([i['path'] for i in plan['samples'][0]['images']]).to(device)
    with torch.inference_mode():
        for _ in range(4):
            tmp = model_forward(warm_images)
            _warm_points, _warm_conf = decode_world(tmp, warm_images.shape)
        torch.cuda.synchronize()
    del tmp, _warm_points, _warm_conf, warm_images
    torch.cuda.reset_peak_memory_stats()

    rows = []
    detector_rows = {r['pair_id']: r for r in detector['rows']}
    with torch.inference_mode():
        for pair in plan['samples']:
            if pair['pair_id'] not in detector_rows:
                raise KeyError(pair['pair_id'])
            for item in pair['images']:
                if sha256(Path(item['path'])) != item['sha256']:
                    raise ValueError('input image hash changed: ' + item['path'])
            torch.cuda.synchronize()
            serial_start = time.perf_counter()
            images = _load_images([i['path'] for i in pair['images']])
            images = images.to(device, non_blocking=False)
            torch.cuda.synchronize()
            pre_h2d = time.perf_counter() - serial_start
            model_start = time.perf_counter()
            pred = model_forward(images)
            model_seconds = time.perf_counter() - model_start
            decode_start = time.perf_counter()
            points, conf = decode_world(pred, images.shape)
            torch.cuda.synchronize()
            decode_seconds = time.perf_counter() - decode_start
            output_copy_start = time.perf_counter()
            boxes = detector_rows[pair['pair_id']]['boxes_original']
            scores = world_box_scores(pair, boxes, points)
            serial_core_seconds = time.perf_counter() - serial_start
            # SIFT reconstruction quality and archival writes are offline
            # diagnostics; neither belongs to an online identity-assignment
            # FPS.  Keep them outside the core serial timing.
            proxy_start = time.perf_counter()
            proxy = reconstruction_proxy(pair, points)
            archive = out / (pair['pair_id'] + '.npz')
            np.savez_compressed(archive, points=points, confidence=conf)
            offline_seconds = time.perf_counter() - proxy_start
            output_copy_seconds = time.perf_counter() - output_copy_start
            rows.append(dict(pair_id=pair['pair_id'], scene=pair['scene'], frame=pair['frame'],
                             shape=list(images.shape), finite=bool(np.isfinite(points).all()),
                             model_seconds=model_seconds, read_preprocess_h2d_seconds=pre_h2d,
                             geometry_decode_seconds=decode_seconds,
                             model_plus_decode_seconds=model_seconds + decode_seconds,
                             serial_rgb_to_assignment_seconds=serial_core_seconds,
                             geometry_assignment_seconds=serial_core_seconds - pre_h2d - model_seconds,
                             offline_quality_and_archive_seconds=offline_seconds,
                             output_copy_geometry_seconds=output_copy_seconds,
                             archive_sha256=sha256(archive),
                             point_shape=list(points.shape), confidence_finite=bool(np.isfinite(conf).all()),
                             confidence_median=float(np.nanmedian(conf)),
                             proxy=proxy, score_summary=dict(
                                 chamfer_valid=int(np.sum(scores['world_chamfer'] > -1e5)),
                                 centroid_valid=int(np.sum(scores['world_centroid'] > -1e5)),
                                 valid_points=int(scores['valid_points']),
                                 scene_scale=float(scores['scene_scale']))))
            del pred, points, conf, images
            if len(rows) % 6 == 0:
                print(json.dumps(dict(stage=kind, completed=len(rows), total=len(plan['samples']),
                                      last_pair=pair['pair_id'])), flush=True)
    report = dict(status='COMPLETE_PAIRED_GEOMETRY_INFERENCE', kind=kind, run_tag=run_tag or 'primary',
                  plan_sha256=sha256(PLAN), plan_source_sha256=plan['source_plan_sha256'],
                  gpu=torch.cuda.get_device_name(), cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  torch=torch.__version__, cuda=torch.version.cuda,
                  source=str(PI3_SOURCE if kind == 'pi3' else OMEGA_SOURCE),
                  source_commit=plan['models'][kind]['source_commit'],
                  weight=str(PI3_WEIGHT if kind == 'pi3' else OMEGA_WEIGHT),
                  weight_sha256=sha256(PI3_WEIGHT if kind == 'pi3' else OMEGA_WEIGHT),
                  precision=plan['models'][kind]['precision'], load_seconds=load_seconds,
                  warmup_calls=4, model=summary([r['model_seconds'] for r in rows]),
                  model_plus_decode=summary([r['model_plus_decode_seconds'] for r in rows]),
                  geometry_decode=summary([r['geometry_decode_seconds'] for r in rows]),
                  read_preprocess_h2d=summary([r['read_preprocess_h2d_seconds'] for r in rows]),
                  geometry_assignment=summary([r['geometry_assignment_seconds'] for r in rows]),
                  serial_rgb_to_assignment=summary([r['serial_rgb_to_assignment_seconds'] for r in rows]),
                  model_pair_fps=float(len(rows) / sum(r['model_seconds'] for r in rows)),
                  module_e2e_pair_fps=float(len(rows) / sum(r['serial_rgb_to_assignment_seconds'] for r in rows)),
                  peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
                  peak_reserved_bytes=int(torch.cuda.max_memory_reserved()),
                  process_seconds=time.perf_counter() - process_start,
                  finite_pairs=sum(int(r['finite']) for r in rows), total_pairs=len(rows),
                  rows=rows, points_are_world=True, gt_used=False,
                  detector_cache_used=True, detector_cost_in_module_e2e=False,
                  fps_unit='one synchronized two-camera timestamp; detector cost is reported separately and composed serial FPS is evaluated posthoc')
    save_json(out / 'manifest.json', report)
    save_json(out / 'status.json', dict(status=report['status']))
    print(json.dumps({k: v for k, v in report.items() if k != 'rows'}), flush=True)


def bbox_iou(a, b):
    a, b = np.asarray(a, dtype=float).copy(), np.asarray(b, dtype=float).copy()
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    a[:, 2:] += a[:, :2]
    b[:, 2:] += b[:, :2]
    lo = np.maximum(a[:, None, :2], b[None, :, :2])
    hi = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.maximum(hi - lo, 0).prod(-1)
    union = (a[:, 2:] - a[:, :2]).prod(-1)[:, None] + (b[:, 2:] - b[:, :2]).prod(-1)[None] - inter
    return inter / np.maximum(union, 1e-8)


def _labels_for_boxes(boxes, item, gt):
    anns = gt[int(item['image_id'])]
    counts = collections.Counter(int(a['instance_id']) for a in anns)
    legal = [a for a in anns if counts[int(a['instance_id'])] == 1]
    labels = [None] * len(boxes)
    if legal and boxes:
        # The live Detectron2 cache stores xyxy, while TRAIN annotations are
        # xywh; convert only for the shared IoU helper.
        boxes_xywh = [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in boxes]
        iou = bbox_iou(boxes_xywh, [a['bbox'] for a in legal])
        for i, j in enumerate(iou.argmax(axis=1)):
            if iou[i, j] >= .5 and int(iou[:, j].argmax()) == i:
                labels[i] = int(legal[j]['instance_id'])
    return labels, sum(int(v != 1) for v in counts.values())


def _rank_metrics(scores, labels):
    counters = {name: collections.Counter() for name in ('world_chamfer', 'world_centroid')}
    for view in (0, 1):
        other = 1 - view
        known = np.asarray([x is not None for x in labels[other]], dtype=bool)
        for q, identity in enumerate(labels[view]):
            if identity is None or not known.any():
                continue
            positives = np.asarray([x == identity for x in labels[other]], dtype=bool)[known]
            if not positives.any() or positives.all():
                continue
            for name in counters:
                matrix = scores[name] if view == 0 else scores[name].T
                values = matrix[q, known]
                best = float(np.max(values)) if len(values) else -1e6
                winners = np.flatnonzero(np.isclose(values, best, rtol=0, atol=1e-8))
                status = 'abstain' if (len(winners) != 1 or best <= -1e5) else ('correct' if positives[winners[0]] else 'wrong')
                counters[name]['eligible'] += 1
                counters[name][status] += 1
    return counters


def _cluster_bootstrap(models, score_name, seed=SEED, draws=10000):
    """Scene-cluster bootstrap for the paired comparison (not iid directions)."""
    scenes = sorted(set(models['omega']['association']['by_scene']) &
                    set(models['pi3']['association']['by_scene']))
    if not scenes:
        return {}
    rng = np.random.default_rng(seed)
    rates = {}
    for kind in ('omega', 'pi3'):
        counts = []
        for scene in scenes:
            c = models[kind]['association']['by_scene'][scene][score_name]
            n = int(c.get('correct', 0)) + int(c.get('wrong', 0))
            counts.append((int(c.get('correct', 0)), n))
        total_correct = sum(x[0] for x in counts)
        total_n = sum(x[1] for x in counts)
        draws_out = []
        for _ in range(draws):
            selected = rng.integers(0, len(counts), size=len(counts))
            correct = sum(counts[i][0] for i in selected)
            n = sum(counts[i][1] for i in selected)
            draws_out.append(correct / n if n else np.nan)
        rates[kind] = dict(correct=int(total_correct), evaluated=int(total_n),
                           accuracy=float(total_correct / total_n) if total_n else None,
                           scene_bootstrap_95=[float(np.nanquantile(draws_out, .025)),
                                               float(np.nanquantile(draws_out, .975))])
    delta = np.asarray([], dtype=float)
    paired = []
    omega_counts = []
    pi3_counts = []
    for scene in scenes:
        for kind, out in (('omega', omega_counts), ('pi3', pi3_counts)):
            c = models[kind]['association']['by_scene'][scene][score_name]
            out.append((int(c.get('correct', 0)), int(c.get('correct', 0)) + int(c.get('wrong', 0))))
    for _ in range(draws):
        selected = rng.integers(0, len(scenes), size=len(scenes))
        on = sum(omega_counts[i][0] for i in selected)
        od = sum(omega_counts[i][1] for i in selected)
        pn = sum(pi3_counts[i][0] for i in selected)
        pd = sum(pi3_counts[i][1] for i in selected)
        paired.append((pn / pd if pd else np.nan) - (on / od if od else np.nan))
    delta = np.asarray(paired, dtype=float)
    return dict(scene_count=len(scenes), omega=rates['omega'], pi3=rates['pi3'],
                pi3_minus_omega=float((rates['pi3']['accuracy'] or 0) -
                                      (rates['omega']['accuracy'] or 0)),
                paired_scene_bootstrap_95=[float(np.nanquantile(delta, .025)),
                                           float(np.nanquantile(delta, .975))])


def evaluate():
    """Read no labels until both model manifests are complete."""
    plan = load_plan()
    manifests = {}
    for kind in ('omega', 'pi3'):
        manifest = json.loads((RUNTIME / kind / 'manifest.json').read_text())
        if manifest['status'] != 'COMPLETE_PAIRED_GEOMETRY_INFERENCE' or manifest['plan_sha256'] != sha256(PLAN):
            raise ValueError('inference artifacts are not frozen for ' + kind)
        manifests[kind] = manifest
    detector = json.loads((RUNTIME / 'detector_cache.json').read_text())
    if detector['plan_sha256'] != sha256(PLAN):
        raise ValueError('detector plan hash mismatch')
    data = json.loads((DATA / 'annotations/train.json').read_text())
    gt = collections.defaultdict(list)
    for ann in data['annotations']:
        gt[int(ann['image_id'])].append(ann)
    detector_rows = {r['pair_id']: r for r in detector['rows']}
    result = dict(status='COMPLETE_PI3_OMEGA_COMPARISON', plan_sha256=sha256(PLAN),
                  annotation_sha256=sha256(DATA / 'annotations/train.json'),
                  labels_read_after_inference=True, models={})
    for kind in ('omega', 'pi3'):
        manifest = manifests[kind]
        rank = {name: collections.Counter() for name in ('world_chamfer', 'world_centroid')}
        by_scene = collections.defaultdict(lambda: {name: collections.Counter() for name in rank})
        proxy_rows = []
        frozen_rows = {r['pair_id']: r for r in manifest['rows']}
        duplicate_rows = 0
        for pair in plan['samples']:
            drow = detector_rows[pair['pair_id']]
            labels = []
            for item, boxes in zip(pair['images'], drow['boxes_original']):
                labs, dup = _labels_for_boxes(boxes, item, gt)
                labels.append(labs)
                duplicate_rows += dup
            archive_path = RUNTIME / kind / (pair['pair_id'] + '.npz')
            with np.load(archive_path) as archive:
                points = archive['points']
            scores = world_box_scores(pair, drow['boxes_original'], points)
            row_rank = _rank_metrics(scores, labels)
            for name in rank:
                rank[name].update(row_rank[name])
                by_scene[pair['scene']][name].update(row_rank[name])
            # SIFT was already computed outside the online timer and frozen
            # in each manifest row; do not repeat the expensive CPU pass.
            proxy = frozen_rows[pair['pair_id']]['proxy']
            proxy_rows.append(dict(pair_id=pair['pair_id'], scene=pair['scene'], **proxy,
                                   valid_points=int(scores['valid_points'])))
        proxy_valid = [r['median_normalized_error'] for r in proxy_rows if r['median_normalized_error'] is not None]
        proxy_p90 = [r['p90_normalized_error'] for r in proxy_rows if r['p90_normalized_error'] is not None]
        detector_by_pair = {r['pair_id']: r for r in detector['rows']}
        paired_composed = [detector_by_pair[r['pair_id']]['pair_seconds'] +
                           r['serial_rgb_to_assignment_seconds']
                           for r in manifest['rows']]
        composed_stats = summary(paired_composed)
        model_e2e = composed_stats['mean_ms'] / 1000.0 - detector['pair_serial']['mean_ms'] / 1000.0
        detector_mean = detector['pair_serial']['mean_ms'] / 1000.0
        result['models'][kind] = dict(
            weight_sha256=manifest['weight_sha256'], gpu=manifest['gpu'],
            model_latency=manifest['model'], module_e2e=manifest['serial_rgb_to_assignment'],
            model_pair_fps=manifest['model_pair_fps'], module_e2e_pair_fps=manifest['module_e2e_pair_fps'],
            composed_detector_plus_module=dict(
                detector_pair_serial_ms=detector['pair_serial']['mean_ms'],
                latency=composed_stats,
                mean_ms=float(composed_stats['mean_ms']),
                pair_timestamp_fps=float(1000.0 / composed_stats['mean_ms']),
                camera_image_fps=float(2000.0 / composed_stats['mean_ms']),
                scope='serial measured live GMT detector + model read/preprocess/forward/world-point association; no GMT tracker'),
            reconstruction_proxy=dict(
                pairs=len(proxy_rows), pairs_with_valid_sift=int(len(proxy_valid)),
                median_of_pair_medians=float(np.median(proxy_valid)) if proxy_valid else None,
                p90_of_pair_medians=float(np.quantile(proxy_valid, .9)) if proxy_valid else None,
                median_of_pair_p90=float(np.median(proxy_p90)) if proxy_p90 else None,
                interpretation='relative cross-view world-point agreement; lower is better; no 3D GT'),
            association=dict(scores={k: dict(v) for k, v in rank.items()},
                             by_scene={str(s): {k: dict(v) for k, v in d.items()} for s, d in by_scene.items()},
                             duplicate_gt_instance_rows_excluded=int(duplicate_rows),
                             interpretation='known-candidate ranking only; eligible directions are correlated and not an open-world detector metric'),
            rows=proxy_rows)
    result['comparison'] = {score: _cluster_bootstrap(result['models'], score)
                            for score in ('world_chamfer', 'world_centroid')}
    result['shared_detector'] = dict(pair_serial=detector['pair_serial'], model=detector['detector_model'],
                                     cache=str(RUNTIME / 'detector_cache.json'),
                                     tracker_or_gmt_main_flow_called=False)
    save_json(ROOT / 'results/pi3_omega_pair_comparison_v1.json', result)
    save_json(RUNTIME / 'comparison_result.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'detect', 'infer', 'evaluate'])
    parser.add_argument('--model', choices=['omega', 'pi3'])
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare()
    elif args.command == 'detect':
        detect()
    elif args.command == 'infer':
        if args.model is None:
            raise SystemExit('--model required for infer')
        infer(args.model)
    else:
        evaluate()
