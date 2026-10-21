"""Bounded two-view depth/pose feasibility and latency probe, separate from GMT.

The plan is written before inference. Inputs are synchronized original TRAIN
frames and public frozen detector boxes. Labels are used only by ``evaluate``.
This is a development probe; it cannot reopen any previous GMT admission gate.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = Path('/data/DATASETS/TRACKING/JDE/VisionTrack')
SOURCE = Path('/data1/liuyeqiang/WWW_omega_runtime_20261016/official_vggt_omega')
CHECKPOINT = Path('/data1/liuyeqiang/WWW_omega_runtime_20261016/weights/vggt_omega_1b_512.pt')
AUDIT = Path('/data1/liuyeqiang/WWW_omega_fulltrain_runtime_20261016/full_train_logit_audit_v1')
RUNTIME = Path('/data1/liuyeqiang/WWW_omega_pair_geometry_runtime_20261021')
PLAN = ROOT / 'manifests/omega_pair_geometry_fps_probe_v1.json'
CHECKPOINT_SHA = 'c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def stats(values):
    values = np.asarray(values, dtype=float)
    return dict(n=len(values), mean_ms=float(values.mean() * 1000),
                p50_ms=float(np.median(values) * 1000),
                p95_ms=float(np.quantile(values, .95) * 1000))


def prepare():
    if PLAN.exists():
        raise FileExistsError(PLAN)
    source_protocol = json.loads((ROOT / 'manifests/omega_full24_training_protocol_v1.json').read_text())
    scenes = source_protocol['scene_split']['fit_video_ids']
    metadata = json.loads((DATA / 'annotations/train.json').read_text())['images']
    samples = []
    for scene in scenes:
        frames = collections.defaultdict(list)
        for image in metadata:
            if int(image['video_id']) == scene:
                frames[int(image['frame_id'])].append(image)
        times = sorted(frames)
        selected = [times[round((len(times) - 1) * quantile)] for quantile in (.1, .5, .9)]
        wanted = {int(i['id']) for frame in selected for i in frames[frame]}
        boxes = collections.defaultdict(list)
        detector_path = AUDIT / f'scene{scene}_full_logit_v1/detector_provenance.jsonl.gz'
        with gzip.open(detector_path, 'rt') as stream:
            for line in stream:
                row = json.loads(line)
                if int(row['source'][0]) in wanted:
                    boxes[int(row['source'][0])].append(dict(source=row['source'], bbox=row['detection']['bbox']))
        for frame in selected:
            images = []
            for row in sorted(frames[frame], key=lambda r: int(r['view_id'])):
                path = DATA / 'train' / row['file_name']
                images.append(dict(image_id=int(row['id']), camera_id=int(row['view_id']), path=str(path),
                                   width=int(row['width']), height=int(row['height']), sha256=sha256(path),
                                   detections=boxes[int(row['id'])]))
            if len(images) != 2:
                raise ValueError('synchronized two-view pair required')
            samples.append(dict(pair_id=f'scene{scene}_frame{frame}', scene=scene, frame=frame, images=images))
    plan = dict(status='LOCKED_BEFORE_NEW_INFERENCE', seed=20261021, fit_scene_ids=scenes,
                sampling='18 prior fit scenes only; 10%, 50%, 90% time quantiles, 54 pairs; no selection by outputs',
                purpose='dual-view reconstruction plus 2D detections lifted into common gauge; FPS feasibility only',
                configs=[dict(name='max512', mode='max_size', resolution=512, gpu=2),
                         dict(name='balanced512', mode='balanced', resolution=512, gpu=3),
                         dict(name='max384', mode='max_size', resolution=384, gpu=4),
                         dict(name='max256', mode='max_size', resolution=256, gpu=5)],
                checkpoint=str(CHECKPOINT), checkpoint_sha256=CHECKPOINT_SHA,
                source=str(SOURCE), source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=SOURCE, text=True).strip(),
                warmup_calls=5, component_repeats=10, max512_batch4_repeats=10,
                precision='official native fp16 aggregator AMP on V100; original fp32 camera/depth head guards',
                timing='CUDA synchronized; model resident; per-pair model+pose decoding and serial read/preprocess/H2D/model timings; no detector or GMT included',
                geometry_scores=['symmetric projected bbox grid support with relative depth consistency 0.10',
                                 'negative common-gauge footpoint distance normalized by pair median depth'],
                grid=dict(nx=5, ny=7, x_fraction=[.2, .8], y_fraction=[.2, .9]),
                labels='schema inspected before inference; numerical label comparisons only posthoc original TRAIN; duplicate GT instance_id in either view excluded; mutual-best detector-GT IoU >= 0.5; only queries with positive and negative candidates',
                abstention='projected support <= 0 or tied best score; missing/nonfinite depth or cameras',
                quality='no tuning; all 54 pairs and four configs retained; no proof of identity or tracking improvement',
                old_gates='remain closed; no GMT integration, no TEST, no 416 reproduction',
                resources=dict(physical_gpus=[2, 3, 4, 5], max_inference_process_seconds_sum=1800, cache_limit_bytes=2147483648),
                gmt_baseline=dict(scenes=[2, 8, 17], continuous_prefix_groups=100, warmup_groups=10,
                                  scope='optional short timing probe including current-group reading/detection/causal tracking; no full-video FPS claim'),
                samples=samples)
    if sha256(CHECKPOINT) != CHECKPOINT_SHA:
        raise ValueError('checkpoint hash mismatch')
    save(PLAN, plan)
    print(json.dumps(dict(plan=str(PLAN), pairs=len(samples), sha256=sha256(PLAN))))


def world_points(uv, depth, extrinsics, intrinsics):
    """Backproject camera depth with OpenCV camera-from-world extrinsics."""
    uv = np.asarray(uv, dtype=float)
    rays = np.column_stack([uv, np.ones(len(uv))]) @ np.linalg.inv(intrinsics).T
    camera = rays * np.asarray(depth)[:, None]
    return (camera - extrinsics[:, 3]) @ extrinsics[:, :3]


def project_points(world, extrinsics, intrinsics):
    camera = np.asarray(world) @ extrinsics[:, :3].T + extrinsics[:, 3]
    image = camera @ intrinsics.T
    return image[:, :2] / np.maximum(image[:, 2:3], 1e-12), camera[:, 2]


def infer(config_name, execution='eager'):
    import torch
    sys.path.insert(0, str(SOURCE))
    from vggt_omega.models import VGGTOmega
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    from vggt_omega.utils.pose_enc import encoding_to_camera

    plan = json.loads(PLAN.read_text())
    config = next(c for c in plan['configs'] if c['name'] == config_name)
    output = RUNTIME / ('native_fp16' if execution == 'eager' else 'cuda_graph_fp16') / config_name
    output.mkdir(parents=True, exist_ok=False)
    save(output / 'status.json', dict(status='RUNNING', plan_sha256=sha256(PLAN), config=config))
    torch.set_num_threads(2)
    torch.manual_seed(plan['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    process_started = time.perf_counter()
    if sha256(CHECKPOINT) != CHECKPOINT_SHA:
        raise ValueError('checkpoint hash mismatch')
    # PyTorch 2.6 reports BF16 emulation as supported on V100 by default.
    # The official automatic selector consequently chooses BF16 on this host.
    # Explicit outer FP16 implements the locked native precision contract;
    # VGGTOmega.forward retains its own disabled-autocast FP32 head guards.
    model = VGGTOmega(autocast=False).eval().to('cuda').requires_grad_(False)
    state = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
    state = state.get('model', state.get('state_dict', state))
    keys = len(state)
    model.load_state_dict(state, strict=True)
    del state
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - process_started

    def preprocess(pair):
        return load_and_preprocess_images([i['path'] for i in pair['images']], mode=config['mode'],
                                          image_resolution=config['resolution'], patch_size=16).unsqueeze(0)

    def forward(images):
        with torch.autocast('cuda', dtype=torch.float16):
            predictions = model(images)
        extrinsics, intrinsics = encoding_to_camera(predictions['pose_enc'], images.shape[-2:])
        return predictions, extrinsics, intrinsics

    images = preprocess(plan['samples'][0]).to('cuda')
    cold_start = time.perf_counter()
    with torch.inference_mode():
        for _ in range(plan['warmup_calls']):
            result = forward(images)
        torch.cuda.synchronize()
        warmup_seconds = time.perf_counter() - cold_start
        del result
        graph_setup_seconds = None
        if execution == 'cuda_graph':
            setup_started = time.perf_counter()
            reference_forward = forward
            static_images = images.clone()
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    graph_result = reference_forward(static_images)
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                graph_result = reference_forward(static_images)
            torch.cuda.synchronize()

            def graph_forward(current):
                if current.shape != static_images.shape:
                    raise ValueError('CUDA Graph requires the locked fixed shape')
                static_images.copy_(current)
                graph.replay()
                return graph_result

            forward = graph_forward
            graph_setup_seconds = time.perf_counter() - setup_started
        torch.cuda.reset_peak_memory_stats()
        rows = []
        for pair in plan['samples']:
            for image in pair['images']:
                if sha256(image['path']) != image['sha256']:
                    raise ValueError('input image changed')
            torch.cuda.synchronize()
            serial_start = time.perf_counter()
            cpu_images = preprocess(pair)
            images = cpu_images.to('cuda')
            torch.cuda.synchronize()
            pre_seconds = time.perf_counter() - serial_start
            start = time.perf_counter()
            predictions, extrinsics, intrinsics = forward(images)
            torch.cuda.synchronize()
            compute_seconds = time.perf_counter() - start
            serial_seconds = time.perf_counter() - serial_start
            arrays = dict(depth=predictions['depth'][0, :, :, :, 0].float().cpu().numpy(),
                          confidence=predictions['depth_conf'][0].float().cpu().numpy(),
                          extrinsics=extrinsics[0].cpu().numpy(), intrinsics=intrinsics[0].cpu().numpy())
            finite = all(np.isfinite(a).all() for a in arrays.values())
            parity = None
            if execution == 'cuda_graph':
                reference_path = RUNTIME / 'native_fp16' / config_name / (pair['pair_id'] + '.npz')
                with np.load(reference_path) as reference:
                    parity = {k: float(np.max(np.abs(a - reference[k]) / (1 + np.abs(reference[k])))) for k, a in arrays.items()}
                if max(parity.values()) > 1e-4:
                    raise ValueError(f'CUDA Graph changed outputs beyond locked tolerance: {parity}')
            archive = output / (pair['pair_id'] + '.npz')
            np.savez(archive, **arrays)
            rows.append(dict(pair_id=pair['pair_id'], finite=finite, shape=list(images.shape),
                             compute_seconds=compute_seconds, read_preprocess_h2d_seconds=pre_seconds,
                             serial_read_to_geometry_seconds=serial_seconds, archive_sha256=sha256(archive),
                             eager_parity_max_normalized_error=parity))
            del predictions, extrinsics, intrinsics, arrays

        component = {}
        for part in (('aggregator', 'camera', 'depth') if execution == 'eager' else ()):
            times = []
            if part != 'aggregator':
                with torch.autocast('cuda', dtype=torch.float16):
                    tokens, patch_start = model.aggregator(images)
            for _ in range(plan['component_repeats']):
                torch.cuda.synchronize()
                start = time.perf_counter()
                if part == 'aggregator':
                    with torch.autocast('cuda', dtype=torch.float16):
                        value = model.aggregator(images)
                elif part == 'camera':
                    with torch.autocast('cuda', enabled=False):
                        value = model.camera_head(tokens, patch_token_start=patch_start)
                else:
                    with torch.autocast('cuda', enabled=False):
                        value = model.dense_head(tokens, images=images, patch_token_start=patch_start)
                torch.cuda.synchronize()
                times.append(time.perf_counter() - start)
                del value
            component[part] = stats(times)
            if part != 'aggregator':
                del tokens
        batch4 = None
        if config_name == 'max512' and execution == 'eager':
            batch_images = images.expand(4, -1, -1, -1, -1).contiguous()
            for _ in range(2):
                result = forward(batch_images)
            torch.cuda.synchronize()
            del result
            times = []
            for _ in range(plan['max512_batch4_repeats']):
                torch.cuda.synchronize()
                start = time.perf_counter()
                result = forward(batch_images)
                torch.cuda.synchronize()
                times.append(time.perf_counter() - start)
                del result
            batch4 = dict(batch_latency=stats(times), pairs_per_second=4 / float(np.mean(times)),
                          scope='four independent repeated physical pairs per batch, model-only throughput; buffering latency excluded')

    report = dict(status='COMPLETE_DUAL_VIEW_RECONSTRUCTION_PROBE', plan_sha256=sha256(PLAN), config=config,
                  checkpoint_sha256=CHECKPOINT_SHA, strict_loaded_keys=keys,
                  source_commit=plan['source_commit'], script_sha256=sha256(__file__),
                  gpu=torch.cuda.get_device_name(), cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  torch=torch.__version__, cuda=torch.version.cuda, precision='explicit fp16 aggregator / fp32 heads',
                  execution=execution, graph_setup_seconds=graph_setup_seconds,
                  bf16_reported_supported=torch.cuda.is_bf16_supported(),
                  bf16_native_supported=torch.cuda.is_bf16_supported(including_emulation=False),
                  finite_pairs=sum(r['finite'] for r in rows), total_pairs=len(rows),
                  load_seconds=load_seconds, warmup_seconds=warmup_seconds,
                  compute=stats([r['compute_seconds'] for r in rows]),
                  read_preprocess_h2d=stats([r['read_preprocess_h2d_seconds'] for r in rows]),
                  serial_read_to_geometry=stats([r['serial_read_to_geometry_seconds'] for r in rows]),
                  component=component, batch4=batch4, model_resident=True,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                  process_seconds=time.perf_counter() - process_started, rows=rows,
                  fps_unit='one synchronized two-camera timestamp; no detector/GMT/render/export included',
                  reduced_resolution_512_weights=config['resolution'] != 512, annotations_used=False,
                  tracking_or_identity_improvement='NOT_MEASURED', official_test='NOT_ACCESSED')
    save(output / 'manifest.json', report)
    save(output / 'status.json', dict(status=report['status']))
    print(json.dumps({k: v for k, v in report.items() if k not in ('rows', 'component')}), flush=True)


def sample_depth(depth, uv):
    import cv2
    uv = np.asarray(uv, dtype=np.float32)
    sampled = cv2.remap(depth.astype(np.float32), uv[:, 0].reshape(1, -1), uv[:, 1].reshape(1, -1),
                        cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)[0]
    return sampled


def geometry_scores(pair, arrays):
    """Use only full-image predicted geometry and public detection boxes."""
    depth, E, K = (arrays[k] for k in ('depth', 'extrinsics', 'intrinsics'))
    height, width = depth.shape[-2:]
    boxes = []
    for image in pair['images']:
        b = np.asarray([d['bbox'] for d in image['detections']], dtype=float).reshape(-1, 4)
        b[:, 2:] += b[:, :2]
        b *= np.asarray([width / image['width'], height / image['height']] * 2)
        boxes.append(b)
    if not all(np.isfinite(a).all() for a in (depth, E, K)):
        return dict(projected_support=np.zeros((len(boxes[0]), len(boxes[1]))),
                    foot_3d=np.full((len(boxes[0]), len(boxes[1])), -1e6))
    directed = []
    feet = []
    feet_valid = []
    for view in (0, 1):
        other = 1 - view
        supports, foot_points, foot_valid = [], [], []
        for box in boxes[view]:
            xs = box[0] + np.linspace(.2, .8, 5) * (box[2] - box[0])
            ys = box[1] + np.linspace(.2, .9, 7) * (box[3] - box[1])
            gx, gy = np.meshgrid(xs, ys)
            uv = np.column_stack([gx.ravel(), gy.ravel()])
            z = sample_depth(depth[view], uv)
            valid = (z > 0) & np.isfinite(z) & (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
            world = world_points(uv, z, E[view], K[view])
            projected, other_z = project_points(world, E[other], K[other])
            observed_z = sample_depth(depth[other], projected)
            valid &= (other_z > 0) & (observed_z > 0) & (np.abs(observed_z - other_z) / np.maximum(other_z, 1e-8) <= .10)
            if len(boxes[other]):
                inside = ((projected[:, None, 0] >= boxes[other][None, :, 0]) & (projected[:, None, 0] <= boxes[other][None, :, 2]) &
                          (projected[:, None, 1] >= boxes[other][None, :, 1]) & (projected[:, None, 1] <= boxes[other][None, :, 3]))
                supports.append((inside & valid[:, None]).mean(axis=0))
            else:
                supports.append(np.empty(0))
            foot = np.asarray([[(box[0] + box[2]) * .5, box[1] + .97 * (box[3] - box[1])]])
            foot_z = sample_depth(depth[view], foot)
            foot_points.append(world_points(foot, foot_z, E[view], K[view])[0])
            foot_valid.append(bool(foot_z[0] > 0 and 0 <= foot[0, 0] < width and 0 <= foot[0, 1] < height))
        directed.append(np.asarray(supports).reshape(len(boxes[view]), len(boxes[other])))
        feet.append(np.asarray(foot_points).reshape(-1, 3))
        feet_valid.append(np.asarray(foot_valid, dtype=bool))
    support = np.sqrt(directed[0] * directed[1].T)
    scale = max(float(np.median(depth)), 1e-8)
    distance = np.linalg.norm(feet[0][:, None] - feet[1][None], axis=-1) / scale
    distance[~(feet_valid[0][:, None] & feet_valid[1][None])] = 1e6
    return dict(projected_support=support, foot_3d=-distance)


def bbox_iou(a, b):
    a, b = np.asarray(a, dtype=float).copy(), np.asarray(b, dtype=float).copy()
    a[:, 2:] += a[:, :2]
    b[:, 2:] += b[:, :2]
    lo = np.maximum(a[:, None, :2], b[None, :, :2])
    hi = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.maximum(hi - lo, 0).prod(-1)
    union = (a[:, 2:] - a[:, :2]).prod(-1)[:, None] + (b[:, 2:] - b[:, :2]).prod(-1)[None] - inter
    return inter / np.maximum(union, 1e-8)


def evaluate():
    plan = json.loads(PLAN.read_text())
    reports = {c['name']: json.loads((RUNTIME / 'native_fp16' / c['name'] / 'manifest.json').read_text()) for c in plan['configs']}
    if not all(r['status'] == 'COMPLETE_DUAL_VIEW_RECONSTRUCTION_PROBE' and r['plan_sha256'] == sha256(PLAN) for r in reports.values()):
        raise ValueError('all inference artifacts must be frozen before reading labels')
    gt = collections.defaultdict(list)
    data = json.loads((DATA / 'annotations/train.json').read_text())
    for ann in data['annotations']:
        gt[int(ann['image_id'])].append(ann)
    summaries, raw = {}, []
    for config in plan['configs']:
        name = config['name']
        counters = {score: collections.Counter() for score in ('projected_support', 'foot_3d')}
        by_scene = collections.defaultdict(lambda: {score: collections.Counter() for score in counters})
        latencies = []
        duplicates = 0
        tiny_heights = []
        for pair in plan['samples']:
            labels = []
            for image in pair['images']:
                all_gt = gt[image['image_id']]
                counts = collections.Counter(int(g['instance_id']) for g in all_gt)
                legal = [g for g in all_gt if counts[int(g['instance_id'])] == 1]
                duplicates += sum(counts[int(g['instance_id'])] != 1 for g in all_gt)
                result = [None] * len(image['detections'])
                if legal and image['detections']:
                    iou = bbox_iou([d['bbox'] for d in image['detections']], [g['bbox'] for g in legal])
                    for i, j in enumerate(iou.argmax(axis=1)):
                        if iou[i, j] >= .5 and int(iou[:, j].argmax()) == i:
                            result[i] = int(legal[j]['instance_id'])
                labels.append(result)
            path = RUNTIME / 'native_fp16' / name / (pair['pair_id'] + '.npz')
            with np.load(path) as archive:
                arrays = {k: archive[k] for k in archive.files}
            start = time.perf_counter()
            scores = geometry_scores(pair, arrays)
            latencies.append(time.perf_counter() - start)
            for image in pair['images']:
                tiny_heights.extend(d['bbox'][3] * arrays['depth'].shape[-2] / image['height'] for d in image['detections'])
            row_record = dict(pair_id=pair['pair_id'], scene=pair['scene'], labels=labels,
                              score_matrices={k: v.tolist() for k, v in scores.items()})
            raw.append(dict(config=name, **row_record))
            for view in (0, 1):
                other = 1 - view
                known = np.asarray([l is not None for l in labels[other]], dtype=bool)
                for query, identity in enumerate(labels[view]):
                    if identity is None:
                        continue
                    positives = np.asarray([l == identity for l in labels[other]], dtype=bool)[known]
                    if not positives.any() or positives.all():
                        continue
                    for score, matrix in scores.items():
                        values = (matrix if view == 0 else matrix.T)[query, known]
                        best = float(np.max(values))
                        winners = np.flatnonzero(np.isclose(values, best, rtol=0, atol=1e-8))
                        status = 'abstain' if (len(winners) != 1 or best <= -1e5 or (score == 'projected_support' and best <= 0)) else 'correct' if positives[winners[0]] else 'wrong'
                        for count in (counters[score], by_scene[pair['scene']][score]):
                            count['eligible'] += 1
                            count[status] += 1
        summaries[name] = dict(scores={k: dict(v) for k, v in counters.items()},
                               by_scene={str(k): {s: dict(c) for s, c in v.items()} for k, v in by_scene.items()},
                               raw_duplicate_gt_rows_excluded=duplicates,
                               cpu_geometry_all_candidate_latency=stats(latencies),
                               detector_height_network_pixels=dict(median=float(np.median(tiny_heights)), p10=float(np.quantile(tiny_heights, .1))),
                               timing={k: reports[name][k] for k in ('compute', 'serial_read_to_geometry', 'component', 'batch4', 'load_seconds', 'peak_allocated_bytes')})
    save(ROOT / 'results/omega_pair_geometry_fps_probe_v1.json', dict(status='COMPLETE_BOUNDED_DEVELOPMENT_PROBE', plan_sha256=sha256(PLAN),
         sampling=plan['sampling'], annotation_sha256=sha256(DATA / 'annotations/train.json'),
         interpretation='ranking only; no geometry reliability certification, no assignment/threshold or full tracking benefit measured; old GMT gates stay closed',
         configs=summaries, raw_scores=str(RUNTIME / 'posthoc_score_matrices.json'), test_accessed=False))
    save(RUNTIME / 'posthoc_score_matrices.json', raw)
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'infer', 'evaluate'])
    parser.add_argument('--config', choices=['max512', 'balanced512', 'max384', 'max256'])
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare()
    elif args.command == 'infer':
        infer(args.config)
    else:
        evaluate()
