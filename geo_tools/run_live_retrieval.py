"""Read images -> frozen GMT -> causal clip index -> target query, without caches."""
import argparse
import collections
import resource
import time
import numpy as np
import torch
from common import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-ids', nargs='+', type=int, default=[18])
    parser.add_argument('--tag', required=True)
    parser.add_argument('--max-frames', type=int)
    parser.add_argument('--backbone', choices=['none', 'gmt', 'dinov2', 'vggt'], default='none')
    parser.add_argument('--interval', type=int, default=10)
    parser.add_argument('--total-call-budget', type=int, default=-1)
    parser.add_argument('--query-interval', type=int, default=20)
    parser.add_argument('--index-mib', type=int, default=100)
    parser.add_argument('--diagnose-scoring', action='store_true')
    tracking_group = parser.add_mutually_exclusive_group()
    tracking_group.add_argument('--tracking-native-prior', action='store_true',
                        help='explicit head-free fixed-prior tracking control; retrieval features remain separately selected')
    tracking_group.add_argument('--tracking-vggt-dense', action='store_true',
                               help='frozen primary-seed dense B7 tracking candidate; retrieval ranking remains separately selected')
    args = parser.parse_args()
    # Frozen development selection; no TEST and no GT-driven query selection.
    allowed = {1, 10, 11, 18, 19}
    if len(set(args.video_ids)) != len(args.video_ids) or not set(args.video_ids) <= allowed:
        raise ValueError('choose distinct strict development scenes')
    if min(args.interval, args.query_interval, args.index_mib) < 1:
        raise ValueError('positive intervals and memory limit required')
    out = RUNTIME/'retrieval'/args.tag
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    sys.path[:0] = [str(RUNTIME/'sources/vggt'), str(RUNTIME/'sources/dinov2')]
    from gtr.geometry.isolated_stream import IsolatedCausalStream
    from gtr.geometry.runtime import GeometryRuntime
    from gtr.geometry.retrieval_index import CausalClipIndex
    from gtr.geometry.memory_audit import retained_payload_bytes
    from gtr.evaluation.mot_evaluation import custom_instances_to_coco_json
    cfg, model = load_gmt()
    seed_all(20261014)
    tracking_options = None
    tracking_factory = None
    tracking_head_path = None
    if args.tracking_native_prior or args.tracking_vggt_dense:
        tracking_options = dict(ENABLED=True, FEATURE_ENABLED=True, GEOMETRY_ENABLED=False,
            FEATURE_WEIGHT=2., GEOMETRY_WEIGHT=0., FEATURE_BACKBONE='gmt', FEATURE_LAYERS=[-1],
            SYMMETRIZE_VIEW_ORDER=False,
            PROJECTION_DIM=0, POOLING='reid', HEAD_PATH='', MAX_RESIDUAL=2.,
            SPARSE_UPDATE_ENABLED=False, ASYNC_WORKER=False, REUSE_TARGET_FEATURES=False,
            CALIBRATION_PATH=str(ROOT/'manifests/calibration_gmt_reid_absolute_train_v1.json'))
        if args.tracking_vggt_dense:
            tracking_head_path = RUNTIME/'heads/low_rank_three_backbones_train_v1/vggt_seed20261014.pt'
            tracking_options.update(GEOMETRY_ENABLED=True,GEOMETRY_WEIGHT=.5,FEATURE_BACKBONE='vggt',
                FEATURE_LAYERS=[17],POOLING='roi_align',SYMMETRIZE_VIEW_ORDER=True,HEAD_PATH=str(tracking_head_path),
                VGGT_CHECKPOINT=str(RUNTIME/'weights/vggt_1b.pt'),GEOMETRY_METHOD='vggt_camera',
                SPARSE_UPDATE_ENABLED=True,UPDATE_POLICY='fixed',UPDATE_INTERVAL=1,MIN_UPDATE_INTERVAL=1,
                COMPUTE_BUDGET=200,BUDGET_WINDOW=100,TOTAL_CALL_BUDGET=-1)
        def tracking_factory(base_geo):
            geo = base_geo.clone()
            for key, value in tracking_options.items():
                setattr(geo, key, value)
            runtime = GeometryRuntime(geo)
            if args.tracking_native_prior:
                from gtr.geometry.native_prior import install
                install(runtime)
            return runtime
    streams = {video: IsolatedCausalStream(model, runtime_factory=tracking_factory) for video in args.video_ids}
    # Read image metadata only here. Annotation identities/boxes are never read
    # by inference, publication, query selection or ranking.
    data = train_data()
    sequences = {}
    for video in args.video_ids:
        groups = collections.defaultdict(list)
        for record in video_records(data, video, args.max_frames):
            groups[record['frame_id']].append(record)
        sequences[video] = sorted(groups.items())
    extractors = {}
    head_path = None
    if args.backbone != 'none':
        head_path = RUNTIME/'heads/low_rank_three_backbones_train_v1'/f'{args.backbone}_seed20261014.pt'
        geo = cfg.MODEL.GEO.clone()
        options = dict(ENABLED=True, FEATURE_ENABLED=True, GEOMETRY_ENABLED=False, FEATURE_WEIGHT=2., GEOMETRY_WEIGHT=0.,
                       FEATURE_BACKBONE=args.backbone, FEATURE_LAYERS=[17] if args.backbone == 'vggt' else [-1],
                       PROJECTION_DIM=0 if args.backbone != 'dinov2' else 256, POOLING='reid' if args.backbone == 'gmt' else 'roi_align',
                       SYMMETRIZE_VIEW_ORDER=args.backbone == 'vggt', HEAD_PATH=str(head_path), MAX_RESIDUAL=2.,
                       VGGT_CHECKPOINT=str(RUNTIME/'weights/vggt_1b.pt'), DINOV2_CHECKPOINT=str(RUNTIME/'weights/dinov2_vitl14_reg4.pth'),
                       SPARSE_UPDATE_ENABLED=True, UPDATE_POLICY='fixed', UPDATE_INTERVAL=args.interval, MIN_UPDATE_INTERVAL=1,
                       COMPUTE_BUDGET=200, BUDGET_WINDOW=100, TOTAL_CALL_BUDGET=args.total_call_budget)
        for key, value in options.items():
            setattr(geo, key, value)
        extractors = {video: GeometryRuntime(geo.clone()) for video in args.video_ids}
    index = CausalClipIndex(max_bytes=args.index_mib << 20)
    predictions = {video: [] for video in args.video_ids}
    query_latencies, round_latencies = [], []
    diagnostic_latencies = collections.defaultdict(list)
    images, published, requests, augmented_queries = 0, 0, 0, 0
    shared_encoder, shared_head = None, None
    shared_tracking_encoder, shared_tracking_head = None, None
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with (out/'queries.jsonl').open('w') as query_file, (out/'gallery_snapshots.jsonl').open('w') as gallery_file:
        for round_id in range(max(map(len, sequences.values()))):
            torch.cuda.synchronize()
            round_start = time.perf_counter()
            index.advance(round_id)
            current_targets, current_images = [], []
            for video in args.video_ids:
                if round_id >= len(sequences[video]):
                    continue
                timestamp, records = sequences[video][round_id]
                current_images.extend(record['image_id'] for record in records)
                mapped = map_records(cfg, records)
                if args.tracking_vggt_dense:
                    streams[video].runtime.vggt, streams[video].runtime.head = shared_tracking_encoder, shared_tracking_head
                outputs = streams[video].step(mapped)
                if args.tracking_vggt_dense:
                    shared_tracking_encoder, shared_tracking_head = streams[video].runtime.vggt, streams[video].runtime.head
                instances = [row['instances'] for row in outputs]
                if video in extractors:
                    extractor = extractors[video]
                    extractor.vggt, extractor.head = shared_encoder, shared_head
                    extractor.observe(instances, records, round_id)
                    shared_encoder, shared_head = extractor.vggt, extractor.head
                for record, instance in zip(records, instances):
                    cpu = instance.to('cpu')
                    predictions[video].extend(custom_instances_to_coco_json(cpu, record['image_id']))
                    boxes = cpu.pred_boxes.tensor.numpy()
                    gmt = cpu.reid_features.numpy()
                    for position in range(len(cpu)):
                        # Tracking metadata can contain a zero-column prior
                        # field. Only the separate retrieval extractor supplies
                        # ranking vectors; native ranking must ignore it.
                        extra = cpu.geo_vector[position].numpy() if video in extractors and cpu.has('geo_valid') and bool(cpu.geo_valid[position]) else None
                        target = dict(scene=video, camera=int(record['view_id']), track=int(cpu.track_ids[position]), round_id=round_id,
                                      image_id=int(record['image_id']), frame_id=int(timestamp), bbox=boxes[position].tolist(),
                                      gmt=gmt[position], extra=extra)
                        index.publish(**target)
                        current_targets.append(target)
                        published += 1
                    images += 1
            # Query only after all currently arrived streams publish; no future
            # source frame or later extraction can update this snapshot.
            if round_id % args.query_interval == 0:
                gallery_file.write(json.dumps(dict(round_id=round_id, query_image_ids=current_images, entries=index.snapshot()))+'\n')
                scale, bias = (float(shared_head.scale.detach().cpu()), float(shared_head.bias.detach().cpu())) if shared_head is not None else (1., 0.)
                for target in current_targets:
                    query_start = time.perf_counter()
                    kwargs = dict(scene=target['scene'], camera=target['camera'], round_id=round_id, gmt=target['gmt'], extra=target['extra'])
                    if shared_head is not None:
                        kwargs.update(feature_weight=2., scale=scale, bias=bias)
                    response = index.query(**kwargs)
                    public = {key: value for key, value in target.items() if key not in {'gmt', 'extra'}}
                    # Count response serialization in query latency, file I/O in
                    # total input-to-publication/query round latency.
                    payload = json.dumps(dict(query=public, response=response))
                    query_latencies.append(time.perf_counter()-query_start)
                    if args.diagnose_scoring:
                        diagnostic_responses = {}
                        for scoring in ['native', 'native_encoded', 'learned_cosine', 'equal_cosine_encoded']:
                            scoring_start = time.perf_counter()
                            diagnostic_responses[scoring] = index.query(**kwargs, scoring=scoring)
                            json.dumps(diagnostic_responses[scoring])
                            diagnostic_latencies[scoring].append(time.perf_counter()-scoring_start)
                        payload = json.dumps(dict(query=public, response=response, diagnostic_responses=diagnostic_responses))
                    query_file.write(payload+'\n')
                    requests += 1
                    augmented_queries += target['extra'] is not None
            torch.cuda.synchronize()
            round_latencies.append(time.perf_counter()-round_start)
            if round_id % 100 == 0:
                print(json.dumps(dict(round_id=round_id, images=images, requests=requests, index_payload_bytes=index.bytes)), flush=True)
    seconds = time.perf_counter()-started
    for video, rows in predictions.items():
        save_json(out/f'video_{video:02d}.json', rows)
    def percentiles(values):
        return dict(p50=float(np.percentile(values, 50)), p95=float(np.percentile(values, 95))) if values else None
    report = provenance()
    if args.tracking_native_prior and any(stream.runtime.vggt is not None or stream.runtime.head is not None for stream in streams.values()):
        raise ValueError('head-free tracking control unexpectedly loaded an extra model')
    report.update(video_ids=args.video_ids, camera_streams=2*len(args.video_ids), max_frames=args.max_frames, causal_online=True, full_video=args.max_frames is None,
                  setup_time_included=False, first_foundation_load_included=True, seconds=seconds, images=images, camera_image_fps=images/seconds,
                  synchronized_round_fps=len(round_latencies)/seconds, round_latency_seconds=percentiles(round_latencies), query_latency_seconds=percentiles(query_latencies),
                  requests=requests, augmented_queries=augmented_queries, published_targets=published, used_cache=False,
                  speed_eligible=not args.diagnose_scoring,diagnostic_scoring=args.diagnose_scoring,
                  scoring_mode_latency_seconds={mode:percentiles(values) for mode,values in diagnostic_latencies.items()},
                  scoring_diagnostic_scope='same actual query/gallery snapshot; native and learned on identical encoded candidates and identical source observations (native_encoded uses GMT vector saved at the extra-feature observation); no GT candidate selection; absent features count as failed queries on the full-gallery denominator; archived identity vectors describe their past source observations and remain usable until clip TTL, without propagated geometry claims',
                  scoring_timing_scope='when diagnostics enabled, complete producer runs five ranking rules serially; its full throughput is diagnostic cost, not single-ranker production throughput; per-mode scoring/response serialization measured separately',
                  protocol=('causal GMT with explicit head-free fixed-prior tracking control' if args.tracking_native_prior else 'causal dense B7 primary-seed VGGT tracking candidate' if args.tracking_vggt_dense else 'causal GMT OFF tracking')+'; independent per-scene banks, geometry runtimes and RNG; online optional TRAIN head features used only for retrieval',
                  tracking_method='native_fixed_prior' if args.tracking_native_prior else 'vggt_dense' if args.tracking_vggt_dense else 'gmt_off',
                  tracking_options=tracking_options,
                  tracking_head_checkpoint_sha256=sha256(tracking_head_path) if tracking_head_path else None,
                  tracking_runtime_stats={str(video):dict(updates=stream.runtime.calls,vggt_calls=stream.runtime.vggt_calls,
                      failures=stream.runtime.failed,valid_geometry_calls=stream.runtime.geometry_valid_calls,
                      seconds=stream.runtime.seconds,**getattr(stream.runtime,'native_prior_stats',{})) for video,stream in streams.items()},
                  temporal_scope='within-scene cameras synchronized by dataset frame_id; unrelated scenes interleaved by ordinal replay round, no physical cross-scene synchronization claim',
                  query_scope='every current detector target at fixed query interval; all other camera clips including unrelated scene distractors; no GT selection',
                  latency_scope='local in-process query scoring and response serialization; no HTTP/network latency claim; full rounds charge reads, detector, GTA, extraction, CPU publication, queries and log I/O',
                  backbone=args.backbone, interval=args.interval, total_call_budget_per_scene=args.total_call_budget, query_interval=args.query_interval,
                  head_checkpoint_sha256=sha256(head_path) if head_path else None,
                  index=dict(clip_groups=index.clip_groups, ttl_groups=index.ttl_groups, feature_age=index.feature_age, max_payload_bytes=index.max_bytes,
                             peak_payload_bytes=index.peak_bytes, evictions=index.evictions, entries=len(index.entries)),
                  peak_host_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                  retained_tracker_payload_bytes={str(video):retained_payload_bytes([stream.tracker.instances, stream.tracker.reids, stream.state[:3]]) for video, stream in streams.items()},
                  memory_scope='index cap bounds serialized metadata plus vector payload; original GMT history/memory remains unchanged and is audited separately; RSS and CUDA peaks include weights/allocator, not just cache',
                  peak_gpu_bytes=torch.cuda.max_memory_allocated(), peak_reserved_gpu_bytes=torch.cuda.max_memory_reserved(),
                  extractors={str(video): dict(updates=runtime.calls, vggt_calls=runtime.vggt_calls, dinov2_calls=runtime.dinov2_calls, failures=runtime.failed, seconds=runtime.seconds) for video, runtime in extractors.items()},
                  split_manifest_sha256=sha256(ROOT/'manifests/geo_split_v1.json'), artifacts={path.name:sha256(path) for path in sorted(out.iterdir()) if path.is_file()})
    save_json(out/'manifest.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
