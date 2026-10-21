"""Run one complete causal GMT audit under the locked Ω-FullTrain protocol.

This file deliberately leaves the earlier twelve-scene TrackVGGT audit
untouched.  It reuses only the pure labeling/summarization functions and
executes a fresh, unmodified CausalGMT stream for the twelve scenes that were
not in that prior association audit.  Ground truth is opened after inference
and the independent OFF-40 prefix check.
"""
import argparse
import collections
import gzip
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

from common import *
from audit_full_train_identity_events import original_truth, summarize

PROTOCOL = ROOT / 'manifests/omega_full24_training_protocol_v1.json'
PRIOR_AUDIT_ROOT = Path('/data1/liuyeqiang/WWW_trackvggt_runtime/full_train_audit_v1')
PRIOR_IDS = {4, 5, 6, 7, 8, 9, 12, 13, 17, 20, 21, 24}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-id', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261014)
    parser.add_argument('--max-retained-events', type=int, default=1000)
    parser.add_argument('--precheck-groups', type=int, default=40)
    parser.add_argument('--head-checkpoint', type=Path)
    parser.add_argument('--feature-root', type=Path, default=Path('/data1/liuyeqiang/WWW_omega_fulltrain_runtime_20261016/omega512_features_v1'))
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text())
    all_ids = set(protocol['dataset']['video_ids'])
    if args.video_id not in all_ids:
        raise ValueError('video id is outside the locked 24-scene TRAIN protocol')
    if args.seed != protocol['stages']['A']['seeds'][0] or not 0 <= args.max_retained_events <= 2000:
        raise ValueError('fixed seed/retention required')
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    data = train_data()
    full_images = video_records(data, args.video_id)
    images = full_images
    groups = collections.defaultdict(list)
    for image in images:
        groups[image['frame_id']].append(image)
    expected_frames = list(range(1, len(groups) + 1))
    if sorted(groups) != expected_frames:
        raise ValueError('nonconsecutive full scene frame scope')
    started = time.perf_counter()
    manifest = dict(status='RUNNING_GT_NOT_READ', protocol=str(PROTOCOL), protocol_sha256=sha256(PROTOCOL), video_id=args.video_id,
                    scene_role='fit' if args.video_id in protocol['scene_split']['fit_video_ids'] else 'validation',
                    full_video=True, groups_planned=len(groups), images_planned=len(images), seed=args.seed,
                    max_retained_events=args.max_retained_events, causal_online=True, continuous_from_first_frame=True,
                    prior_audit_ids=sorted(PRIOR_IDS), provenance=provenance(args.seed))
    save_json(args.output / 'manifest.json', manifest)
    from gtr.geometry.decision_trace import DecisionTrace
    from gtr.geometry.online_runner import CausalGMT
    from gtr.evaluation.mot_evaluation import custom_instances_to_coco_json

    cfg, model = load_gmt(seed=args.seed)
    head_runtime = None
    if args.head_checkpoint is not None:
        from gtr.geometry.omega_head_runtime import CachedOmegaResidual
        image_meta = {int(row['id']): row for row in data['images'] if int(row['video_id']) == args.video_id}
        head_runtime = CachedOmegaResidual(args.head_checkpoint, args.feature_root, args.video_id, image_meta, device='cuda')
    trace = DecisionTrace(model, args.output)
    trace.install()
    original_write = trace.write
    original_association = model.get_asso
    ranking_stream = gzip.open(args.output / 'rankings.jsonl.gz', 'xt')
    pending = []
    primary_calls = 0
    score_rows = 0
    recording_enabled = True

    def association(instances, k, **kwargs):
        nonlocal score_rows
        source_context = None
        if head_runtime is not None:
            query = instances[k]
            query_sources = query.decision_trace_source.detach().cpu().tolist()
            reference_sources = []
            for value in instances:
                if value is not query:
                    reference_sources.extend(value.decision_trace_source.detach().cpu().tolist())
            source_context = (query_sources, reference_sources)
        result = original_association(instances, k=k, **kwargs)
        if kwargs.get('geo_context', True) is False:
            return result
        if head_runtime is not None:
            result = head_runtime.adjust(result, instances, k, model.roi_heads._activate_asso, source_context=source_context)
        query = instances[k]
        refs = [value for index, value in enumerate(instances) if index != k]
        logits = result[0][0]
        ids = torch.cat([value.track_ids for value in refs]) if refs else torch.empty(0, dtype=torch.long, device=logits.device)
        unique = ids.unique(sorted=True)
        if tuple(logits.shape) != (len(query), len(ids)):
            raise ValueError('real primary association shape mismatch')
        if refs:
            probabilities = torch.cat(model.roi_heads._activate_asso(logits.split([len(value) for value in refs], dim=1)), dim=1)
            scores = probabilities @ (ids[:, None] == unique[None, :]).float()
        else:
            scores = logits.new_zeros((len(query), 0))
        order = torch.argsort(scores, dim=1, descending=True, stable=True)
        base = float(model.overlap_thresh)
        passed = []
        raw = []
        margins = []
        null = []
        ranked = []
        for row in range(len(query)):
            raw.append(int(ids[logits[row].argmax()]) if len(ids) else None)
            ranked.append(unique[order[row]].detach().cpu().tolist())
            if len(order[row]) > 1:
                margins.append(float((scores[row, order[row][0]] - scores[row, order[row][1]]) / (scores[row, order[row][0]].abs() + 1e-8)))
            else:
                margins.append(1.0)
            if model.not_mult_thresh:
                boundary = scores.new_tensor(base)
            else:
                multiplicity = torch.zeros_like(unique, dtype=scores.dtype)
                if len(ids):
                    multiplicity = (ids[:, None] == unique[None, :]).sum(0).to(scores.dtype)
                boundary = scores.new_tensor(base) * multiplicity
            passed.append(unique[scores[row] > boundary].detach().cpu().tolist())
            null.append(bool((logits[row] <= 0).all()) if len(ids) else False)
        if recording_enabled:
            pending.append(dict(queries=query.decision_trace_source.detach().cpu().tolist(), ranked_tracks=ranked, raw_top_track=raw,
                                relative_margin=margins, threshold_pass_tracks=passed, null_dominates_all_blocks=null,
                                base_threshold=base, track_ids=unique.detach().cpu().tolist(),
                                track_scores=scores.detach().float().cpu().tolist()))
            score_rows += len(query)
        if recording_enabled and len(pending) != 1:
            raise ValueError('unexpected overlapping primary association calls')
        return result

    def write(stream, record):
        nonlocal primary_calls
        original_write(stream, record)
        if stream is trace.decisions and record['stage'] != 'run_memory_tracker':
            if not pending:
                raise ValueError('solver decision has no matching rank record')
            ranking = pending.pop()
            if ranking['queries'] != record['queries']:
                raise ValueError('raw observation/solver query mismatch')
            ranking_stream.write(json.dumps(ranking, separators=(',', ':'), allow_nan=False) + '\n')
            primary_calls += 1

    trace.write = write
    model.get_asso = association
    seed_all(args.seed)
    tracker = CausalGMT(model)
    predictions = []
    inference_started = time.perf_counter()
    try:
        for frame, group in sorted(groups.items()):
            outputs = tracker.step(map_records(cfg, group))
            for record, output in zip(group, outputs):
                predictions.extend(custom_instances_to_coco_json(output['instances'].to('cpu'), record['image_id']))
            if tracker.frame % 100 == 0:
                print(json.dumps(dict(scene=args.video_id, groups=tracker.frame, total_groups=len(groups), primary_calls=primary_calls)), flush=True)
    finally:
        model.get_asso = original_association
        trace.close()
        ranking_stream.close()
    if pending:
        raise ValueError('unconsumed primary rank records')
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - inference_started
    save_json(args.output / 'predictions.json', predictions)

    # Independent native OFF pass.  No GT has been opened before this point.
    seed_all(args.seed)
    if head_runtime is not None:
        # The OFF replay must carry the same frozen head while all DecisionTrace
        # hooks are absent; otherwise the replay would intentionally differ
        # from the candidate and would not certify deterministic prefix parity.
        recording_enabled = False
        offline_original_inference = model.inference

        def offline_inference(*inference_args, **inference_kwargs):
            result = offline_original_inference(*inference_args, **inference_kwargs)
            batch = inference_args[0]
            if len(result) != len(batch):
                raise ValueError('OFF replay inference output mismatch')
            for record, instance in zip(batch, result):
                image = int(record['image_id'])
                count = len(instance)
                device = instance.pred_boxes.tensor.device
                instance.decision_trace_source = torch.stack((
                    torch.full((count,), image, device=device, dtype=torch.long),
                    torch.arange(count, device=device),
                ), dim=1)
            return result

        model.inference = offline_inference
        model.get_asso = association
    baseline = CausalGMT(model)
    prefix = []
    prefix_images = set()
    for group in list(dict(sorted(groups.items())).values())[:args.precheck_groups]:
        outputs = baseline.step(map_records(cfg, group))
        for record, output in zip(group, outputs):
            prefix_images.add(record['image_id'])
            prefix.extend(custom_instances_to_coco_json(output['instances'].to('cpu'), record['image_id']))
    expected_prefix = [row for row in predictions if row['image_id'] in prefix_images]
    if prefix != expected_prefix:
        raise ValueError('read-only OFF prefix prediction parity failed before GT labeling')
    if head_runtime is not None:
        model.inference = offline_original_inference
    model.get_asso = original_association
    manifest.update(status='INFERENCE_PARITY_VERIFIED_PENDING_POST_LABELS', groups=len(groups), images=len(images),
                    primary_calls=primary_calls, inference_seconds=inference_seconds, prefix_raw_prediction_exact=True)
    save_json(args.output / 'manifest.json', manifest)

    # This is the first call that reads raw GT.  summarize() validates the
    # prior immutable GT hashes and emits masked identity outcomes.
    summary = summarize(args.output, data, args.video_id, images, args.max_retained_events)
    save_json(args.output / 'summary.json', summary)
    manifest.update(status='COMPLETE_FULL_CAUSAL_AUDIT_WITH_NATIVE_SCORE_MATRICES' if head_runtime is None else 'COMPLETE_FULL_CAUSAL_HEAD_AUDIT', full_video=True, wall_seconds=time.perf_counter() - started,
                    raw_GT_labeling_after_full_inference_and_OFF_parity=True, GT_features_to_tracker=False,
                    future_query_features=False, vggt_calls=0, dinov2_calls=0, speed_eligible=False,
                    persistent_native_feature_vectors=0, persistent_dense_candidate_score_matrices=score_rows,
                    predictions_sha256=sha256(args.output / 'predictions.json'),
                    artifacts={path.name: sha256(path) for path in args.output.iterdir() if path.is_file() and path.name != 'manifest.json'},
                    source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
    if head_runtime is not None:
        manifest.update(head_checkpoint=str(args.head_checkpoint), head_checkpoint_sha256=sha256(args.head_checkpoint), feature_root=str(args.feature_root),
                       cached_external_feature_calls=head_runtime.calls, adjusted_query_rows=head_runtime.adjusted_queries,
                       missing_external_feature_sources=head_runtime.missing_sources, speed_eligible=False,
                       integration_scope='causal validation audit with frozen per-source Ω cache; no live Ω inference FPS claim')
    save_json(args.output / 'manifest.json', manifest)
    print(json.dumps(dict(scene=args.video_id, status=manifest['status'], groups=len(groups), primary_calls=primary_calls, score_rows=score_rows,
                          counts=summary['counts'], rates=summary['rates'])), flush=True)


if __name__ == '__main__':
    main()
