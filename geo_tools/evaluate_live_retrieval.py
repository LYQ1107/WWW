"""Post-run GT evaluation of public causal query responses; no re-ranking."""
import argparse
import collections
import numpy as np
from scipy.optimize import linear_sum_assignment
from common import *


def iou_matrix(left, right):
    left, right = np.asarray(left, dtype=float).reshape(-1, 4), np.asarray(right, dtype=float).reshape(-1, 4)
    lt = np.maximum(left[:, None, :2], right[None, :, :2])
    rb = np.minimum(left[:, None, :2]+left[:, None, 2:], right[None, :, :2]+right[None, :, 2:])
    intersection = np.prod(np.maximum(rb-lt, 0), axis=-1)
    area1 = np.prod(left[:, 2:], axis=-1)[:, None]
    area2 = np.prod(right[:, 2:], axis=-1)[None, :]
    return intersection / np.maximum(area1+area2-intersection, 1e-12)


def ranking_outcome(response, positive_ids):
    ranks = [rank for rank, clip_id in enumerate(response['ranked_clip_ids'], 1) if clip_id in positive_ids]
    return min(ranks) if ranks else None


def empty_ranking_stats():
    return dict(requests=0, matched=0, eligible=0, covered=0, hits1=0, hits5=0, hits10=0, reciprocal=0.)


def record_ranking(stats, response, positive_ids):
    # Eligibility is defined by the original full published gallery. Candidate
    # restrictions or absent query features cannot shrink this denominator.
    stats['eligible'] += 1
    rank = ranking_outcome(response, positive_ids)
    if rank is not None:
        stats['covered'] += 1
        stats['reciprocal'] += 1/rank
        for k in [1,5,10]:
            stats[f'hits{k}'] += rank <= k


def finalize_ranking(stats):
    ratio = lambda numerator, denominator: numerator/denominator if denominator else None
    return dict(**stats, recall_at_k={str(k):ratio(stats[f'hits{k}'], stats['eligible']) for k in [1,5,10]},
                mrr=ratio(stats['reciprocal'], stats['eligible']), candidate_positive_coverage=ratio(stats['covered'], stats['eligible']),
                success_at_k_all_requests={str(k):ratio(stats[f'hits{k}'], stats['requests']) for k in [1,5,10]})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--query-scene', type=int)
    parser.add_argument('--verify-existing',action='store_true',help='recompute and exactly verify an existing metric file without replacing it')
    parser.add_argument('--outcome-output',type=Path,help='optional separate post-run per-query GT outcome artifact; no inference/reranking')
    args = parser.parse_args()
    report_path = args.run/('retrieval.metrics.json' if args.query_scene is None else f'retrieval.scene_{args.query_scene}.metrics.json')
    if report_path.exists() and not args.verify_existing:
        raise FileExistsError(report_path)
    if args.verify_existing and not report_path.exists():raise FileNotFoundError(report_path)
    if args.outcome_output is not None and args.outcome_output.exists():raise FileExistsError(args.outcome_output)
    retained=json.loads(report_path.read_text()) if args.verify_existing else None
    manifest = json.loads((args.run/'manifest.json').read_text())
    if args.query_scene is not None and args.query_scene not in manifest['video_ids']:
        raise ValueError('query scene is not present in this run')
    for name, expected in manifest['artifacts'].items():
        if sha256(args.run/name) != expected:
            raise ValueError(f'changed public run artifact: {name}')
    data = train_data()
    labels, gt_counts, gt_hashes, image_scenes = {}, {}, {}, {}
    for scene in manifest['video_ids']:
        records = video_records(data, scene)
        image_by_view_frame = {(r['view_id'], r['frame_id']): r['image_id'] for r in records}
        scene_name = next(v['file_name'] for v in data['videos'] if v['id'] == scene)
        ground_truth = collections.defaultdict(list)
        for view in sorted({r['view_id'] for r in records}):
            path = DATA/'train'/f'{scene_name}_View{view}'/'gt/gt.txt'
            gt_hashes[str(path)] = sha256(path)
            seen = set()
            for line in path.read_text().splitlines():
                fields = [float(value) for value in line.split(',')]
                frame, identity = map(int, fields[:2])
                if (frame, identity) in seen:
                    raise ValueError('strict retrieval evaluation requires unique raw frame GT identities')
                seen.add((frame, identity))
                # Same pedestrian eligibility as MotChallenge2DBox. All raw
                # rows remain in source; ignored classes are not positives.
                if fields[6] == 0 or (len(fields) > 7 and fields[7] != 1):
                    continue
                image_id = image_by_view_frame[(view, frame)]
                ground_truth[image_id].append((identity, fields[2:6]))
        predictions = collections.defaultdict(list)
        for row in json.loads((args.run/f'video_{scene:02d}.json').read_text()):
            predictions[row['image_id']].append(row)
        for record in records:
            image_scenes[record['image_id']] = scene
            gt_counts[record['image_id']] = len(ground_truth[record['image_id']])
        for image_id, rows in predictions.items():
            if len({row['track_id'] for row in rows}) != len(rows):
                raise ValueError('ambiguous detector labels: duplicate predicted track IDs within one image')
            truth = ground_truth[image_id]
            gt_counts[image_id] = len(truth)
            for row in rows:
                labels[(image_id, int(row['track_id']))] = None
            if not truth:
                continue
            overlaps = iou_matrix([row['bbox'] for row in rows], [row[1] for row in truth])
            cost = np.where(overlaps >= .5, 1-overlaps, 1e6)
            row_ids, col_ids = linear_sum_assignment(cost)
            for row_id, col_id in zip(row_ids, col_ids):
                if overlaps[row_id, col_id] >= .5:
                    labels[(image_id, int(rows[row_id]['track_id']))] = (scene, truth[col_id][0])
    snapshots, query_images = {}, set()
    for line in (args.run/'gallery_snapshots.jsonl').open():
        snapshot = json.loads(line)
        query_images.update(image_id for image_id in snapshot['query_image_ids'] if args.query_scene is None or image_scenes[image_id] == args.query_scene)
        rows = []
        for clip in snapshot['entries']:
            counts = collections.Counter(labels.get((obs['image_id'], clip['track'])) for obs in clip['observations'])
            counts.pop(None, None)
            winner = max(counts, key=lambda identity: (counts[identity], identity)) if counts else None
            # Strict majority over every detected observation, including
            # unmatched detections. Mixed tracks cannot claim every identity.
            identity = winner if winner is not None and counts[winner] > len(clip['observations'])/2 else None
            rows.append((clip['clip_id'], (clip['scene'], clip['camera']), identity))
        snapshots[snapshot['round_id']] = rows
    requests = matched = eligible = 0
    hits = {1:0, 5:0, 10:0}
    reciprocal = reciprocal10 = 0.
    per_scene = collections.defaultdict(empty_ranking_stats)
    diagnostic_stats = collections.defaultdict(empty_ranking_stats)
    outcomes=[]
    for line in (args.run/'queries.jsonl').open():
        row = json.loads(line)
        query, response = row['query'], row['response']
        if args.query_scene is not None and query['scene'] != args.query_scene:
            continue
        for mode in row.get('diagnostic_responses', {}):
            diagnostic_stats[mode]['requests'] += 1
        requests += 1
        scene = per_scene[query['scene']]
        scene['requests'] += 1
        query_images.add(query['image_id'])
        identity = labels.get((query['image_id'], query['track']))
        outcome=None
        if args.outcome_output is not None:
            outcome=dict(query={key:query[key] for key in ('scene','camera','round_id','image_id','frame_id','bbox')},
                         published_track_id=query['track'],gt_identity=identity,matched=identity is not None,eligible=False,
                         first_positive_rank=None,positive_gallery_clips=0)
            outcomes.append(outcome)
        if identity is None:
            continue
        matched += 1
        scene['matched'] += 1
        for mode in row.get('diagnostic_responses', {}):
            diagnostic_stats[mode]['matched'] += 1
        positive_ids = {clip_id for clip_id, camera, target_id in snapshots[query['round_id']]
                        if camera != (query['scene'], query['camera']) and target_id == identity}
        if not positive_ids:
            continue
        eligible += 1
        if outcome is not None:outcome.update(eligible=True,positive_gallery_clips=len(positive_ids))
        record_ranking(scene, response, positive_ids)
        for mode, diagnostic_response in row.get('diagnostic_responses', {}).items():
            record_ranking(diagnostic_stats[mode], diagnostic_response, positive_ids)
        rank = ranking_outcome(response, positive_ids)
        if outcome is not None:outcome['first_positive_rank']=rank
        if rank is not None:
            reciprocal += 1/rank
            reciprocal10 += (1/rank if rank <= 10 else 0.)
            for k in hits:
                hits[k] += rank <= k
    ratio = lambda numerator, denominator: numerator/denominator if denominator else None
    report = dict(status='PASS', run_manifest_sha256=sha256(args.run/'manifest.json'), raw_gt_sha256=gt_hashes,
                  requests=requests, gt_matched_queries=matched, gallery_eligible_queries=eligible,
                  detector_query_precision=ratio(matched, requests), gt_query_detection_coverage=ratio(matched, sum(gt_counts.get(i, 0) for i in query_images)),
                  gallery_eligibility_among_matched=ratio(eligible, matched),
                  recall_at_k={str(k):ratio(count, eligible) for k, count in hits.items()}, mrr=ratio(reciprocal, eligible), mrr_at_10=ratio(reciprocal10, eligible),
                  success_at_k_all_requests={str(k):ratio(count, requests) for k, count in hits.items()},
                  query_scene=args.query_scene,per_scene={scene:finalize_ranking(stats) for scene,stats in per_scene.items()},
                  scoring_diagnostics={mode:finalize_ranking(stats) for mode,stats in diagnostic_stats.items()},
                  scoring_diagnostic_denominator='all native-gallery eligible detector queries retained across every scorer, including missing-feature queries and positives without encoded gallery candidates',
                  metric_protocol='one-to-one detector/GT IoU>=0.5; identity=(scene,raw_GT_ID); clip positive requires strict GT majority across all its observations; Recall and full MRR conditional on GT-matched query with at least one positive other-camera clip; all candidate ranks retained',
                  query_selection_uses_gt=False, ranking_uses_gt=False, future_gallery_allowed=False,
                  limitations='finite local dataset replay, not live Internet feeds; clip identity majority is an evaluation convention')
    if args.verify_existing:
        if json.loads(json.dumps(report))!=retained:raise ValueError('retained metrics do not match exact post-run recomputation')
    else:save_json(report_path, report)
    if args.outcome_output is not None:
        save_json(args.outcome_output,dict(status='MATCHES_RETAINED_METRICS' if args.verify_existing else 'WRITTEN_WITH_METRICS',
            run=str(args.run),query_scene=args.query_scene,metric=str(report_path),metric_sha256=sha256(report_path),
            run_manifest_sha256=sha256(args.run/'manifest.json'),raw_gt_sha256=gt_hashes,outcomes=outcomes,
            speed_eligible=False,analysis_only=True,
            scope='post-run original ranked responses labelled with GT; all matched, unmatched and ineligible detector queries retained; no query selection, ranking, inference or metric overwrite',provenance=provenance()))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
