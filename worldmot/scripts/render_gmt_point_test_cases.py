#!/usr/bin/env python3
"""Render actual frozen TEST RGB, detector boxes, global IDs and point outputs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

from analyze_gmt_point_test import match_boxes, raw_gt
from gmt_point_report_common import NAS, LEGAL, artifact, new_routes, read_json
from worldmot.data.visiontrack import VisionTrack


def draw(row, scene, target, gt, dataset, settings, output, case):
    prediction = next(p for p in row['point_predictions']
                      if p['reference_key'] == [1, row['timestamp']] and [2, row['timestamp']] in p['image_keys'])
    source = row['detections_by_camera']['1']
    mapping = match_boxes(source, gt[1][row['timestamp']])
    index = next(i for i, j in mapping.items() if gt[1][row['timestamp']][j]['id'] == target)
    detection_id = int(source[index]['detection_id'])
    offset = 0
    selected = None
    for did, count in prediction['source_detections']:
        if int(did) == detection_id:
            selected = slice(offset, offset + count)
        offset += count
    if selected is None:
        raise ValueError('Chosen source detection has no logged real queries')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    point_summary = {}
    for column, view in enumerate((1, 2)):
        image = np.asarray(Image.open(dataset.image_path(dataset.images[int(row['image_ids'][str(view)])])).convert('RGB'))
        for ax in axes[:, column]:
            ax.imshow(image)
            ax.set_xlim(0, image.shape[1]); ax.set_ylim(image.shape[0], 0)
            ax.axis('off')
            for obj in row['tracks_2d_by_camera'][str(view)]:
                x, y, w, h = obj['bbox_xywh']
                color = plt.get_cmap('tab20')(int(obj['global_id']) % 20)
                ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=color, linewidth=1.1))
                ax.text(x, y, str(obj['global_id']), color='white', fontsize=8,
                        bbox=dict(facecolor=color, alpha=.8, pad=1))
            ann = next(g for g in gt[view][row['timestamp']] if g['id'] == target)
            x, y, w, h = ann['bbox']
            ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor='cyan', linestyle='--', linewidth=1.5))
        at = prediction['image_keys'].index([view, row['timestamp']])
        xy = np.asarray(prediction['predicted_xy'][at])[selected]
        valid = np.asarray(prediction['valid'][at], dtype=bool)[selected]
        conf = np.asarray(prediction['confidence'][at])[selected]
        visibility = np.asarray(prediction['visibility'][at])[selected]
        good = valid & (visibility >= settings['minimum_visibility']) & (conf >= settings['minimum_confidence'])
        # Every marker is a logged network prediction, including rejected ones.
        axes[1, column].scatter(xy[valid & ~good, 0], xy[valid & ~good, 1], s=18, c='orange', marker='x')
        axes[1, column].scatter(xy[good, 0], xy[good, 1], s=18, c='lime', marker='o' if view == 1 else 'x')
        point_summary[str(view)] = dict(query_count=len(xy), in_image=int(valid.sum()),
                                        passing_frozen_gate=int(good.sum()), median_confidence=float(np.median(conf)))
        axes[0, column].set_title(f'TEST {scene["name"]} frame {row["timestamp"]}, camera {view}: delivered IDs')
        axes[1, column].set_title(f'Actual A-anchored points: {int(good.sum())}/{len(xy)} pass; median conf={np.median(conf):.3f}')
    fig.suptitle(f'{case}: same raw GT person {target}; cyan dashed = raw GT; green = frozen point gate pass; orange = rejected', fontsize=11)
    fig.tight_layout()
    fig.savefig(output, dpi=120)
    plt.close(fig)
    return dict(case=case, scene=scene['scene'], timestamp=row['timestamp'], GT_identity=target,
                image_ids=row['image_ids'], point_summary=point_summary, image=artifact(output))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--nas', type=Path, default=NAS)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    route = next(r for r in new_routes(args.nas) if r.key == 'point_joint')
    ds = VisionTrack('/data/DATASETS/TRACKING/JDE/VisionTrack',
                     annotation=str(args.nas / 'fulltrain_20261022/audit/test_metadata.json'),
                     split='test', load_annotations=False)
    audit = read_json(args.nas / 'fulltrain_20261022/audit/data_audit.json')['test']['scenes']
    examples = {}
    # Deterministic first logged success/failure in ascending legal scene/time.
    # GT selects only the illustration; no inference or threshold is changed.
    for scene in sorted(audit, key=lambda s: s['scene']):
        if scene['scene'] not in LEGAL:
            continue
        gt = raw_gt(scene)
        directory = route.predictions / f'scene_{scene["scene"]:02d}'
        manifest = read_json(directory / 'manifest.json')
        with (directory / 'tracks.jsonl').open() as stream:
            for line in stream:
                row = json.loads(line)
                if not any(p['reference_key'] == [1, row['timestamp']] and [2, row['timestamp']] in p['image_keys']
                           for p in row.get('point_predictions', [])):
                    continue
                outputs = {}
                for view in (1, 2):
                    objects = row['tracks_2d_by_camera'][str(view)]
                    matches = match_boxes(objects, gt[view][row['timestamp']])
                    outputs[view] = {gt[view][row['timestamp']][j]['id']: objects[i]['global_id'] for i, j in matches.items()}
                for target in sorted(set(outputs[1]) & set(outputs[2])):
                    case = 'shared_id_success' if outputs[1][target] == outputs[2][target] else 'same_person_split_failure'
                    if case in examples:
                        continue
                    image = args.output / f'gmt_point_test_{case}.jpg'
                    examples[case] = draw(row, scene, target, gt, ds, manifest['settings'], image, case)
                    print(json.dumps(examples[case]), flush=True)
                if len(examples) == 2:
                    break
        if len(examples) == 2:
            break
    (args.output / 'gmt_point_test_cases.json').write_text(json.dumps(dict(
        status='ACTUAL_FROZEN_TEST_ILLUSTRATIONS', selection='First logged example of each outcome in ascending legal scene/time',
        GT_only_used_posthoc=True, no_threshold_or_model_changes=True, examples=examples), indent=2) + '\n')


if __name__ == '__main__':
    main()
