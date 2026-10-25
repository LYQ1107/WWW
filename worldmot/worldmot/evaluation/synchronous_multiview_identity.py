"""Posthoc synchronous identity metrics, independent of online tracking state.

Events are keyed by original GT identity and camera, never detection ordinal.
Conditional recall does not measure detector recall and is not CVIDF1/HOTA.
"""
from collections import Counter
import numpy as np
from ..training.world_object_identity_targets import box_iou_matrix, identity_key


def match_boxes(tracks, gt_by_camera, iou_threshold=.5):
    """Maximum-cardinality, maximum-IoU one-to-one matching with ambiguity mask."""
    from scipy.optimize import linear_sum_assignment
    matched = {}; audit = Counter(); masked = []; unmatched = []
    for camera in sorted(set(gt_by_camera) | {int(x['camera_id']) for x in tracks}):
        gt = list(gt_by_camera.get(camera, [])); counts = Counter(identity_key(x) for x in gt)
        clean, bad = [], []
        for x in gt:
            b = np.asarray(x.get('bbox', []), float)
            valid = (identity_key(x) is not None and counts[identity_key(x)] == 1
                     and not x.get('ignore', False) and not x.get('iscrowd', False)
                     and x.get('category_id', 1) == 1 and b.shape == (4,)
                     and np.isfinite(b).all() and (b[2:] > 0).all())
            (clean if valid else bad).append(x)
        audit['valid_GT'] += len(clean); audit['ambiguous_or_invalid_GT'] += len(bad)
        detections = [(i, x) for i, x in enumerate(tracks) if int(x['camera_id']) == camera]
        boxes = [x['bbox_xywh'] for _, x in detections]
        bad_boxes = [x['bbox'] for x in bad if np.asarray(x.get('bbox', [])).shape == (4,)
                     and np.isfinite(x['bbox']).all() and min(x['bbox'][2:]) > 0]
        ambiguous = box_iou_matrix(boxes, bad_boxes)
        excluded = {i for i in range(len(detections)) if ambiguous.shape[1]
                    and ambiguous[i].max() >= iou_threshold}
        masked.extend(detections[i][0] for i in excluded)
        overlap = box_iou_matrix(boxes, [x['bbox'] for x in clean])
        eligible = overlap >= iou_threshold
        if excluded: eligible[list(excluded)] = False
        if eligible.size:
            # The extra reward makes one extra legal match worth more than
            # any sum of IoU improvements; invalid edges are never emitted.
            reward = eligible * (overlap + min(eligible.shape) + 1)
            rows, cols = linear_sum_assignment(-reward)
            for i, j in zip(rows, cols):
                if eligible[i, j]:
                    index, track = detections[i]
                    matched[camera, identity_key(clean[j])] = dict(track, match_IoU=float(overlap[i,j]),
                                                                 prediction_index=index)
        used = {x['prediction_index'] for x in matched.values()}
        unmatched.extend(i for i, _ in detections if i not in used and i not in masked)
    audit.update(detections=len(tracks), matched=len(matched), masked_detections=len(masked),
                 unmatched_detections=len(unmatched))
    return matched, {'counts': dict(audit), 'ambiguity_mask_indices': masked,
                     'unmatched_prediction_indices': unmatched}


def events(matched, allowed_keys=None):
    """Yield each physical cross-camera GT pair once."""
    keys = sorted(set(matched) if allowed_keys is None else set(matched) & set(allowed_keys))
    for i, a in enumerate(keys):
        for b in keys[i+1:]:
            if a[0] == b[0]: continue
            yield (a, b), a[1] == b[1], matched[a]['global_id'] == matched[b]['global_id']


def update_counts(counts, same, shared):
    counts['same_person_pairs' if same else 'different_person_pairs'] += 1
    counts['TP_crossview' if same and shared else 'FN_crossview' if same else
           'FP_crossview' if shared else 'TN_crossview'] += 1


def summarize(counts):
    c = Counter(counts); tp, fp, fn = (c[x] for x in ('TP_crossview','FP_crossview','FN_crossview'))
    return {**dict(c), 'Precision_crossview': tp/(tp+fp) if tp+fp else None,
            'Recall_crossview': tp/(tp+fn) if tp+fn else None,
            'F1_crossview': 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
            'False_Merge_Rate': fp/c['different_person_pairs'] if c['different_person_pairs'] else None,
            'scope': 'conditional on unambiguous one-to-one detection matches; not a MOT metric'}


def paired_counts(left, right, keys=None):
    common = set(left) & set(right)
    if keys is not None: common &= set(keys)
    counts = Counter(); by_kind = {'same_person': Counter(), 'different_person': Counter()}
    for (a,b), same, shared in events(left, common):
        correct_left = shared == same
        correct_right = (right[a]['global_id'] == right[b]['global_id']) == same
        kind = ('both_correct' if correct_left and correct_right else 'both_wrong' if
                not correct_left and not correct_right else 'right_corrects_left' if correct_right
                else 'right_regresses_left')
        counts[kind] += 1; by_kind['same_person' if same else 'different_person'][kind] += 1
    return dict(counts), {k: dict(v) for k,v in by_kind.items()}
