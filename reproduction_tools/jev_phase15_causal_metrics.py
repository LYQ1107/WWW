"""Actual future CLEAR counts retaining pre-intervention matching history.

Perception is frozen, so original prepared GT/boxes/scores are reused exactly.
Only tracker IDs from the executed native branch replace the original IDs.
The official preprocessing and CLEAR implementation remain untouched.
"""
import copy
import collections
import numpy as np
from jev_phase15_common import *


def truncate(value, original_frames, stop):
    if isinstance(value, list) and len(value) == original_frames: return value[:stop]
    if isinstance(value, dict): return {k: truncate(v, original_frames, stop) for k, v in value.items()}
    return value


class FrozenClearFuture:
    def __init__(self, variant, seed, video, case_override=None, duplicate_gt_diagnostic=False):
        sys.path.insert(0, str(ROOT / 'TrackEval'))
        for name, kind in [('float', float), ('int', int), ('bool', bool)]:
            if name not in np.__dict__: setattr(np, name, kind)
        import trackeval
        self.trackeval = trackeval; self.video = video
        out = old_case(variant, seed, video) if case_override is None else Path(case_override)
        original = read(out / 'RESULT.json')
        manifest = read(out / 'strict_eval/tracking_eval_runtime_state/native/prepared/manifest.json')
        cfg = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
        cfg.update(GT_FOLDER=manifest['trackeval_gt'], TRACKERS_FOLDER=manifest['trackeval_trackers'],
                   TRACKERS_TO_EVAL=['GMT'], BENCHMARK='VisionTrack', SPLIT_TO_EVAL='test',
                   SKIP_SPLIT_FOL=True, SEQ_INFO=manifest['seq_lengths'], PRINT_CONFIG=False)
        if duplicate_gt_diagnostic:
            assert video == 14, 'explicit isolated diagnostic scope only'
            from evaluate_visiontrack import PermissiveVisionTrackDataset
            dataset_class = PermissiveVisionTrackDataset
        else: dataset_class = trackeval.datasets.MotChallenge2DBox
        self.dataset = dataset_class(cfg)
        self.templates = {}; self.original = original; self.raw_by_key = collections.defaultdict(list)
        annotations = read(ANNOTATIONS)
        images = {im['id']: im for im in annotations['images'] if im['video_id'] == video}
        self.images = {(im['frame_id'] - 1, im['view_id'] - 1): im for im in images.values()}
        for row in read(out / 'RAW_PREDICTIONS.json'):
            im = images[row['image_id']]; self.raw_by_key[im['frame_id'] - 1, im['view_id'] - 1].append(row)
        self.detector_rows = {}
        for sequence in manifest['sequences']:
            view = int(sequence.rsplit('View', 1)[1]) - 1
            template = self.dataset.get_raw_seq_data('GMT', sequence)
            self.templates[view] = template
            for frame, identifiers in enumerate(template['tracker_ids']):
                rows = self.raw_by_key[frame, view]
                lookup = {p['track_id']: index for index, p in enumerate(rows)}
                assert len(rows) == len(lookup) == len(identifiers)
                self.detector_rows[frame, view] = [lookup[int(ref)] for ref in identifiers]
        self.evaluator = {'kind': 'unmodified TrackEval CLEAR + MotChallenge2DBox preprocessing',
            'CLEAR_code': ref(ROOT / 'TrackEval/trackeval/metrics/clear.py'),
            'dataset_code': ref(ROOT / 'TrackEval/trackeval/datasets/mot_challenge_2d_box.py'),
            'GT_and_box_source': ref(out / 'strict_eval/tracking_eval_runtime_state/native/prepared/manifest.json'),
            'strict_GT_unique': not any(manifest.get('gt_duplicate_rows',{}).values()),
            'duplicate_gt_diagnostic': duplicate_gt_diagnostic,
            'raw_GT_duplicate_rows': manifest.get('gt_duplicate_rows',{}),
            'permissive_dataset_code': ref(ROOT/'reproduction_tools/evaluate_visiontrack.py') if duplicate_gt_diagnostic else None,
            'strict_clear_causal_gate_eligible': not duplicate_gt_diagnostic,
            'internal_split_label_test_is_frozen_TRAIN_subset': True, 'official_TEST_accessed': False}

    def ids(self, instances):
        result = {}
        for index, inst in enumerate(instances):
            frame, view = divmod(index, 2)
            if not inst.has('track_ids'): continue
            identifiers = inst.track_ids.cpu().tolist()
            rows = self.raw_by_key[frame, view]
            assert len(identifiers) == len(rows), 'branch changed frozen perception detections'
            im = self.images[frame, view]; boxes = inst.pred_boxes.tensor.cpu().numpy().copy()
            boxes[:, [0, 2]] *= im['width'] / inst.image_size[1]
            boxes[:, [1, 3]] *= im['height'] / inst.image_size[0]
            boxes[:, 2:] -= boxes[:, :2]
            expected = np.asarray([p['bbox'] for p in rows]).reshape(-1, 4)
            assert np.allclose(boxes, expected, rtol=0, atol=0.001), 'branch changed frozen detection geometry'
            assert len(identifiers) == len(set(identifiers)), 'illegal same-camera identity collision'
            result[frame, view] = identifiers
        return result

    def count(self, ids, stops):
        result = {}; total = 0
        for view, template in self.templates.items():
            end = stops[view]
            if end == 0: result[str(view)] = 0; continue
            data = copy.deepcopy(template); frames = data['num_timesteps']
            data = truncate(data, frames, end); data['num_timesteps'] = end
            replacement = []
            for frame in range(end):
                rows = self.detector_rows[frame, view]
                replacement.append(np.asarray([ids[frame, view][row] for row in rows], dtype=int))
            data['tracker_ids'] = replacement
            data = self.dataset.get_preprocessed_seq_data(data, 'pedestrian')
            value = int(self.trackeval.metrics.CLEAR({'PRINT_CONFIG': False}).eval_sequence(data)['IDSW'])
            result[str(view)] = value; total += value
        return {'IDSW': total, 'per_camera_IDSW': result}

    def horizon(self, ids, boundary, horizon, total_frames):
        frame, view = boundary
        end = min(total_frames, frame + horizon)
        past = self.count(ids, {camera: frame + int(camera < view) for camera in [0, 1]})
        complete = self.count(ids, {0: end, 1: end})
        return {'future_CLEAR_IDSW': complete['IDSW'] - past['IDSW'],
                'cumulative_CLEAR_IDSW': complete['IDSW'], 'unchanged_prefix_CLEAR_IDSW': past['IDSW'],
                'per_camera_future_CLEAR_IDSW': {camera: complete['per_camera_IDSW'][camera] - past['per_camera_IDSW'][camera] for camera in ['0', '1']},
                'scene_frame_horizon_requested': horizon, 'scene_frames_observed': end - frame,
                'right_censored_at_video_end': end < frame + horizon,
                'matching_history_retained_from_frame_zero': True, 'evaluator': self.evaluator}


if __name__ == '__main__':
    evaluator = FrozenClearFuture('multi_question', 20261009, 17)
    native = {key: [p['track_id'] for p in rows] for key, rows in evaluator.raw_by_key.items()}
    counts = evaluator.count(native, {0: evaluator.original['frames'], 1: evaluator.original['frames']})
    assert counts['IDSW'] == evaluator.original['strict_online_metrics']['IDSW'] == 329
    save(OUT / 'causal_metric_test/RESULT.json', {'status': 'PASS', 'binding': binding(),
         'source_official_IDSW': 329, 'actual_reconstructed_IDSW': counts['IDSW'], 'evaluator': evaluator.evaluator,
         'scientific_scope': 'metric wiring qualification only; no commitment benefit claim'})
    print('PHASE15_TRUE_FUTURE_CLEAR_METRIC_WIRING_PASS', counts, flush=True)
