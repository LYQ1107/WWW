import itertools
import json
import numpy as np
import os
from collections import defaultdict
from multiprocessing import freeze_support
from pathlib import Path
import pycocotools.mask as mask_util
from detectron2.structures import Boxes, BoxMode, pairwise_iou
from fvcore.common.file_io import PathManager
from detectron2.evaluation.coco_evaluation import COCOEvaluator, _evaluate_predictions_on_coco
from ..tracking.naive_tracker import track
from ..tracking import trackeval

def eval_track(out_dir, year):
    freeze_support()

    default_eval_config = trackeval.Evaluator.get_default_eval_config()
    default_eval_config['DISPLAY_LESS_PROGRESS'] = True
    
    default_dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()

    default_metrics_config = {'METRICS': ['HOTA', 'CLEAR', 'Identity']}
    config = {
        **default_eval_config, **default_dataset_config, **default_metrics_config}  # Merge default configs
    config['GT_FOLDER'] = 'datasets/mot/MOT{}/'.format(year)
    config['SPLIT_TO_EVAL'] = 'half_val'
    config['TRACKERS_FOLDER'] = out_dir
    eval_config = {k: v for k, v in config.items() if k in default_eval_config.keys()}
    dataset_config = {k: v for k, v in config.items() if k in default_dataset_config.keys()}
    metrics_config = {k: v for k, v in config.items() if k in default_metrics_config.keys()}
    print('config', config)
    # Run code
    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
    metrics_list = []
    for metric in [trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity, trackeval.metrics.VACE]:
        if metric.get_name() in metrics_config['METRICS']:
            metrics_list.append(metric())
    evaluator.evaluate(dataset_list, metrics_list)


def save_cocojson_as_mottxt(out_dir, videos, video2images, per_image_preds):
    if os.path.exists(out_dir):
        print('removing', out_dir)
        os.system('rm -rf {}'.format(out_dir))
    os.makedirs(out_dir)
    for video in videos:
        video_id = video['id']
        file_name = video['file_name']
        out_path = out_dir + '/{}.txt'.format(file_name)
        f = open(out_path, 'w')
        images = video2images[video_id]
        tracks = defaultdict(list)
        for image_info in images:
            result = per_image_preds[image_info['id']]
            frame_id = image_info['frame_id']
            for item in result:
                if not ('track_id' in item):
                    assert 0, 'No track ID!!'
                tracking_id = item['track_id']
                bbox = item['bbox']
                bbox = [bbox[0], bbox[1], bbox[2], bbox[3]]
                tracks[tracking_id].append([frame_id] + bbox)
        rename_track_id = 0
        for track_id in sorted(tracks):
            rename_track_id += 1
            for t in tracks[track_id]:
                f.write('{},{},{:.2f},{:.2f},{:.2f},{:.2f},-1,-1,-1,-1\n'.format(
                    t[0], rename_track_id, t[1], t[2], t[3], t[4])) # 
        f.close()


def track_and_eval_mot(out_dir, data, preds, dataset_name):
    videos = sorted(data['videos'], key=lambda x: x['id'])
    images = sorted(data['images'], key=lambda x: x['id'])
    video2images = defaultdict(list)
    for image in images:
        video2images[image['video_id']].append(image)
    for video in video2images:
        video2images[video] = sorted(
            video2images[video], key=lambda x: x['frame_id'])
    per_image_preds = defaultdict(list)
    for x in preds:
        if x['score'] > 0.4:
            per_image_preds[x['image_id']].append(x)
    has_track_id = len(preds) > 0 and 'track_id' in preds[0]
    del preds
    year = '20' if '20' in dataset_name else '17'
    split = 'trainval' if year == '17' else 'train'
    mot_out_dir = out_dir + '/moteval/{}/pred/data/'.format(split)
    if not has_track_id:
        print('Runing naive tracker')
        mot_out_dir = out_dir + '/moteval/{}/naive/data/'.format(split)
        for video in videos:
            images = video2images[video['id']]
            print('Runing tracking ...', video['file_name'], len(images))
            preds = [per_image_preds[x['id']] for x in images]
            preds = track(preds)
    save_cocojson_as_mottxt(
        mot_out_dir, videos, video2images, per_image_preds)
    eval_track(out_dir + '/moteval', year)


def custom_instances_to_coco_json(instances, img_id):
    """
    Add track_id
    """
    num_instance = len(instances)
    if num_instance == 0:
        return []

    boxes = instances.pred_boxes.tensor.numpy()
    boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    boxes = boxes.tolist()
    scores = instances.scores.tolist()
    classes = instances.pred_classes.tolist()

    has_mask = instances.has("pred_masks")
    if has_mask:
        rles = [
            mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
            for mask in instances.pred_masks
        ]
        for rle in rles:
            rle["counts"] = rle["counts"].decode("utf-8")

    has_keypoints = instances.has("pred_keypoints")
    if has_keypoints:
        keypoints = instances.pred_keypoints

    has_track_id = instances.has("track_ids")
    if has_track_id:
        track_ids = instances.track_ids

    results = []
    for k in range(num_instance):
        result = {
            "image_id": img_id,
            "category_id": classes[k],
            "bbox": boxes[k],
            "score": scores[k],
        }
        if has_mask:
            result["segmentation"] = rles[k]
        if has_keypoints:
            keypoints[k][:, :2] -= 0.5
            result["keypoints"] = keypoints[k].flatten().tolist()
        if has_track_id:
            result['track_id'] = int(track_ids[k].item())
        results.append(result)
    
    return results


def align_visiontrack_inputs_to_outputs(dataset_name, inputs, outputs):
    """Return input records in the order used by GMT test outputs.

    ``GMTDatasetMapper`` stores multi-view records in view blocks, while
    ``GTRRCNN.sliding_inference_GMT`` appends each frame's views together and
    postprocesses/returns them in frame-major order.  The evaluator must use
    the same order when attaching ``image_id`` to each output.  Keep this
    correction local to the VisionTrack evaluator so ordinary Detectron2
    evaluators and single-view inputs retain their original behavior.
    """
    if dataset_name not in {"VISION_train", "VISION_test"} or len(inputs) <= 1:
        return inputs
    if len(inputs) != len(outputs):
        raise ValueError(
            "VisionTrack evaluator received different input/output lengths: "
            f"{len(inputs)} != {len(outputs)}"
        )
    view_num = int(inputs[0].get("view_num", -1))
    if view_num <= 1:
        return inputs
    if any(int(record.get("view_num", view_num)) != view_num for record in inputs):
        raise ValueError("VisionTrack batch contains inconsistent view_num values")
    frames, remainder = divmod(len(inputs), view_num)
    if remainder:
        raise ValueError(
            "VisionTrack multi-view batch is not divisible by view_num: "
            f"{len(inputs)} % {view_num} != 0"
        )
    # Input records are [view1 all frames, view2 all frames, ...]; model
    # outputs are [frame1 all views, frame2 all views, ...].
    return [inputs[view * frames + frame] for frame in range(frames) for view in range(view_num)]


class MOTEvaluator(COCOEvaluator):
    def __init__(self, dataset_name, cfg, distributed, output_dir=None, *, use_fast_impl=True):
        super().__init__(dataset_name, cfg, distributed, output_dir=output_dir, use_fast_impl=use_fast_impl)
        self.dataset_name = dataset_name
        # VisionTrack train/test can contain close to a million detections.
        # Keeping every prediction in COCOEvaluator._predictions and then
        # flattening the complete list creates an avoidable host-memory peak.
        # Stream these two benchmark splits to JSONL and flatten them once
        # during final serialization instead. The model forward pass and JEV
        # trace are unchanged; the stream is only an evaluator spool.
        self._stream_visiontrack = dataset_name in {"VISION_train", "VISION_test"}
        self._stream_path = None
        self._stream_handle = None
        if self._stream_visiontrack:
            stream_dir = Path(output_dir or self._output_dir)
            stream_dir.mkdir(parents=True, exist_ok=True)
            self._stream_path = stream_dir / "predictions_stream.jsonl"
            self._stream_handle = self._stream_path.open("w", encoding="utf-8")


    def process(self, inputs, outputs):
        """
        custom_instances_to_coco_json
        """
        aligned_inputs = align_visiontrack_inputs_to_outputs(
            self.dataset_name, inputs, outputs
        )
        for input, output in zip(aligned_inputs, outputs):
            prediction = {"image_id": input["image_id"]}
            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)
                prediction["instances"] = custom_instances_to_coco_json(
                    instances, input["image_id"])
            if "proposals" in output:
                prediction["proposals"] = output["proposals"].to(self._cpu_device)
            if self._stream_visiontrack:
                assert self._stream_handle is not None
                # Only instances are consumed by _eval_predictions for
                # VisionTrack. Excluding proposals keeps the spool bounded
                # and matches the previous coco_instances_results payload.
                self._stream_handle.write(
                    json.dumps(
                        {"image_id": prediction["image_id"], "instances": prediction.get("instances", [])},
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                # COCOEvaluator.evaluate() uses a non-empty prediction list
                # as its dispatch signal. Keep one tiny sentinel so it calls
                # our streaming _eval_predictions implementation rather than
                # returning early; all real rows remain on disk.
                if not self._predictions:
                    self._predictions.append({"instances": []})
            else:
                self._predictions.append(prediction)


    def _eval_predictions(self, predictions, img_ids=None):
        """
        Evaluate predictions on the given tasks.
        Fill self._results with the metrics of the tasks.
        """
        assert img_ids is None
        if self._stream_visiontrack:
            assert self._stream_path is not None
            assert self._stream_handle is not None
            self._stream_handle.flush()
            self._stream_handle.close()
            self._stream_handle = None

            reverse_id_mapping = None
            if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
                reverse_id_mapping = {
                    v: k for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()
                }
            file_path = os.path.join(self._output_dir, "coco_instances_results.json")
            self._logger.info("Streaming results to %s", file_path)
            with self._stream_path.open("r", encoding="utf-8") as source, Path(file_path).open(
                "w", encoding="utf-8"
            ) as target:
                target.write("[")
                first = True
                for line in source:
                    record = json.loads(line)
                    for result in record.get("instances", []):
                        if reverse_id_mapping is not None:
                            category_id = result["category_id"]
                            if category_id not in reverse_id_mapping:
                                raise ValueError(
                                    f"unknown VisionTrack category_id={category_id}"
                                )
                            result["category_id"] = reverse_id_mapping[category_id]
                        if not first:
                            target.write(",")
                        json.dump(result, target, separators=(",", ":"))
                        first = False
                target.write("]")
            # Formal VisionTrack tracking/JEV metrics are computed from the
            # trace and aligned counterfactual data; COCO's in-memory
            # evaluator is not a required output for these two splits.
            self._results = {}
            return

        self._logger.info("Preparing results for COCO format ...")
        coco_results = list(itertools.chain(*[x["instances"] for x in predictions]))
        tasks = self._tasks or self._tasks_from_predictions(coco_results)

        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            reverse_id_mapping = {
                v: k for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()
            }
            for result in coco_results:
                category_id = result["category_id"]
                assert (
                    category_id in reverse_id_mapping
                ), "A prediction has category_id={}, which is not available in the dataset.".format(
                    category_id
                )
                result["category_id"] = reverse_id_mapping[category_id]

        file_path = os.path.join(self._output_dir, "coco_instances_results.json")
        self._logger.info("Saving results to {}".format(file_path))
        with PathManager.open(file_path, "w") as f:
            f.write(json.dumps(coco_results))
            f.flush()

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        self._logger.info(
            "Evaluating predictions with {} COCO API...".format(
                "unofficial" if self._use_fast_impl else "official"
            )
        )
        for task in sorted(tasks):
            coco_eval = (
                _evaluate_predictions_on_coco(
                    self._coco_api,
                    coco_results,
                    task,
                    kpt_oks_sigmas=self._kpt_oks_sigmas,
                )
                if len(coco_results) > 0
                else None  # cocoapi does not handle empty results very well
            )

            res = self._derive_coco_results(
                coco_eval, task, class_names=self._metadata.get("thing_classes")
            )
            self._results[task] = res

        # The legacy helper infers an MOT17/MOT20 year from the dataset name.
        # VisionTrack is a multi-view benchmark with its own downloaded GT and
        # must be evaluated by the explicit current-repository evaluator.  We
        # still persist COCO-style predictions above; silently routing them to
        # an unrelated MOT year would produce invalid metrics or fail on a
        # missing GT directory.
        if self.dataset_name not in {"VISION_train", "VISION_test"}:
            track_and_eval_mot(
                self._output_dir, self._coco_api.dataset, coco_results,
                dataset_name=self.dataset_name)
