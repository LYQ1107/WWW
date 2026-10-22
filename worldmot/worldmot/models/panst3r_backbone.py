"""Thin, auditable adapter around the official PanSt3R source tree.

The adapter imports only the official PanSt3R/MUSt3R packages selected by
the caller.  It does not import any GMT package or tracker state.
"""

from __future__ import annotations

from pathlib import Path
import sys
import time
from typing import Mapping

import numpy as np
from PIL import Image

from .geometry_interface import GeometryBackend, GeometryFrame
from .panoptic_instances import _to_numpy
from ..geometry.camera_model import CameraModel


class PanSt3RDependencyError(RuntimeError):
    pass


class PanSt3RBackend(GeometryBackend):
    def __init__(self, panst3r_root: str | Path, must3r_root: str | Path,
                 weights: str | Path, device: str = "cuda:0", image_size: int = 512,
                 classes: tuple[str, ...] = ("person",), num_keyframes: int = 2,
                 use_retrieval: bool = False, postprocess: str = "standard_v2",
                 max_bs: int = 1, amp: str | bool = False, verbose: bool = False):
        self.panst3r_root = Path(panst3r_root).resolve()
        self.must3r_root = Path(must3r_root).resolve()
        self.weights = Path(weights).resolve()
        self.device_name = str(device)
        self.image_size = int(image_size)
        self.classes = tuple(classes)
        self.num_keyframes = int(num_keyframes)
        self.use_retrieval = bool(use_retrieval)
        self.postprocess_name = str(postprocess)
        self.max_bs = int(max_bs) if max_bs else None
        self.amp = amp
        self.verbose = bool(verbose)
        if not self.weights.exists():
            raise FileNotFoundError(f"PanSt3R checkpoint does not exist: {self.weights}")
        self._install_paths()
        try:
            import torch
            self.torch = torch
            if not torch.cuda.is_available() and self.device_name.startswith("cuda"):
                raise PanSt3RDependencyError("CUDA requested but torch.cuda.is_available() is false")
            self.device = torch.device(self.device_name)
            from panst3r.panst3r import PanSt3R
            self.PanSt3R = PanSt3R
            from must3r.tools.image import get_resize_function
            from must3r.model import get_pointmaps_activation
            from must3r.engine.inference import postprocess as must3r_postprocess
            from panst3r.engine.postprocess import panoptic_inference_v2, panoptic_inference_v1
            self.get_resize_function = get_resize_function
            self.get_pointmaps_activation = get_pointmaps_activation
            self.must3r_postprocess = must3r_postprocess
            self.panoptic_inference_v2 = panoptic_inference_v2
            self.panoptic_inference_v1 = panoptic_inference_v1
        except Exception as exc:
            raise PanSt3RDependencyError(f"unable to import official PanSt3R/MUSt3R: {exc}") from exc
        try:
            self.model = self.PanSt3R.from_checkpoint(str(self.weights))
            self.model.eval().to(self.device)
            # This is a real text vocabulary call.  It may download the
            # checkpoint's SigLIP text asset; failure is surfaced explicitly.
            self.model.set_vocab(list(self.classes), device=self.device)
            self.pointmaps_activation = self.get_pointmaps_activation(self.model.must3r_decoder, verbose=False)
        except Exception as exc:
            raise PanSt3RDependencyError(f"unable to load PanSt3R checkpoint/vocabulary: {exc}") from exc

    def _install_paths(self) -> None:
        candidates = [
            str(self.panst3r_root / "src"), str(self.panst3r_root),
            str(self.must3r_root), str(self.must3r_root / "dust3r"),
        ]
        for p in reversed(candidates):
            if p not in sys.path:
                sys.path.insert(0, p)

    def _preprocess(self, image: np.ndarray):
        import torchvision.transforms.functional as TF
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"expected HxWx3 RGB image, got {image.shape}")
        h, w = image.shape[:2]
        pil = Image.fromarray(image.astype(np.uint8), mode="RGB")
        resize_fn, _, to_orig = self.get_resize_function(self.image_size, self.model.must3r_encoder.patch_size, h, w)
        resized = resize_fn(pil)
        tensor = TF.to_tensor(resized)
        tensor = TF.normalize(tensor, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        model_to_image = np.asarray(to_orig, dtype=np.float64)
        image_to_model = np.linalg.inv(model_to_image)
        true_shape = np.asarray([tensor.shape[-2], tensor.shape[-1]], dtype=np.int32)
        return tensor, true_shape, image_to_model, model_to_image

    @staticmethod
    def _one(x):
        if hasattr(x, "detach"):
            x = x.detach().cpu()
        while getattr(x, "ndim", 0) > 0 and x.shape[0] == 1:
            x = x[0]
        return x

    def _normalize_queries(self, out_queries) -> np.ndarray:
        if isinstance(out_queries, (list, tuple)):
            # Two keyframes are represented as one tensor in v2.  If a list is
            # returned, concatenate only along the view axis then select the
            # first shared query set.
            vals = [_to_numpy(self._one(x)) for x in out_queries]
            out_queries = vals[0] if vals else np.empty((0, 0), np.float32)
        q = _to_numpy(out_queries)
        # MaskTransformer emits either BxQxC or QxNxC depending on the
        # multi-aspect-ratio path.  Keep the Q dimension (query ids in the
        # panoptic result), and average only batch/view copies.
        if q.ndim == 3:
            if q.shape[0] <= 4 and q.shape[1] > 4:
                q = q.mean(axis=0)       # B x Q x C
            elif q.shape[1] <= 4 and q.shape[0] > 4:
                q = q.mean(axis=1)       # Q x N x C
            else:
                q = q[0]
        elif q.ndim > 3:
            q = q.reshape(-1, q.shape[-2], q.shape[-1]).mean(axis=0)
        if q.ndim != 2:
            q = q.reshape(q.shape[-2], q.shape[-1])
        return q.astype(np.float32, copy=False)

    def _postprocess_pointmap(self, raw):
        raw = raw.to(self.device) if hasattr(raw, "to") else raw
        return self.must3r_postprocess(raw, pointmaps_activation=self.pointmaps_activation, compute_cam=True)

    @staticmethod
    def _as_view(raw, index: int):
        if isinstance(raw, (list, tuple)):
            return raw[index]
        if getattr(raw, "ndim", 0) >= 1:
            return raw[index]
        return raw

    @staticmethod
    def _cpu_array(x):
        if hasattr(x, "detach"):
            x = x.detach().cpu().numpy()
        return np.asarray(x)

    def reconstruct(self, images: Mapping[int, np.ndarray], timestamp: int) -> GeometryFrame:
        # Keeping the timing boundaries explicit also makes this method usable
        # in CPU-only audits without importing torch at module import time.
        t0 = time.perf_counter()
        views = sorted(images)
        if len(views) != 2:
            raise ValueError(f"WorldMOT v0 requires exactly two synchronized views, got {views}")
        tensors, shapes, image_to_model, model_to_image = [], [], {}, {}
        for view in views:
            tensor, true_shape, i2m, m2i = self._preprocess(np.asarray(images[view]))
            tensors.append(tensor)
            shapes.append(true_shape)
            image_to_model[int(view)] = i2m
            model_to_image[int(view)] = m2i
        torch = self.torch
        batch = [x.to(self.device, non_blocking=True) for x in tensors]
        true_shape = torch.as_tensor(np.stack(shapes), dtype=torch.int32, device=self.device)
        t_model_start = time.perf_counter()
        with torch.no_grad():
            pointmaps_raw, panout = self.model.forward_inference_multi_ar(
                batch, true_shape, list(self.classes), num_keyframes=self.num_keyframes,
                use_retrieval=self.use_retrieval, max_bs=self.max_bs, outdevice="cpu", amp=self.amp)
        t_model = time.perf_counter() - t_model_start
        t_pan_start = time.perf_counter()
        size = true_shape.detach().cpu().numpy()
        if self.postprocess_name == "standard_v1":
            pan_preds = self.panoptic_inference_v1(panout["pred_logits"], panout["pred_masks"], size,
                                                    label_mode=self.model.panoptic_decoder.label_mode, device="cpu", multi_ar=True)
        elif self.postprocess_name == "standard_v2":
            pan_preds = self.panoptic_inference_v2(panout["pred_logits"], panout["pred_masks"], size,
                                                    label_mode=self.model.panoptic_decoder.label_mode, device="cpu", multi_ar=True)
        else:
            raise ValueError(f"unsupported non-QUBO postprocess: {self.postprocess_name}")
        # Official v2 returns one batch item whose pan/conf are lists over views.
        batch_pred = pan_preds[0]
        query_features = self._normalize_queries(panout.get("out_queries"))
        t_panoptic = time.perf_counter() - t_pan_start
        t_post_start = time.perf_counter()
        pointmaps, confs, cameras, panoptic = {}, {}, {}, {}
        for idx, view in enumerate(views):
            processed = self._postprocess_pointmap(self._as_view(pointmaps_raw, idx))
            pts = self._cpu_array(processed["pts3d"])
            conf = self._cpu_array(processed.get("conf", np.ones(pts.shape[:2])))
            while pts.ndim > 3 and pts.shape[0] == 1: pts = pts[0]
            while conf.ndim > 2 and conf.shape[0] == 1: conf = conf[0]
            pointmaps[int(view)] = pts.astype(np.float64)
            confs[int(view)] = conf.astype(np.float64)
            focal = float(np.asarray(self._cpu_array(processed.get("focal", 1.0))).reshape(-1)[0])
            h, w = pts.shape[:2]
            c2w = self._cpu_array(processed.get("c2w", np.eye(4))).reshape(-1, 4, 4)[0]
            orig_h, orig_w = np.asarray(images[view]).shape[:2]
            K = np.array([[focal, 0.0, w / 2.0], [0.0, focal, h / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)
            cameras[int(view)] = CameraModel(K, c2w, (orig_h, orig_w), (h, w), image_to_model[int(view)], name=str(view))
            segments = batch_pred.get("segments_info", [])
            pan = batch_pred.get("pan", [])[idx]
            panoptic[int(view)] = {"pan": pan, "segments_info": [dict(s) for s in segments]}
        t_pointmap_post = time.perf_counter() - t_post_start
        t_total = time.perf_counter() - t0
        finite = all(np.isfinite(x).all() for x in pointmaps.values())
        segment_count = sum(len(x["segments_info"]) for x in panoptic.values())
        status = "OK" if finite else "INVALID_WORLD_GEOMETRY"
        if segment_count == 0:
            status = "NO_PERSON_DETECTIONS"
        return GeometryFrame(
            timestamp=int(timestamp), images=images, pointmaps=pointmaps, confidences=confs,
            panoptic=panoptic, query_features=query_features, cameras=cameras,
            image_to_model=image_to_model, model_to_image=model_to_image, status=status,
            diagnostics={"total_s": float(t_total), "preprocess_s": float(max(0.0, t_model_start - t0)),
                         "model_only_s": float(t_model), "panoptic_s": float(t_panoptic),
                         "pointmap_postprocess_s": float(t_pointmap_post),
                         "model_total_s": float(t_model), "raw_pointmap_shapes": [list(np.shape(x)) for x in pointmaps_raw],
                         "query_shape": list(query_features.shape), "segments": int(segment_count),
                         "finite": bool(finite)},
        )
