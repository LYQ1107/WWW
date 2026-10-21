"""Frozen cached Ω evidence plus a bounded causal residual for GMT audits.

The cache contains only image/detector keyed frozen features.  This module does
not read GT and never creates a track history; it uses the live ``Instances``
references supplied by GMT and adjusts every association decoder layer before
the original Hungarian/threshold code runs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


class CachedOmegaResidual:
    def __init__(self, checkpoint: Path, feature_dir: Path, scene_id: int, image_meta: dict[int, dict], device: str = "cuda"):
        from geo_tools.train_omega_full24_heads import BoundedResidual

        self.checkpoint_path = Path(checkpoint)
        state = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        self.input_indices = np.asarray(state["input_indices"], dtype=np.int64)
        self.mean = torch.as_tensor(state["normalizer_mean"], dtype=torch.float32, device=device)
        self.std = torch.as_tensor(state["normalizer_std"], dtype=torch.float32, device=device)
        self.head = BoundedResidual(len(self.input_indices), hidden_dim=int(state.get("hidden_dim", 64)), max_residual=float(state.get("max_residual", 0.25))).to(device)
        self.head.load_state_dict(state["state_dict"], strict=True)
        self.head.eval().requires_grad_(False)
        self.device = torch.device(device)
        self.image_meta = image_meta
        folder = Path(feature_dir) / f"scene{scene_id}_omega512_v1"
        manifest = json.loads((folder / "manifest.json").read_text())
        if manifest.get("status") != "COMPLETE_OMEGA512_FEATURE_EXTRACTION":
            raise ValueError(f"incomplete Ω feature store for scene {scene_id}")
        archive = np.load(folder / "features.npz")
        keys = np.asarray(archive["source_keys"], dtype=np.int64)
        values = np.asarray(archive["features"], dtype=np.float32)
        values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8)
        self.index = {tuple(key): row for row, key in enumerate(keys.tolist())}
        self.values = values
        self.calls = 0
        self.adjusted_queries = 0
        self.missing_sources = 0

    def _candidate_matrix(self, query_sources, references, native_scores, query_camera):
        refs_by_track = {}
        for column, ref in enumerate(references):
            refs_by_track.setdefault(int(ref["track"]), []).append((column, tuple(int(value) for value in ref["source"])))
        track_ids = sorted(refs_by_track)
        if not track_ids:
            return None
        q_vectors = []
        for source in query_sources:
            source = tuple(int(value) for value in source)
            if source not in self.index:
                self.missing_sources += 1
                q_vectors.append(None)
            else:
                q_vectors.append(self.values[self.index[source]])
        features = []
        for q_vector, score_row in zip(q_vectors, native_scores):
            if q_vector is None:
                features.append(None)
                continue
            order = np.argsort(-np.asarray(score_row), kind="stable")
            rank_by_track = {track: rank for rank, track in enumerate(np.asarray(track_ids)[order])}
            sorted_scores = np.asarray(score_row)[order]
            margin = float((sorted_scores[0] - sorted_scores[1]) / (abs(sorted_scores[0]) + 1e-8)) if len(sorted_scores) > 1 else 1.0
            row_features = []
            for track_index, track in enumerate(track_ids):
                refs = refs_by_track[track]
                source_values = [self.values[self.index[source]] for _, source in refs if source in self.index]
                if not source_values:
                    self.missing_sources += 1
                    row_features.append(None)
                    continue
                similarity = np.asarray(source_values, dtype=np.float32) @ q_vector
                cameras = np.asarray([int(self.image_meta[source[0]]["view_id"]) for _, source in refs if source in self.index])
                top_count = min(2, len(similarity))
                row_features.append([
                    float(score_row[track_index]),
                    float(rank_by_track[track]),
                    float(np.max(similarity)),
                    float(np.mean(similarity)),
                    float(np.mean(np.sort(similarity)[-top_count:])),
                    float(np.log1p(len(refs))),
                    float(np.mean(cameras != query_camera)),
                    float(np.mean(cameras == query_camera)),
                    margin,
                    float(len(track_ids)),
                ])
            features.append(row_features)
        return track_ids, features

    @torch.inference_mode()
    def adjust(self, result, instances, k, activate, source_context=None):
        if not result or not result[0]:
            return result
        layers = result[0]
        base_logits = layers[0]
        refs = [value for index, value in enumerate(instances) if index != k]
        query = instances[k]
        if source_context is None:
            query_sources = query.decision_trace_source.detach().cpu().tolist()
            reference_sources = []
            for value in refs:
                reference_sources.extend(value.decision_trace_source.detach().cpu().tolist())
        else:
            query_sources, reference_sources = source_context
        reference_tracks = []
        for value in refs:
            reference_tracks.extend(value.track_ids.detach().cpu().tolist())
        if not reference_tracks:
            return result
        ids = torch.as_tensor(reference_tracks, dtype=torch.long, device=base_logits.device)
        unique = ids.unique(sorted=True)
        probabilities = torch.cat(activate(base_logits.split([len(value) for value in refs], dim=1)), dim=1)
        scores = probabilities @ (ids[:, None] == unique[None, :]).float()
        query_camera = int(self.image_meta[int(query_sources[0][0])]["view_id"]) if query_sources else -1
        candidate = self._candidate_matrix(query_sources, [dict(source=source, track=track) for source, track in zip(reference_sources, reference_tracks)], scores.detach().float().cpu().tolist(), query_camera)
        if candidate is None:
            return result
        track_ids, rows = candidate
        deltas = torch.zeros((len(query_sources), len(track_ids)), dtype=base_logits.dtype, device=base_logits.device)
        for row_index, row_features in enumerate(rows):
            valid = [value is not None for value in row_features]
            if any(valid):
                values = torch.as_tensor([value if value is not None else [0.0] * 10 for value in row_features], dtype=torch.float32, device=base_logits.device)
                normalized = (values - self.mean) / self.std
                residual = self.head(normalized[:, self.input_indices]).to(deltas.dtype)
                residual[~torch.as_tensor(valid, device=deltas.device)] = 0
                deltas[row_index] = residual
        adjusted = []
        for layer in layers:
            modified = layer.clone()
            for track_index, track in enumerate(track_ids):
                columns = torch.nonzero(ids == int(track), as_tuple=False).flatten()
                if len(columns):
                    modified[:, columns] += deltas[:, track_index : track_index + 1]
            adjusted.append(modified)
        self.calls += 1
        self.adjusted_queries += len(query_sources)
        return (adjusted, *result[1:])
