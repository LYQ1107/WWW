"""GMT association-transformer adapter for cached-perception v2 replay.

The adapter reconstructs Detectron2 ``Instances`` from immutable cache
payloads, appends the branch-local historical assignments, and calls the
repository's association transformer once for the current proposal. It does
not own tracker state or choose actions.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Mapping, Sequence

import torch
from detectron2.structures import Boxes, Instances


class GMTAssociationTransformerAdapter:
    """Callable backend compatible with ``CachedPerceptionMutableAssociationV2``."""

    formal_gmt_association_adapter = True

    def __init__(self, model, *, view_num: int = 1, history_limit: int = 16):
        self.model = model
        self.view_num = max(1, int(view_num))
        self.history_limit = max(1, int(history_limit))
        # Cache only the device-side immutable tensors.  Formal replay calls
        # the adapter repeatedly for the same current/future payload while
        # exploring action branches; rebuilding these tensors dominated the
        # host-side overhead.  Keep a bounded LRU so a long shard cannot
        # consume the whole GPU with cached perceptions.
        self._payload_tensors = OrderedDict()
        self._payload_cache_limit = max(512, self.history_limit * 4)

    @staticmethod
    def _payload_key(payload: Mapping[str, object]):
        try:
            return (
                int(payload["video_id"]),
                int(payload["frame"]),
                int(payload["view"]),
            )
        except (KeyError, TypeError, ValueError):
            # Keep the lightweight adapter test and any legacy in-memory
            # payloads valid when they do not carry cache coordinates.
            return ("object", id(payload))

    def _instances(self, payload: Mapping[str, object], device: torch.device) -> Instances:
        key = self._payload_key(payload)
        cached = self._payload_tensors.get(key)
        if cached is None:
            image_size = tuple(int(value) for value in payload["image_size"])
            boxes = torch.as_tensor(
                payload["pred_boxes"], dtype=torch.float32, device=device
            )
            reid_features = torch.as_tensor(
                payload["reid_features"], dtype=torch.float32, device=device
            )
            cached = (image_size, boxes, reid_features)
            self._payload_tensors[key] = cached
            while len(self._payload_tensors) > self._payload_cache_limit:
                self._payload_tensors.popitem(last=False)
        else:
            self._payload_tensors.move_to_end(key)

        image_size, boxes, reid_features = cached
        instance = Instances(image_size)
        instance.pred_boxes = Boxes(boxes)
        instance.reid_features = reid_features
        return instance

    def __call__(self, perception, track_ids: Sequence[int], state) -> torch.Tensor:
        device = next(self.model.parameters()).device
        current = self._instances(perception, device)
        history = list(state.association_history[-self.history_limit :])
        if not history:
            return torch.zeros(
                (len(current), len(track_ids)), dtype=torch.float32, device=device
            )

        instances = []
        previous_ids = []
        for item in history:
            historical = self._instances(item["perception"], device)
            assignments = dict(item["assignments"])
            historical.track_ids = torch.tensor(
                [int(assignments[row]) for row in range(len(historical))],
                dtype=torch.long,
                device=device,
            )
            instances.append(historical)
            previous_ids.extend(historical.track_ids.tolist())
        instances.append(current)
        if not previous_ids:
            return torch.zeros(
                (len(current), len(track_ids)), dtype=torch.float32, device=device
            )

        traj_ids = torch.tensor(previous_ids, dtype=torch.long, device=device)
        reid_features = torch.cat(
            [instance.reid_features for instance in instances], dim=0
        )[None]
        with torch.no_grad():
            outputs, _, _, _, _, _ = self.model.roi_heads._forward_transformer(
                instances,
                reid_features,
                self.view_num,
                len(instances) - 1,
                None,
                None,
                traj_ids,
            )
            active = self.model.roi_heads._activate_asso(outputs)[0]
        current_count = len(current)
        historical_count = len(previous_ids)
        if tuple(active.shape) != (current_count, historical_count):
            raise ValueError(
                "GMT association transformer shape mismatch: "
                f"{tuple(active.shape)} != {(current_count, historical_count)}"
            )
        unique = torch.tensor(track_ids, dtype=torch.long, device=device)
        id_inds = (unique[None, :] == traj_ids[:, None]).float()
        return torch.mm(active, id_inds).detach().cpu()
