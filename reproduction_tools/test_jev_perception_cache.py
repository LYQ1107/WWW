"""CPU contract tests for the immutable perception cache."""

from pathlib import Path
import tempfile

import torch
from detectron2.structures import Boxes, Instances

from gtr.modeling.jev_perception_cache import (
    CACHE_VERSION,
    FrozenPerceptionCache,
    FrozenPerceptionCacheWriter,
)


def main():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        instances = Instances((32, 40))
        instances.pred_boxes = Boxes(torch.tensor([[1.0, 2.0, 10.0, 12.0]]))
        instances.scores = torch.tensor([0.9])
        instances.pred_classes = torch.tensor([3])
        instances.reid_features = torch.tensor([[1.0, 2.0, 3.0]])
        with FrozenPerceptionCacheWriter(root) as writer:
            record = writer.write(
                video_id=4,
                frame=5,
                view=1,
                instances=instances,
                metadata={"file_name": "fixture.jpg"},
            )
        cache = FrozenPerceptionCache(root)
        payload = cache.load(4, 5, 1)
        assert record["cache_version"] == CACHE_VERSION
        assert torch.equal(payload["pred_boxes"], instances.pred_boxes.tensor)
        assert torch.equal(payload["reid_features"], instances.reid_features)
        assert payload["metadata"]["file_name"] == "fixture.jpg"
    print("JEV perception-cache invariants: PASS")


if __name__ == "__main__":
    main()

