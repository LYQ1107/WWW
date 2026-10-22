"""Image decoding kept separate from the annotation adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
from PIL import Image

from .visiontrack import ImageRecord, SynchronizedPair, VisionTrack


def load_pair(dataset: VisionTrack, pair: SynchronizedPair) -> Mapping[int, np.ndarray]:
    images = {}
    for view, record in pair.images.items():
        with Image.open(dataset.image_path(record)) as im:
            images[view] = np.asarray(im.convert("RGB"), dtype=np.uint8).copy()
    return images
