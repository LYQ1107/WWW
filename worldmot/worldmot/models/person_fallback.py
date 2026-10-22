"""Optional independent person detector interface.

This module never imports a GMT detector.  It is intentionally conservative:
unless an explicitly configured external detector is available it returns no
detections and records ``DEPENDENCY_BLOCKED`` rather than fabricating boxes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass
class PersonDetection:
    view_id: int
    bbox_xywh: tuple[float, float, float, float]
    score: float
    source: str = "fallback-person"


class PersonFallback:
    def __init__(self, backend: str | None = None, score_threshold: float = 0.35):
        self.backend = backend
        self.score_threshold = float(score_threshold)
        self.status = "DISABLED" if not backend else "NOT_INITIALIZED"
        self._model = None

    def infer(self, images: Mapping[int, np.ndarray]) -> list[PersonDetection]:
        if not self.backend:
            self.status = "DEPENDENCY_BLOCKED"
            return []
        # Loading arbitrary detector checkpoints is deliberately explicit.  A
        # future adapter can be added without changing the PanSt3R path.
        self.status = "DEPENDENCY_BLOCKED"
        return []
