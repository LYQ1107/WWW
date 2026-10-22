"""VGGT-Ω native 3-D person-query interface and training gate.

The public VGGT-Det reference head is a ScanNet head with 18 indoor object
classes and no ``person`` class.  This module records that fact explicitly and
refuses to run random or mismatched weights.  A future legally trained
VisionTrack person head can implement ``predict`` behind the same interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


PRETRAINED_PERSON_3D_HEAD_AVAILABLE = "PRETRAINED_PERSON_3D_HEAD_AVAILABLE"
PERSON_3D_HEAD_REQUIRES_TRAINING = "PERSON_3D_HEAD_REQUIRES_TRAINING"

VGGT_DET_REFERENCE = "https://github.com/yangcaoai/VGGT-Det-CVPR2026"
VGGT_DET_COMMIT = "73ea4cf94a05fa50279ee0b6095fb06f6d042dac"
VGGT_DET_CONFIG = "projects/VGGTDet/config/vggtdet_scannet.py"
VGGT_DET_HEAD = "projects/VGGTDet/vggtdet/vggt_head.py"
SCANNET_CLASSES = (
    "cabinet", "bed", "chair", "sofa", "table", "door", "window", "bookshelf",
    "picture", "counter", "desk", "curtain", "refrigerator", "showercurtrain",
    "toilet", "sink", "bathtub", "garbagebin",
)


@dataclass
class Omega3DQueryStatus:
    state: str
    reason: str
    reference_repository: str = VGGT_DET_REFERENCE
    reference_commit: str = VGGT_DET_COMMIT
    reference_config: str = VGGT_DET_CONFIG
    reference_head: str = VGGT_DET_HEAD
    classes: tuple[str, ...] = SCANNET_CLASSES
    checkpoint: str | None = None
    trained_for_person: bool = False

    def to_dict(self) -> dict:
        return {
            "state": self.state, "reason": self.reason,
            "reference_repository": self.reference_repository,
            "reference_commit": self.reference_commit,
            "reference_config": self.reference_config,
            "reference_head": self.reference_head,
            "reference_classes": list(self.classes),
            "checkpoint": self.checkpoint,
            "trained_for_person": self.trained_for_person,
        }


class Omega3DQueryDetector:
    """A safe placeholder until a person-supervised 3-D head exists."""

    def __init__(self, checkpoint: str | Path | None = None,
                 config: str | Path | None = None,
                 trained_for_person: bool = False, **_: Any):
        self.checkpoint = str(Path(checkpoint).resolve()) if checkpoint else None
        self.config = str(Path(config).resolve()) if config else VGGT_DET_CONFIG
        if trained_for_person and self.checkpoint and Path(self.checkpoint).is_file():
            # The interface is ready for an explicitly trained adapter, but no
            # generic loader is allowed to reinterpret ScanNet weights as a
            # person detector.
            self.status = Omega3DQueryStatus(
                PRETRAINED_PERSON_3D_HEAD_AVAILABLE,
                "checkpoint is explicitly declared VisionTrack-person trained",
                checkpoint=self.checkpoint, trained_for_person=True,
            )
        else:
            self.status = Omega3DQueryStatus(
                PERSON_3D_HEAD_REQUIRES_TRAINING,
                "public VGGT-Det checkpoint/config is ScanNet-only (18 classes; no person class); "
                "a VisionTrack person head needs registered supervised training",
                checkpoint=self.checkpoint, trained_for_person=False,
            )
        self.model = None

    @property
    def state(self) -> str:
        return self.status.state

    def manifest(self) -> dict:
        return self.status.to_dict()

    def predict(self, geometry_frame):
        if self.state != PRETRAINED_PERSON_3D_HEAD_AVAILABLE or self.model is None:
            raise RuntimeError(f"Omega 3-D person head unavailable: {self.status.reason}")
        return self.model.predict(geometry_frame)

