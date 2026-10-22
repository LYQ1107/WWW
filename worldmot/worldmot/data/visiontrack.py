"""Small, GMT-independent adapter for VisionTrack COCO-style annotations."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class ImageRecord:
    image_id: int
    video_id: int
    video_name: str
    view_id: int
    frame_id: int
    file_name: str
    width: int
    height: int


@dataclass(frozen=True)
class SynchronizedPair:
    scene_id: int
    scene_name: str
    frame_id: int
    images: Mapping[int, ImageRecord]


class VisionTrack:
    """Read-only VisionTrack indexing and causal pair iteration.

    This class intentionally reads only the public JSON annotation and image
    paths.  Ground truth is exposed through ``ground_truth`` for evaluation
    after inference; the runtime never calls that method.
    """

    def __init__(self, root: str | Path, annotation: str | None = None,
                 image_root: str | None = None, views: Sequence[int] = (1, 2),
                 split: str = "train", load_annotations: bool = True):
        if split not in ("train", "test"):
            raise ValueError("split must be train or test")
        self.split = split
        self.load_annotations = bool(load_annotations)
        annotation = annotation or f"annotations/{split}.json"
        image_root = image_root or split
        self.root = Path(root)
        self.annotation_path = self.root / annotation
        self.image_root = self.root / image_root
        self.views = tuple(int(v) for v in views)
        with self.annotation_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        self.categories = data.get("categories", [])
        self.videos = {int(v["id"]): v for v in data.get("videos", [])}
        self.images: Dict[int, ImageRecord] = {}
        self._by_scene_view_frame: Dict[Tuple[int, int, int], ImageRecord] = {}
        for x in data.get("images", []):
            scene_id = int(x["video_id"])
            view_id = int(x["view_id"])
            rec = ImageRecord(
                image_id=int(x["id"]), video_id=scene_id,
                video_name=str(self.videos.get(scene_id, {}).get("file_name", scene_id)),
                view_id=view_id, frame_id=int(x["frame_id"]),
                file_name=str(x["file_name"]), width=int(x.get("width", 0)),
                height=int(x.get("height", 0)),
            )
            self.images[rec.image_id] = rec
            self._by_scene_view_frame[(scene_id, view_id, rec.frame_id)] = rec
        self._annotations: Dict[int, List[dict]] = {}
        for ann in data.get("annotations", []) if self.load_annotations else []:
            self._annotations.setdefault(int(ann["image_id"]), []).append(ann)

    def scene_ids(self) -> List[int]:
        return sorted(self.videos)

    def scene_name(self, scene_id: int) -> str:
        return str(self.videos[int(scene_id)]["file_name"])

    def frames(self, scene_id: int, require_all_views: bool = True) -> List[int]:
        scene_id = int(scene_id)
        by_frame: Dict[int, set] = {}
        for (sid, view, frame), _ in self._by_scene_view_frame.items():
            if sid == scene_id and view in self.views:
                by_frame.setdefault(frame, set()).add(view)
        if require_all_views:
            return sorted(f for f, views in by_frame.items() if all(v in views for v in self.views))
        return sorted(by_frame)

    def synchronized_pairs(self, scene_id: int, start_frame: Optional[int] = None,
                           limit: Optional[int] = None) -> Iterator[SynchronizedPair]:
        frames = self.frames(scene_id, require_all_views=True)
        if start_frame is not None:
            frames = [f for f in frames if f >= int(start_frame)]
        if limit is not None:
            frames = frames[: int(limit)]
        name = self.scene_name(scene_id)
        for frame in frames:
            images = {
                v: self._by_scene_view_frame[(int(scene_id), v, frame)] for v in self.views
            }
            yield SynchronizedPair(int(scene_id), name, frame, images)

    def image_path(self, image: ImageRecord) -> Path:
        path = self.image_root / image.file_name
        if not path.exists():
            raise FileNotFoundError(f"VisionTrack image is missing: {path}")
        return path

    def ground_truth(self, image_id: int) -> List[dict]:
        """Return copied annotations for post-hoc evaluation only."""
        if not self.load_annotations:
            raise RuntimeError("inference metadata-only adapter forbids GT access")
        return [dict(x) for x in self._annotations.get(int(image_id), [])]

    def ground_truth_pair(self, pair: SynchronizedPair) -> Dict[int, List[dict]]:
        return {v: self.ground_truth(rec.image_id) for v, rec in pair.images.items()}

    def audit(self, scene_ids: Optional[Iterable[int]] = None) -> dict:
        scenes = self.scene_ids() if scene_ids is None else [int(x) for x in scene_ids]
        out = {"root": str(self.root), "annotation": str(self.annotation_path), "scenes": []}
        for sid in scenes:
            frames = self.frames(sid)
            missing = []
            if frames:
                sample = next(self.synchronized_pairs(sid, limit=1))
                for v, rec in sample.images.items():
                    if not (self.image_root / rec.file_name).exists():
                        missing.append(v)
            out["scenes"].append({
                "scene_id": sid, "scene_name": self.scene_name(sid),
                "synchronized_frames": len(frames),
                "first_frame": frames[0] if frames else None,
                "last_frame": frames[-1] if frames else None,
                "missing_sample_views": missing,
            })
        return out
