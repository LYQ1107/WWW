from pathlib import Path

from worldmot.data.visiontrack import VisionTrack


def test_visiontrack_pair_adapter_reads_real_manifest():
    root = Path("/data/DATASETS/TRACKING/JDE/VisionTrack")
    if not (root / "annotations/train.json").exists():
        return
    ds = VisionTrack(root)
    assert len(ds.scene_ids()) == 24
    pair = next(ds.synchronized_pairs(8, limit=1))
    assert sorted(pair.images) == [1, 2]
    assert pair.images[1].frame_id == pair.images[2].frame_id
    assert ds.image_path(pair.images[1]).exists()
