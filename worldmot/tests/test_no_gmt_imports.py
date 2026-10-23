from pathlib import Path


def test_optional_gmt_detector_is_the_only_native_gmt_dependency_boundary():
    root = Path(__file__).resolve().parents[1]
    offenders = []
    for path in (root / "worldmot").rglob("*.py"):
        # The explicitly requested detector-replacement route has one isolated,
        # lazy optional adapter. Original V1/V2 never import GMT themselves.
        if path.relative_to(root / 'worldmot').as_posix() == 'models/gmt_detector_adapter.py':
            continue
        text = path.read_text(encoding="utf-8")
        if "import gtr" in text or "from gtr" in text or "import detectron2" in text or "from detectron2" in text:
            offenders.append(str(path))
    assert not offenders, offenders
