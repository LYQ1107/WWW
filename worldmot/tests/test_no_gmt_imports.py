from pathlib import Path


def test_worldmot_source_has_no_gmt_or_gtr_imports():
    root = Path(__file__).resolve().parents[1]
    offenders = []
    for path in (root / "worldmot").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "import gtr" in text or "from gtr" in text or "import detectron2" in text or "from detectron2" in text:
            offenders.append(str(path))
    assert not offenders, offenders
