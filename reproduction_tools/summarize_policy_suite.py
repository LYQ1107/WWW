"""Create a compact, comparable table from isolated VisionTrack evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def scalar(value):
    if isinstance(value, list):
        return sum(float(item) for item in value) / len(value) if value else 0.0
    return float(value)


def read_policy(path: Path) -> Dict[str, Any]:
    report = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
    combined = report["combined_metrics"]
    return {
        "path": str(path),
        "HOTA": scalar(combined["HOTA"]["HOTA"]),
        "HOTA_0": scalar(combined["HOTA"]["HOTA(0)"]),
        "DetA": scalar(combined["HOTA"]["DetA"]),
        "AssA": scalar(combined["HOTA"]["AssA"]),
        "IDF1": scalar(combined["Identity"]["IDF1"]),
        "IDP": scalar(combined["Identity"]["IDP"]),
        "IDR": scalar(combined["Identity"]["IDR"]),
        "MOTA": scalar(combined["CLEAR"]["MOTA"]),
        "IDSW": scalar(combined["CLEAR"]["IDSW"]),
        "Frag": scalar(combined["CLEAR"]["Frag"]),
        "Dets": scalar(combined["Count"]["Dets"]),
        "GT_Dets": scalar(combined["Count"]["GT_Dets"]),
        "allow_duplicate_gt": bool(report.get("allow_duplicate_gt", False)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("policies", nargs="+", help="NAME=EVALUATION_DIRECTORY")
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite {args.output}")
    entries = {}
    for item in args.policies:
        if "=" not in item:
            raise ValueError(f"policy must be NAME=DIR: {item}")
        name, directory = item.split("=", 1)
        if not name:
            raise ValueError("empty policy name")
        entries[name] = read_policy(Path(directory).resolve())
    duplicate_modes = {entry["allow_duplicate_gt"] for entry in entries.values()}
    report = {
        "status": "PASS",
        "policies": entries,
        "comparison_warning": (
            "all policies must use the same frozen GMT checkpoint, config, GT manifest, "
            "and duplicate-GT evaluation mode"
        ),
        "duplicate_gt_modes_present": len(duplicate_modes),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
