#!/usr/bin/env python3
"""Create a comparable cross-view metric table from isolated reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def read_report(path: Path) -> Dict[str, Any]:
    report = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
    rows = {}
    for name, values in report["reports"].items():
        rows[name] = {
            "time_axis": values["time_axis"],
            "CVIDF1": float(values["CVIDF1"]),
            "CVMA": float(values["CVMA"]),
            "identity_idf1_raw": float(values["identity_idf1_raw"]),
            "clear_mota_raw": float(values["clear_mota_raw"]),
        }
    return {
        "path": str(path),
        "allow_duplicate_gt": bool(report.get("allow_duplicate_gt", False)),
        "reports": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("policies", nargs="+", help="NAME=REPORT_DIRECTORY")
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite {args.output}")
    entries: Dict[str, Any] = {}
    for item in args.policies:
        if "=" not in item:
            raise ValueError(f"policy must be NAME=DIR: {item}")
        name, directory = item.split("=", 1)
        if not name:
            raise ValueError("empty policy name")
        entries[name] = read_report(Path(directory).resolve())
    duplicate_modes = {entry["allow_duplicate_gt"] for entry in entries.values()}
    payload = {
        "status": "PASS",
        "policies": entries,
        "comparison_warning": (
            "all policies must use the same frozen GMT checkpoint, config, GT manifest, "
            "and duplicate-GT evaluation mode"
        ),
        "duplicate_gt_modes_present": len(duplicate_modes),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

