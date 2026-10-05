"""Compare two builders on the same bounded frozen-evidence subset.

This is a prerequisite for approving an implementation-optimized worktree.
It intentionally compares semantic records, not only checkpoint identity.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping


def _records(path: Path) -> list[Mapping[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        record.get("sequence"),
        int(record.get("frame", -1)),
        int(record.get("view", -1)),
        record.get("question_type"),
    )


def _utility_map(record: Mapping[str, Any]) -> dict[str, float]:
    return {
        str(action): float((outcome or {}).get("utility"))
        for action, outcome in (record.get("action_outcomes") or {}).items()
    }


def compare(left_path: Path, right_path: Path, tolerance: float = 1e-6) -> Mapping[str, Any]:
    left = _records(left_path)
    right = _records(right_path)
    left_by_key = {_key(record): record for record in left}
    right_by_key = {_key(record): record for record in right}
    keys_equal = left_by_key.keys() == right_by_key.keys()
    utility_mismatches = 0
    legal_action_mismatches = 0
    best_action_mismatches = 0
    state_mismatches = 0
    for key in left_by_key.keys() & right_by_key.keys():
        a = left_by_key[key]
        b = right_by_key[key]
        if a.get("legal_actions") != b.get("legal_actions"):
            legal_action_mismatches += 1
        if a.get("best_actions") != b.get("best_actions"):
            best_action_mismatches += 1
        if a.get("state_digest") != b.get("state_digest"):
            state_mismatches += 1
        left_utilities = _utility_map(a)
        right_utilities = _utility_map(b)
        if left_utilities.keys() != right_utilities.keys() or any(
            not math.isclose(left_utilities[name], right_utilities[name], abs_tol=tolerance, rel_tol=tolerance)
            for name in left_utilities
        ):
            utility_mismatches += 1
    status = bool(
        len(left) == len(right)
        and keys_equal
        and not legal_action_mismatches
        and not best_action_mismatches
        and not state_mismatches
        and not utility_mismatches
    )
    return {
        "status": "PASS" if status else "FAIL",
        "record_count_left": len(left),
        "record_count_right": len(right),
        "record_keys_equal": bool(keys_equal),
        "state_exact": state_mismatches == 0,
        "legal_actions_exact": legal_action_mismatches == 0,
        "best_actions_exact": best_action_mismatches == 0,
        "utility_tolerance": float(tolerance),
        "utility_mismatches": utility_mismatches,
        "state_mismatches": state_mismatches,
        "legal_action_mismatches": legal_action_mismatches,
        "best_action_mismatches": best_action_mismatches,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    args = parser.parse_args()
    report = compare(args.left.resolve(), args.right.resolve(), args.tolerance)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
