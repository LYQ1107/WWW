"""Compare causal Ω-head validation audits against fresh native audits.

The comparison is keyed by immutable detector source ``(image_id, index)``.
Track IDs are local to each run, so no numeric ID is compared across runs;
correctness is evaluated against each run's own legal positive track set.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
from pathlib import Path


KNOWN_REASON = "known_one_to_one_IoU_ge_0.5"


def _events(path: Path):
    with gzip.open(path / "events.jsonl.gz", "rt") as stream:
        return {
            tuple(int(value) for value in row["key"][1:]): row
            for row in map(json.loads, stream)
        }


def _summary(path: Path):
    return json.loads((path / "summary.json").read_text())


def _eligible(row):
    return (
        row.get("query_label_reason") == KNOWN_REASON
        and not set(row.get("types", [])).intersection({"E", "F"})
        and bool(row.get("positive_tracks"))
    )


def _good(row, field):
    return row.get(field) in set(row.get("positive_tracks", []))


def _compare_scene(baseline: Path, head: Path):
    native = _events(baseline)
    adapted = _events(head)
    common = sorted(set(native).intersection(adapted))
    if set(native) != set(adapted):
        raise ValueError(f"source-key mismatch: {baseline} vs {head}")
    result = {"common_events": len(common), "summary": {"baseline": _summary(baseline), "head": _summary(head)}}
    for field, name in (("accepted_primary_track", "accepted_primary"), ("final_output_track", "final_output")):
        fixes = harms = eligible = 0
        for key in common:
            before, after = native[key], adapted[key]
            if not (_eligible(before) and _eligible(after)):
                continue
            eligible += 1
            before_good = _good(before, field)
            after_good = _good(after, field)
            fixes += int(not before_good and after_good)
            harms += int(before_good and not after_good)
        result[name] = {"eligible": eligible, "fixes": fixes, "harms": harms, "net": fixes - harms}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--head-root", type=Path, required=True)
    parser.add_argument("--scene-ids", nargs="+", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scenes = {}
    for scene in args.scene_ids:
        baseline = args.baseline_root / f"scene{scene}_full_logit_v1"
        candidates = sorted(args.head_root.glob(f"scene{scene}_omega_head_v1*"))
        complete = []
        for candidate in candidates:
            manifest_path = candidate / "manifest.json"
            if manifest_path.is_file() and json.loads(manifest_path.read_text()).get("status") == "COMPLETE_FULL_CAUSAL_HEAD_AUDIT":
                complete.append(candidate)
        if not complete:
            raise ValueError(f"head audit is incomplete: {args.head_root}/scene{scene}_omega_head_v1*")
        head = complete[-1]
        scenes[str(scene)] = _compare_scene(baseline, head)
    aggregate = {}
    for key in ("accepted_primary", "final_output"):
        values = [scenes[str(scene)][key] for scene in args.scene_ids]
        aggregate[key] = {
            "eligible": sum(value["eligible"] for value in values),
            "fixes": sum(value["fixes"] for value in values),
            "harms": sum(value["harms"] for value in values),
        }
        aggregate[key]["net"] = aggregate[key]["fixes"] - aggregate[key]["harms"]
    result = {
        "status": "COMPLETE_CAUSAL_HEAD_VALIDATION_COMPARISON",
        "scene_ids": args.scene_ids,
        "baseline_root": str(args.baseline_root),
        "head_root": str(args.head_root),
        "definition": "same source key; each run's local positive_tracks define identity correctness; E/F/unknown/missing-positive rows excluded",
        "scenes": scenes,
        "aggregate": aggregate,
        "integration_gate": {
            "requires_positive_primary_net": True,
            "accepted_primary_net_positive": aggregate["accepted_primary"]["net"] > 0,
            "decision": "CLOSED_BY_STAGE_A_PROTOCOL_GATE",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "aggregate": aggregate}, sort_keys=True))


if __name__ == "__main__":
    main()
