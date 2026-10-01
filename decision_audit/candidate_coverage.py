#!/usr/bin/env python3
"""Audit Top-K active-ID candidate coverage on the frozen JEV-train split."""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import torch


K_VALUES = (1, 2, 4, 8, 16)


def ranked_ids(row: dict) -> list[int]:
    # The released Hungarian rule uses raw summed trajectory evidence.  Keep
    # the same score for candidate ranking and use numeric ID only as a stable
    # tie-breaker for this audit.
    return [
        int(candidate["global_id"])
        for candidate in sorted(
            row["candidates"],
            key=lambda candidate: (-float(candidate["traj_score"]), int(candidate["global_id"])),
        )
    ]


def load_rows(root: Path, split: dict) -> dict[str, list[dict]]:
    train_scenes = set(split["scenes_by_split"]["JEV-train"])
    rows_by_scene: dict[str, list[dict]] = {}
    for path in sorted((root / "labeled_records").glob("*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        scene = str(payload["scene"])
        if scene in train_scenes:
            rows_by_scene[scene] = list(payload["records"])
    return rows_by_scene


def summarize(rows_by_scene: dict[str, list[dict]], k: int) -> dict:
    totals = collections.Counter()
    per_scene = {}
    for scene, rows in sorted(rows_by_scene.items()):
        scene_counts = collections.Counter()
        for row in rows:
            if row.get("gt_id_for_supervision") is None:
                continue
            if row.get("target_type") != "EXISTING":
                continue
            totals["correct_existing"] += 1
            scene_counts["correct_existing"] += 1
            ids = ranked_ids(row)
            target = int(row["target_global_id"])
            covered = target in ids[:k]
            if covered:
                totals["covered"] += 1
                scene_counts["covered"] += 1
            if int(row["gmt_choice_id"]) != target:
                totals["hard"] += 1
                scene_counts["hard"] += 1
                if covered:
                    totals["hard_recoverable"] += 1
                    scene_counts["hard_recoverable"] += 1
            if not covered:
                totals["candidate_miss"] += 1
                scene_counts["candidate_miss"] += 1
        denominator = scene_counts["correct_existing"]
        per_scene[scene] = {
            **{key: int(value) for key, value in scene_counts.items()},
            "recall": (scene_counts["covered"] / denominator if denominator else None),
        }
    denominator = totals["correct_existing"]
    hard = totals["hard"]
    return {
        "K": int(k),
        "correct_existing": int(denominator),
        "covered": int(totals["covered"]),
        "candidate_miss": int(totals["candidate_miss"]),
        "recall": (totals["covered"] / denominator if denominator else None),
        "hard_decisions": int(hard),
        "hard_recoverable": int(totals["hard_recoverable"]),
        "hard_recoverable_fraction": (
            totals["hard_recoverable"] / hard if hard else None
        ),
        "per_scene": per_scene,
    }


def render(manifest: dict, results: list[dict], chosen: int | None) -> str:
    lines = [
        "# Candidate coverage audit",
        "",
        "This report uses only the frozen `JEV-train` scenes and labels made from",
        "train annotations.  Candidate ranking is released GMT raw `traj_score`,",
        "with Global-ID as a deterministic tie-breaker.  The full active set is",
        "recorded online; Top-K is applied only for this coverage audit.",
        "",
        f"* source split: `{manifest['source_split']}`",
        f"* scenes: {manifest['scene_count']} total in frozen manifest; {len(manifest['train_scenes'])} JEV-train",
        f"* target rows: `target_type=EXISTING` with an IoU>=0.5 train match",
        "",
        "| K | correct existing | covered | candidate miss | recall | hard decisions | hard recoverable |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in results:
        def fmt(value):
            return "n/a" if value is None else f"{value:.4f}" if isinstance(value, float) else str(value)
        lines.append(
            f"| {item['K']} | {item['correct_existing']} | {item['covered']} | "
            f"{item['candidate_miss']} | {fmt(item['recall'])} | {item['hard_decisions']} | "
            f"{item['hard_recoverable']} |"
        )
    lines += [
        "",
        f"**Frozen K decision:** `{chosen if chosen is not None else 'STOP (<95% at K=16)'}`.",
        "",
        "A correct existing target is covered when its active Global ID is in the",
        "released-score Top-K set.  NEW labels are excluded from this recall",
        "denominator.  A hard decision is a released GMT choice different from",
        "the supervision target; `hard recoverable` additionally requires the",
        "target to be in Top-K.",
        "",
        "## Per-scene coverage",
        "",
    ]
    for item in results:
        lines.append(f"### K={item['K']}")
        lines.append("")
        lines.append("| scene | correct existing | covered | miss | recall | hard | hard recoverable |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for scene, values in item["per_scene"].items():
            lines.append(
                f"| {scene} | {values.get('correct_existing', 0)} | {values.get('covered', 0)} | "
                f"{values.get('candidate_miss', 0)} | {fmt(values.get('recall'))} | "
                f"{values.get('hard', 0)} | {values.get('hard_recoverable', 0)} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    split = json.loads(args.split.read_text())
    rows_by_scene = load_rows(args.input, split)
    results = [summarize(rows_by_scene, k) for k in K_VALUES]
    chosen = next((item["K"] for item in results if item["recall"] is not None and item["recall"] >= 0.95), None)
    payload = {
        "format": "gmt-candidate-coverage-v1",
        "source_split": str(args.split.resolve()),
        "scene_count": int(split["scene_count"]),
        "train_scenes": sorted(rows_by_scene),
        "results": results,
        "chosen_K": chosen,
        "stop_reason": "K=16 recall < 0.95" if chosen is None else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    args.output.with_name("CANDIDATE_COVERAGE.md").write_text(render(payload, results, chosen))
    print(json.dumps({"chosen_K": chosen, "results": results}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
