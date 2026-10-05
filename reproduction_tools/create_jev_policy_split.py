"""Create a sequence-disjoint train/validation manifest for JEV search.

The command accepts only the policy-search pool.  Official-test records are
blocked entirely until the canonical final-selection lock exists; this keeps
the train/validation manifest auditable and fail-closed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
from typing import Iterable, List, Mapping, Sequence, Tuple

from jev_dataset_contract import validate_record


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def read_sequences(paths: Sequence[Path]) -> Tuple[List[str], int]:
    sequences = []
    records = 0
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    validate_record(record, allow_future_gt=True)
                    sequence = str(record["sequence"])
                except Exception as exc:
                    raise ValueError(f"invalid record in {path}:{line_number}: {exc}") from exc
                sequences.append(sequence)
                records += 1
    if not sequences:
        raise ValueError("policy-search pool is empty")
    return sequences, records


def split_policy_sequences(
    sequences: Iterable[str], *, seed: int, val_fraction: float
) -> Mapping[str, Tuple[str, ...]]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between zero and one")
    unique = sorted(set(str(value) for value in sequences))
    if len(unique) < 2:
        raise ValueError("at least two complete sequences are required for train/val")
    shuffled = list(unique)
    random.Random(int(seed)).shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * float(val_fraction))))
    val_count = min(val_count, len(shuffled) - 1)
    return {
        "train": tuple(sorted(shuffled[:-val_count])),
        "val": tuple(sorted(shuffled[-val_count:])),
    }


def canonical_sequence_hash(train: Sequence[str], val: Sequence[str]) -> str:
    payload = json.dumps(
        {"train": sorted(set(train)), "val": sorted(set(val))},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def build_manifest(
    inputs: Sequence[Path],
    *,
    seed: int,
    val_fraction: float,
    official_test: Sequence[Path] = (),
) -> Mapping[str, object]:
    if official_test:
        raise ValueError(
            "official TEST data is blocked before FINAL_SELECTION_LOCK; "
            "create the policy split from TRAIN counterfactual records only"
        )
    sequences, record_count = read_sequences(inputs)
    split = split_policy_sequences(sequences, seed=seed, val_fraction=val_fraction)
    return {
        "schema_version": 1,
        "purpose": "jev_policy_architecture_search",
        "seed": int(seed),
        "source_files": [str(path) for path in inputs],
        "source_sha256": {str(path): sha256_file(path) for path in inputs},
        "source_record_count": int(record_count),
        "source_sequence_count": len(set(sequences)),
        "train_sequences": list(split["train"]),
        "val_sequences": list(split["val"]),
        "train_record_count": sum(sequences.count(name) for name in split["train"]),
        "val_record_count": sum(sequences.count(name) for name in split["val"]),
        "official_test_used_for_search": False,
        "official_test_files": [],
        "official_test_sha256": {},
        "official_test_record_count_audited": 0,
        "official_test_sequence_count_audited": 0,
        "official_test_access": "BLOCKED_BEFORE_FINAL_SELECTION_LOCK",
        "sequence_hash": canonical_sequence_hash(split["train"], split["val"]),
        "rules": {
            "split_unit": "complete_sequence",
            "official_test_excluded_from_architecture_search": True,
            "no_frame_level_random_split": True,
            "selection_must_use_policy_val_only": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--official-test", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path, default=Path("manifests/jev_policy_split.json"))
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    args = parser.parse_args()
    manifest = build_manifest(
        args.input,
        seed=args.seed,
        val_fraction=args.val_fraction,
        official_test=args.official_test,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(args.output), "sequence_hash": manifest["sequence_hash"]}, indent=2))


if __name__ == "__main__":
    main()
