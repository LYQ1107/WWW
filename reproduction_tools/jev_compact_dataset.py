"""Memory-mapped representation of one shared JEV counterfactual dataset."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np

from gtr.modeling.jev_decision import ACTION_NAMES, QUESTION_NAMES, action_index, question_index
from jev_dataset_contract import validate_record


COMPACT_SCHEMA_VERSION = 1
OUTCOME_FIELDS = (
    "utility",
    "future_correct_identity_duration",
    "future_identity_switches",
    "future_fragmentation",
    "future_fragments",
    "future_collisions",
    "memory_contamination",
    "contamination_duration",
    "recovery_latency",
    "false_reactivation",
    "new_id_fragmentation",
    "window_idf1",
    "window_assa_proxy",
    "future_events",
    "informative",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _iter_records(paths: Sequence[Path]):
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                try:
                    validate_record(record, allow_future_gt=True)
                except Exception as exc:  # noqa: BLE001 - add source context
                    raise ValueError(f"invalid record at {path}:{line_number}: {exc}") from exc
                yield path, line_number, record


def _checkpoint_digest(record: Mapping[str, Any]) -> str:
    value = str(record["gmt_checkpoint_sha256"])
    return value[7:] if value.startswith("sha256:") else value


def build_compact_dataset(inputs: Sequence[Path], output: Path) -> Mapping[str, Any]:
    if not inputs:
        raise ValueError("at least one JSONL shard is required")
    if output.exists():
        raise RuntimeError(f"refusing to overwrite compact dataset: {output}")
    inputs = [path.resolve() for path in inputs]
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)

    records = 0
    state_dim = None
    sequences = set()
    checkpoints = set()
    horizons = set()
    for _path, _line, record in _iter_records(inputs):
        if int(record["horizon"]) != 8:
            raise ValueError("compact three-way dataset accepts H=8 records only")
        current_dim = len(record["state"]["feature_vector"])
        if state_dim is None:
            state_dim = current_dim
        if current_dim != state_dim:
            raise ValueError("feature dimension mismatch")
        records += 1
        sequences.add(str(record["sequence"]))
        checkpoints.add(_checkpoint_digest(record))
        horizons.add(int(record["horizon"]))
    if records == 0 or state_dim is None:
        raise ValueError("empty counterfactual dataset")
    if len(checkpoints) != 1:
        raise ValueError(f"checkpoint provenance mismatch: {sorted(checkpoints)}")
    if checkpoints != {"cd72823824d16c86ed27c2dfc8323de610aa9f6c0c0b29249aa3de609deabce8"}:
        raise ValueError("compact dataset is not bound to canonical model_20000")

    output.mkdir(parents=True, exist_ok=False)
    shape = (records,)
    arrays = {
        "features": np.lib.format.open_memmap(output / "features.npy", mode="w+", dtype=np.float32, shape=(records, state_dim)),
        "questions": np.lib.format.open_memmap(output / "questions.npy", mode="w+", dtype=np.int8, shape=shape),
        "legal_actions": np.lib.format.open_memmap(output / "legal_actions.npy", mode="w+", dtype=np.int8, shape=(records, len(ACTION_NAMES))),
        "target_probs": np.lib.format.open_memmap(output / "target_probs.npy", mode="w+", dtype=np.float32, shape=(records, len(ACTION_NAMES))),
        "best_mask": np.lib.format.open_memmap(output / "best_mask.npy", mode="w+", dtype=np.bool_, shape=(records, len(ACTION_NAMES))),
        "sample_weight": np.lib.format.open_memmap(output / "sample_weight.npy", mode="w+", dtype=np.float32, shape=shape),
        "sequence_ids": np.lib.format.open_memmap(output / "sequence_ids.npy", mode="w+", dtype=np.int32, shape=shape),
        "video_ids": np.lib.format.open_memmap(output / "video_ids.npy", mode="w+", dtype=np.int32, shape=shape),
        "frames": np.lib.format.open_memmap(output / "frames.npy", mode="w+", dtype=np.int32, shape=shape),
        "views": np.lib.format.open_memmap(output / "views.npy", mode="w+", dtype=np.int16, shape=shape),
        "event_orders": np.lib.format.open_memmap(output / "event_orders.npy", mode="w+", dtype=np.int32, shape=shape),
        "outcomes": np.lib.format.open_memmap(output / "outcomes.npy", mode="w+", dtype=np.float32, shape=(records, len(ACTION_NAMES), len(OUTCOME_FIELDS))),
    }
    arrays["legal_actions"][:] = -1
    arrays["target_probs"][:] = 0.0
    arrays["best_mask"][:] = False
    arrays["outcomes"][:] = np.nan
    sequence_names = sorted(sequences)
    sequence_to_id = {name: index for index, name in enumerate(sequence_names)}
    index = 0
    for _path, _line, record in _iter_records(inputs):
        state = record["state"]
        context = state["online_context"]
        legal = list(record["legal_actions"])
        legal_ids = [action_index(value) for value in legal]
        arrays["features"][index] = np.asarray(state["feature_vector"], dtype=np.float32)
        arrays["questions"][index] = question_index(record["question_type"])
        arrays["legal_actions"][index, : len(legal_ids)] = np.asarray(legal_ids, dtype=np.int8)
        arrays["target_probs"][index, : len(legal_ids)] = np.asarray(record["target_probs"], dtype=np.float32)
        for action in record["best_actions"]:
            arrays["best_mask"][index, action_index(action)] = True
        arrays["sample_weight"][index] = float(record.get("sample_weight", 1.0))
        arrays["sequence_ids"][index] = sequence_to_id[str(record["sequence"])]
        arrays["video_ids"][index] = int(context["video_id"])
        arrays["frames"][index] = int(context["frame"])
        arrays["views"][index] = int(context["view"])
        arrays["event_orders"][index] = int(context["event_order"])
        for action_name, outcome in record["action_outcomes"].items():
            action_id = action_index(action_name)
            for field_index, field in enumerate(OUTCOME_FIELDS):
                value = outcome.get(field)
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    arrays["outcomes"][index, action_id, field_index] = float(value)
        index += 1
    for array in arrays.values():
        array.flush()

    array_hashes = {name: sha256(output / f"{name}.npy") for name in arrays}
    manifest = {
        "status": "PASS",
        "schema_version": COMPACT_SCHEMA_VERSION,
        "format": "numpy_npy_memory_mapped",
        "records": records,
        "state_dim": state_dim,
        "horizon": 8,
        "sequence_names": sequence_names,
        "sequence_count": len(sequence_names),
        "question_names": list(QUESTION_NAMES),
        "action_names": list(ACTION_NAMES),
        "outcome_fields": list(OUTCOME_FIELDS),
        "gmt_checkpoint_sha256": "sha256:" + next(iter(checkpoints)),
        "input_files": [str(path) for path in inputs],
        "input_sha256": {str(path): sha256(path) for path in inputs},
        "array_sha256": array_hashes,
        "official_test_read": False,
        "shared_dataset_contract": "same features/questions/legal-actions/targets/sample-weights for all methods",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


class CompactJEVData:
    """Read-only memory-mapped arrays shared by the three training workers."""

    ARRAY_NAMES = (
        "features",
        "questions",
        "legal_actions",
        "target_probs",
        "best_mask",
        "sample_weight",
        "sequence_ids",
        "video_ids",
        "frames",
        "views",
        "event_orders",
        "outcomes",
    )

    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("status") != "PASS" or self.manifest.get("horizon") != 8:
            raise ValueError("invalid compact JEV manifest")
        for name in self.ARRAY_NAMES:
            setattr(self, name, np.load(self.root / f"{name}.npy", mmap_mode="r"))
        if len(self.features) != int(self.manifest["records"]):
            raise ValueError("compact array length mismatch")
        self.sequence_to_id = {name: index for index, name in enumerate(self.manifest["sequence_names"])}

    def indices_for_sequences(self, names: Iterable[str]) -> np.ndarray:
        ids = np.asarray([self.sequence_to_id[str(name)] for name in names], dtype=np.int32)
        return np.flatnonzero(np.isin(self.sequence_ids, ids))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_compact_dataset(args.input, args.output)
    print(json.dumps({"status": report["status"], "output": str(args.output.resolve()), "records": report["records"]}, indent=2))


if __name__ == "__main__":
    main()
