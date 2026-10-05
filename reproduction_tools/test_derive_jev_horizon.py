"""CPU test for deriving horizons without rerunning a rollout."""

import json
from pathlib import Path
import tempfile

from derive_jev_horizon import derive
from jev_dataset_tools import make_record


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "h32.jsonl"
        outcomes = {
            "ACCEPT_CURRENT": {"utility": 1.0, "sample_weight": 1.0},
            "REASSOCIATE": {"utility": 0.5, "sample_weight": 1.0},
            "START_NEW": {"utility": 0.0, "sample_weight": 1.0},
        }
        record = make_record(
            dataset="fixture",
            sequence="s1",
            frame=0,
            view=0,
            question_type="MATCH_DECISION",
            state={"feature_vector": [0.1]},
            legal_actions=list(outcomes),
            action_outcomes=outcomes,
            gmt_checkpoint_sha256="sha256:checkpoint",
            horizon=32,
        )
        record["horizon_outcomes"] = {
            "1": outcomes,
            "8": outcomes,
            "16": outcomes,
            "32": outcomes,
        }
        source.write_text(json.dumps(record) + "\n", encoding="utf-8")
        source.with_suffix(source.suffix + ".manifest.json").write_text(
            json.dumps(
                {
                    "status": "PASS",
                    "horizon": 32,
                    "derived_horizons": [1, 8, 16, 32],
                    "gmt_checkpoint_sha256": "sha256:checkpoint",
                }
            ),
            encoding="utf-8",
        )
        output = root / "h1.jsonl"
        manifest = derive(source, output, 1)
        assert manifest["status"] == "PASS"
        derived = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
        assert derived["horizon"] == 1
        assert derived["action_outcomes"] == outcomes
    print("Horizon derivation invariants: PASS")


if __name__ == "__main__":
    main()
