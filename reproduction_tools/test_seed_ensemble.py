"""CPU invariant test for the no-best-seed controller aggregation."""

from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile

import torch

from gtr.modeling.jev_baselines import GlobalLearnedThreshold
from gtr.modeling.jev_runtime import build_controller_from_checkpoint


def digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="jev_seed_ensemble_") as raw:
        root = Path(raw)
        members = []
        for seed in (20261003, 20261004, 20261005):
            model = GlobalLearnedThreshold()
            with torch.no_grad():
                model.threshold.add_(float(seed - 20261004) * 0.1)
            path = root / f"seed_{seed}.pth"
            torch.save(
                {
                    "model_name": "global_threshold",
                    "state_dim": 64,
                    "hidden_dim": 64,
                    "model": model.state_dict(),
                },
                path,
            )
            members.append({"seed": seed, "path": str(path), "sha256": digest(path)})
        ensemble_path = root / "ensemble.pth"
        torch.save(
            {
                "checkpoint_type": "seed_ensemble",
                "model_name": "global_threshold",
                "seed_set": [item["seed"] for item in members],
                "aggregation": "mean_probability",
                "members": members,
            },
            ensemble_path,
        )
        controller = build_controller_from_checkpoint(ensemble_path, device="cpu")
        output = controller(
            torch.zeros(1, 64),
            ["MATCH_DECISION"],
            [["ACCEPT_CURRENT", "START_NEW"]],
        )
        assert output["probs"].shape == (1, 2)
        assert torch.isfinite(output["probs"]).all()
        assert torch.allclose(output["probs"].sum(dim=1), torch.ones(1))
    print("Seed ensemble invariants: PASS")


if __name__ == "__main__":
    main()
