"""Object-level feature clip dataset.

Images and dense Ω tokens stay outside Git.  Training consumes sharded NPZ
files produced by the audited feature extractor and refuses GT position fields
in the input cache.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class WorldQueryClipDataset(Dataset):
    def __init__(self, paths: list[str | Path], max_observations: int | None = None):
        self.rows = []
        for path in paths:
            p = Path(path)
            with np.load(p, allow_pickle=False) as data:
                forbidden = {"gt_xyz", "ground_truth_xyz", "oracle_xyz"} & set(data.files)
                if forbidden:
                    raise ValueError(f"GT geometry is forbidden in training cache: {sorted(forbidden)}")
                required = {"observations", "geometry", "assignment", "birth", "visibility"}
                missing = required - set(data.files)
                if missing:
                    raise ValueError(f"cache {p} missing {sorted(missing)}")
                n = int(data["observations"].shape[0])
                for i in range(n):
                    row = {k: data[k][i] for k in required}
                    if "observation_valid" in data.files:
                        row["observation_valid"] = data["observation_valid"][i]
                    if "camera_ids" in data.files:
                        row["camera_ids"] = data["camera_ids"][i]
                    self.rows.append(row)
        self.max_observations = max_observations

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        obs = np.asarray(row["observations"], dtype=np.float32)
        geo = np.asarray(row["geometry"], dtype=np.float32)
        if self.max_observations is not None:
            n = int(self.max_observations)
            obs = obs[:n]; geo = geo[:n]
        output = {
            "observations": torch.from_numpy(obs), "geometry": torch.from_numpy(geo),
            "assignment": torch.from_numpy(np.asarray(row["assignment"], dtype=np.int64)),
            "birth": torch.from_numpy(np.asarray(row["birth"], dtype=np.float32)),
            "visibility": torch.from_numpy(np.asarray(row["visibility"], dtype=np.float32)),
        }
        if "observation_valid" in row:
            output["observation_valid"] = torch.from_numpy(
                np.asarray(row["observation_valid"], dtype=np.bool_)
            )
        if "camera_ids" in row:
            output["camera_ids"] = torch.from_numpy(np.asarray(row["camera_ids"], dtype=np.int64))
        return output
