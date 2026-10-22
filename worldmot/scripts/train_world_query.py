#!/usr/bin/env python3
"""Finite PWQT training entry point; no random checkpoint is valid for MOT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from worldmot.models.persistent_world_queries import PersistentWorldQueryTransformer
from worldmot.training.clip_dataset import WorldQueryClipDataset
from worldmot.training.train_utils import save_checkpoint, seed_everything
from worldmot.training.world_query_criterion import WorldQueryCriterion


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--cache", nargs="+", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--seed", type=int, default=20261022)
    p.add_argument("--device", default="cpu")
    args = p.parse_args(argv)
    seed_everything(args.seed)
    ds = WorldQueryClipDataset(args.cache)
    if not len(ds):
        raise RuntimeError("empty feature cache")
    sample = ds[0]
    model = PersistentWorldQueryTransformer(observation_dim=int(sample["observations"].shape[-1]), cameras=int(sample["visibility"].shape[-1])).to(args.device)
    criterion = WorldQueryCriterion()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=lambda rows: {
        k: torch.stack([r[k] for r in rows]) for k in rows[0]
    })
    history = []
    model.train()
    for epoch in range(args.epochs):
        for row in loader:
            observation_mask = row.get("observation_valid")
            if observation_mask is not None:
                observation_mask = observation_mask.to(args.device)
            camera_ids = row.get("camera_ids")
            if camera_ids is not None:
                camera_ids = camera_ids.to(args.device)
            out = model(row["observations"].to(args.device), row["geometry"].to(args.device),
                        observation_mask=observation_mask, camera_ids=camera_ids)
            targets = {k: v.to(args.device) for k, v in row.items()
                       if k in {"assignment", "birth", "visibility", "observation_valid"}}
            losses = criterion(out, targets)
            opt.zero_grad(set_to_none=True); losses["total"].backward(); opt.step()
            history.append({k: float(v.detach().cpu()) for k, v in losses.items()})
    manifest = {"status": "TRAINED_FINITE_CACHE", "cache": [str(x) for x in args.cache],
                "epochs": args.epochs, "seed": args.seed, "dataset_rows": len(ds),
                "observation_dim": int(sample["observations"].shape[-1]), "history": history,
                "backbone": "frozen_external_rtmdet_omega_features"}
    save_checkpoint(args.output, model, opt, args.epochs, manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
