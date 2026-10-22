#!/usr/bin/env python3
"""Benchmark PWQT decoder latency; labels output as query-only."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from worldmot.evaluation.query_runtime_benchmark import benchmark_query_decoder
from worldmot.models.persistent_world_queries import PersistentWorldQueryTransformer


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--observations", type=int, default=4)
    p.add_argument("--output", required=True)
    args = p.parse_args(argv)
    model = PersistentWorldQueryTransformer()
    row = benchmark_query_decoder(model, args.steps, args.observations, args.device)
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        writer.writeheader(); writer.writerow(row)
    print(row)


if __name__ == "__main__":
    main()

