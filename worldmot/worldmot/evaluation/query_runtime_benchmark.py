"""Timing helpers for the light PWQT decoder (never full-system FPS)."""

from __future__ import annotations

import time

import numpy as np
import torch


def percentile(values: list[float], p: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), p)) if values else float("nan")


@torch.inference_mode()
def benchmark_query_decoder(model, steps: int = 20, observations: int = 4,
                            device: str = "cpu") -> dict:
    model = model.to(device).eval()
    state = model.init_state(1, device=device)
    x = torch.randn(1, observations, model.observation_dim, device=device)
    g = torch.randn(1, observations, model.geometry_dim, device=device)
    mask = torch.ones(1, observations, dtype=torch.bool, device=device)
    for _ in range(3):
        out = model(x, g, state, mask)
        state = out["state"].detach()
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
    times = []
    for _ in range(int(steps)):
        start = time.perf_counter()
        out = model(x, g, state, mask)
        state = out["state"].detach()
        if str(device).startswith("cuda"):
            torch.cuda.synchronize(device)
        times.append(time.perf_counter() - start)
    p50, p95 = percentile(times, 50), percentile(times, 95)
    return {"scope": "query_decoder_only", "device": str(device), "steps": int(steps),
            "observations": int(observations), "p50_s": p50, "p95_s": p95,
            "fps_pair_p50": 1.0 / p50 if p50 > 0 else 0.0,
            "fps_pair_p95_inverse": 1.0 / p95 if p95 > 0 else 0.0,
            "end_to_end": False, "status": "DIAGNOSTIC_ONLY"}

