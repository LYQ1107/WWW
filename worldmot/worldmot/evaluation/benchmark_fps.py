"""Stage timing and resource accounting for real continuous runs."""

from __future__ import annotations

from contextlib import contextmanager
import os
import resource
import time


class RuntimeTimer:
    def __init__(self):
        self.stages: dict[str, list[float]] = {}
        self.last_stages: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str):
        t = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - t
            self.stages.setdefault(name, []).append(elapsed)
            self.last_stages[name] = float(elapsed)

    def add(self, name: str, seconds: float) -> None:
        value = float(seconds)
        self.stages.setdefault(name, []).append(value)
        self.last_stages[name] = value

    def summary(self) -> dict:
        out = {}
        total = []
        for name, values in self.stages.items():
            vals = sorted(float(v) for v in values)
            if not vals:
                continue
            def q(p):
                idx = min(len(vals) - 1, max(0, int(round((len(vals) - 1) * p))))
                return vals[idx]
            out[name] = {"count": len(vals), "mean_s": sum(vals) / len(vals), "p50_s": q(0.50), "p95_s": q(0.95), "total_s": sum(vals)}
            if name == "total": total.extend(vals)
        if total:
            out["pipeline_fps"] = len(total) / max(sum(total), 1e-9)
        out["host_max_rss_mb"] = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0)
        return out


def gpu_memory_summary(device: str = "cuda:0") -> dict:
    try:
        import torch
        if not torch.cuda.is_available():
            return {"available": False}
        idx = torch.device(device).index or 0
        return {"available": True, "device": device,
                "allocated_mb": float(torch.cuda.memory_allocated(idx) / 2**20),
                "reserved_mb": float(torch.cuda.memory_reserved(idx) / 2**20),
                "peak_allocated_mb": float(torch.cuda.max_memory_allocated(idx) / 2**20),
                "peak_reserved_mb": float(torch.cuda.max_memory_reserved(idx) / 2**20)}
    except Exception as exc:
        return {"available": False, "error": str(exc)}
