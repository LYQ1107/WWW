"""Match policy baseline capacity and measure controller overhead.

The script searches hidden widths for reviewer baselines so trainable
parameter differences are explicit rather than hidden in a model name.  FLOP
counts are conservative linear-layer MAC estimates and latency is measured on
the requested device with the same typed legal-action interface used by the
trainer.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Iterable, List

import torch
from torch import nn

from gtr.modeling.jev_decision import legal_actions_for
from train_jev import MODEL_NAMES, choose_model


DEFAULT_CANDIDATES = (
    "question_conditioned_mlp",
    "shared_heads",
    "independent_mlp",
    "nonlinear_state_threshold",
    "question_threshold",
    "question_conditioned_fixed_head",
    "action_conditioned_no_question",
)


def trainable_params(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def estimated_macs(model: nn.Module, state_dim: int, device: torch.device) -> int:
    total = 0

    def hook(module: nn.Module, inputs, output):
        nonlocal total
        if isinstance(module, nn.Linear):
            total += int(output.numel()) * int(module.in_features)

    hooks = [module.register_forward_hook(hook) for module in model.modules() if isinstance(module, nn.Linear)]
    features = torch.zeros(1, state_dim, device=device)
    questions = ["MATCH_DECISION"]
    legal = [legal_actions_for("MATCH_DECISION")]
    with torch.no_grad():
        model.eval()(features, questions, legal)
    for handle in hooks:
        handle.remove()
    return total


def benchmark(model: nn.Module, state_dim: int, device: torch.device, repeats: int = 30) -> Dict[str, float]:
    model = model.to(device).eval()
    features = torch.zeros(1, state_dim, device=device)
    questions = ["MATCH_DECISION"]
    legal = [legal_actions_for("MATCH_DECISION")]
    with torch.no_grad():
        for _ in range(5):
            model(features, questions, legal)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        for _ in range(repeats):
            model(features, questions, legal)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
    return {
        "batch1_latency_ms": 1000.0 * elapsed / repeats,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else None,
    }


def instantiate(name: str, state_dim: int, hidden_dim: int, device: torch.device) -> nn.Module:
    return choose_model(name, state_dim, hidden_dim).to(device).eval()


def match(
    state_dim: int,
    target_model: str,
    target_hidden: int,
    candidates: Iterable[str],
    hidden_widths: Iterable[int],
    device: torch.device,
) -> Dict[str, object]:
    target = instantiate(target_model, state_dim, target_hidden, device)
    target_count = trainable_params(target)
    target_measure = benchmark(target, state_dim, device)
    target_entry = {
        "model": target_model,
        "hidden_dim": target_hidden,
        "trainable_params": target_count,
        "relative_param_difference": 0.0,
        "estimated_macs": estimated_macs(target, state_dim, device),
        **target_measure,
    }
    entries: List[Dict[str, object]] = [target_entry]
    for name in candidates:
        if name not in MODEL_NAMES:
            raise ValueError(f"unknown candidate model: {name}")
        best = None
        for width in hidden_widths:
            model = instantiate(name, state_dim, int(width), device)
            count = trainable_params(model)
            relative = abs(count - target_count) / max(1, target_count)
            choice = (relative, abs(count - target_count), int(width), model)
            if best is None or choice[:3] < best[:3]:
                best = choice
        assert best is not None
        relative, _, width, model = best
        measure = benchmark(model, state_dim, device)
        entries.append(
            {
                "model": name,
                "hidden_dim": width,
                "trainable_params": trainable_params(model),
                "relative_param_difference": relative,
                "estimated_macs": estimated_macs(model, state_dim, device),
                **measure,
            }
        )
    return {
        "status": "PASS",
        "state_dim": state_dim,
        "target": target_entry,
        "device": str(device),
        "candidate_search_widths": list(hidden_widths),
        "models": entries,
        "parameter_match_rule": "minimize absolute trainable-parameter difference; target <2% is a gate",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-dim", type=int, default=64)
    parser.add_argument("--target-model", choices=sorted(MODEL_NAMES), default="jev")
    parser.add_argument("--target-hidden", type=int, default=128)
    parser.add_argument("--candidates", nargs="+", default=list(DEFAULT_CANDIDATES))
    parser.add_argument(
        "--hidden-widths",
        nargs="+",
        type=int,
        default=[
            16,
            24,
            32,
            48,
            64,
            80,
            96,
            112,
            128,
            132,
            136,
            137,
            138,
            139,
            140,
            141,
            142,
            144,
            148,
            152,
            160,
            192,
            224,
            256,
        ],
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=Path("results/capacity_match.json"))
    args = parser.parse_args()
    if args.state_dim < 4 or args.target_hidden < 1:
        raise ValueError("state-dim and target-hidden must be positive")
    device = torch.device(args.device)
    report = match(
        args.state_dim,
        args.target_model,
        args.target_hidden,
        args.candidates,
        args.hidden_widths,
        device,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
