#!/usr/bin/env python3
"""Train/evaluate cached B1--B4 decision baselines on train/cal/dev."""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from gtr.jev.decision_head import IndependentLinear, IndependentMLP, SetChoiceHead


SEEDS = (0, 1, 2)
MODEL_NAMES = ("B1_LINEAR", "B2_MLP", "B3_SET_CHOICE", "B4_SET_CHOICE_VERIFY")


class ChoiceVerify(nn.Module):
    def __init__(self, query_dim: int, option_dim: int) -> None:
        super().__init__()
        self.choice = SetChoiceHead(query_dim, option_dim)
        self.verify = nn.Sequential(
            nn.Linear(query_dim + option_dim + 3, 64), nn.GELU(), nn.Linear(64, 1)
        )

    def forward(self, query, options, mask):
        logits = self.choice(query, options, mask)
        probs = torch.softmax(logits, dim=-1)
        selected = probs.argmax(-1)
        selected_option = options[torch.arange(len(options), device=options.device), selected]
        top = probs.topk(min(2, probs.shape[1]), dim=-1).values
        entropy = -(probs.masked_fill(~mask, 0.0) * torch.log(probs.clamp_min(1e-12))).sum(-1)
        margin = top[:, 0] - (top[:, 1] if top.shape[1] > 1 else top[:, 0])
        verify_features = torch.cat([query, selected_option, probs.max(-1).values[:, None], entropy[:, None], margin[:, None]], dim=-1)
        return logits, self.verify(verify_features).squeeze(-1)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_cache(path: Path, device: torch.device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload


def fit_standardization(train: dict):
    q = train["query"]
    o = train["option"]
    om = train["option_mask"]
    q_mean = q.mean(0)
    q_std = q.std(0, unbiased=False).clamp_min(1e-6)
    valid_options = o[om]
    o_mean = valid_options.mean(0)
    o_std = valid_options.std(0, unbiased=False).clamp_min(1e-6)
    return {"q_mean": q_mean, "q_std": q_std, "o_mean": o_mean, "o_std": o_std}


def standardize(cache: dict, stats: dict, device: torch.device):
    q = (cache["query"] - stats["q_mean"]) / stats["q_std"]
    o = (cache["option"] - stats["o_mean"]) / stats["o_std"]
    extra = (cache["extra_option"] - stats["o_mean"]) / stats["o_std"]
    return {
        "query": q.to(device), "option": o.to(device), "option_mask": cache["option_mask"].to(device),
        "extra_option": extra.to(device), "extra_mask": cache["extra_mask"].to(device),
        "target": cache["target"].to(device), "baseline": cache["baseline"].to(device),
        "supervised": cache["supervised"].to(device), "hard": cache["hard_recoverable"].to(device),
        "metadata": cache["metadata"],
    }


def valid_mask(data: dict, subset: torch.Tensor | None = None):
    mask = data["supervised"] & (data["target"] >= 0)
    if subset is not None:
        mask = mask & subset
    return mask


def choice_loss(logits, target, mask):
    valid = target >= 0
    if not bool(valid.any()):
        return logits.sum() * 0.0
    l = logits[valid]
    t = target[valid]
    m = mask[valid]
    logp = torch.log_softmax(l, dim=-1)
    ce = -logp[torch.arange(len(t), device=l.device), t].mean()
    probs = logp.exp()
    onehot = torch.zeros_like(probs).scatter_(1, t[:, None], 1.0)
    brier = ((probs - onehot) ** 2).masked_fill(~m, 0.0).sum(-1).mean()
    return ce + 0.1 * brier


def make_model(name: str, qdim: int, odim: int):
    if name == "B1_LINEAR":
        return IndependentLinear(qdim + odim)
    if name == "B2_MLP":
        return IndependentMLP(qdim + odim)
    if name == "B3_SET_CHOICE":
        return SetChoiceHead(qdim, odim)
    if name == "B4_SET_CHOICE_VERIFY":
        return ChoiceVerify(qdim, odim)
    raise ValueError(name)


def forward(model, name: str, data: dict):
    if name in ("B1_LINEAR", "B2_MLP"):
        return model(data["query"], data["option"], data["option_mask"]), None
    if name == "B3_SET_CHOICE":
        return model(data["query"], data["option"], data["option_mask"]), None
    return model(data["query"], data["option"], data["option_mask"])


def metric_bundle(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    row_filter: torch.Tensor | None = None,
):
    valid = target >= 0
    if row_filter is not None:
        valid = valid & row_filter
    if not bool(valid.any()):
        return {"n": 0, "accuracy": None, "nll": None, "brier": None, "ece": None}
    logits = logits[valid].float()
    target = target[valid]
    mask = mask[valid]
    probs = torch.softmax(logits, dim=-1)
    pred = probs.argmax(-1)
    confidence = probs.max(-1).values
    correct = pred == target
    nll = -torch.log(probs[torch.arange(len(target), device=target.device), target].clamp_min(1e-12)).mean()
    onehot = torch.zeros_like(probs).scatter_(1, target[:, None], 1.0)
    brier = ((probs - onehot) ** 2).masked_fill(~mask, 0.0).sum(-1).mean()
    ece = torch.zeros((), device=logits.device)
    for lo in torch.linspace(0, 1, 11, device=logits.device)[:-1]:
        hi = lo + 0.1
        selected = (confidence >= lo) & ((confidence < hi) if hi < 1 else (confidence <= hi))
        if bool(selected.any()):
            ece = ece + selected.float().mean() * (confidence[selected].mean() - correct[selected].float().mean()).abs()
    return {"n": int(len(target)), "accuracy": float(correct.float().mean()), "nll": float(nll), "brier": float(brier), "ece": float(ece)}


def baseline_metric(data: dict):
    valid = valid_mask(data)
    target = data["target"][valid]
    baseline = data["baseline"][valid]
    return {"n": int(len(target)), "accuracy": float((target == baseline).float().mean()) if len(target) else None}


def fit_temperature(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    valid = target >= 0
    logits, target, mask = logits[valid].float(), target[valid], mask[valid]
    log_temp = torch.zeros((), device=logits.device, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temp], lr=0.1, max_iter=50, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = -torch.log_softmax(logits / log_temp.exp(), -1)[torch.arange(len(target), device=target.device), target].mean()
        loss.backward()
        return loss
    optimizer.step(closure)
    return float(log_temp.exp().detach().cpu())


def uncertainty_metrics(verify_logits: torch.Tensor, choice_logits: torch.Tensor, target: torch.Tensor):
    valid = target >= 0
    if not bool(valid.any()):
        return {"n": 0, "auroc": None, "auprc": None, "brier": None, "ece": None, "risk_coverage": {}}
    probs = torch.sigmoid(verify_logits[valid]).detach().cpu().numpy()
    choice = torch.softmax(choice_logits[valid], -1)
    pred = choice.argmax(-1).cpu()
    truth = (pred == target[valid].cpu()).numpy().astype(np.int64)
    brier = float(np.mean((probs - truth) ** 2))
    positives = int(truth.sum())
    negatives = int(len(truth) - positives)
    if positives == 0 or negatives == 0:
        auroc = auprc = None
    else:
        order = np.argsort(-probs, kind="mergesort")
        sorted_truth = truth[order]
        cumulative_pos = np.cumsum(sorted_truth)
        rank_sum = float(np.where(sorted_truth > 0)[0].sum() + positives)
        auroc = (rank_sum - positives * (positives - 1) / 2.0) / (positives * negatives)
        precision = cumulative_pos / np.arange(1, len(truth) + 1)
        auprc = float((precision * sorted_truth).sum() / positives)
    ece = 0.0
    for lo in np.linspace(0, 1, 10, endpoint=False):
        hi = lo + 0.1
        selected = (probs >= lo) & ((probs < hi) if hi < 1 else (probs <= hi))
        if selected.any():
            ece += float(selected.mean() * abs(probs[selected].mean() - truth[selected].mean()))
    risk = {}
    for keep in (1.0, 0.9, 0.8, 0.7, 0.5):
        n = max(1, int(len(probs) * keep))
        order = np.argsort(-probs)[:n]
        risk[str(int(keep * 100))] = {"retained": n, "accuracy": float(truth[order].mean())}
    return {"n": int(len(probs)), "auroc": auroc, "auprc": auprc, "brier": brier, "ece": ece, "risk_coverage": risk}


def distractor_metrics(model, name: str, data: dict, n: int):
    valid = valid_mask(data) & data["hard"]
    # Distractors are appended before NEW; only rows with enough extra options
    # are estimable.  For an existing target its index is unchanged.
    available = data["extra_mask"][:, :n].all(-1)
    valid = valid & available
    if not bool(valid.any()):
        return {"n": 0, "accuracy": None, "target_probability": None}
    q = data["query"][valid]
    base_opt = data["option"][valid, :-1]
    extra = data["extra_option"][valid, :n]
    new = data["option"][valid, -1:]
    options = torch.cat([base_opt, extra, new], dim=1)
    mask = torch.ones(options.shape[:2], dtype=torch.bool, device=options.device)
    logits = model(q, options, mask) if name in ("B1_LINEAR", "B2_MLP", "B3_SET_CHOICE") else model.choice(q, options, mask)
    target = data["target"][valid]
    probs = torch.softmax(logits, -1)
    return {"n": int(len(target)), "accuracy": float((probs.argmax(-1) == target).float().mean()),
            "target_probability": float(probs[torch.arange(len(target), device=target.device), target].mean())}


def permutation_test(model, data: dict, n: int = 1000):
    valid = valid_mask(data)
    indices = torch.where(valid)[0][:n]
    if not len(indices):
        return {"n": 0, "max_abs_probability_difference": None}
    q = data["query"][indices]
    opt = data["option"][indices].clone()
    mask = data["option_mask"][indices].clone()
    original = torch.softmax(model(q, opt, mask), -1)
    generator = torch.Generator(device=opt.device).manual_seed(12345)
    for row in range(len(opt)):
        perm = torch.randperm(opt.shape[1] - 1, generator=generator, device=opt.device)
        opt[row, :-1] = opt[row, :-1][perm]
        mask_row = mask[row, :-1].clone()
        mask[row, :-1] = mask_row[perm]
    permuted = torch.softmax(model(q, opt, mask), -1)
    # Undo the permutation before comparing semantic option probabilities.
    # Re-run with the same deterministic permutations to map positions back.
    generator = torch.Generator(device=opt.device).manual_seed(12345)
    restored = permuted.clone()
    for row in range(len(opt)):
        perm = torch.randperm(opt.shape[1] - 1, generator=generator, device=opt.device)
        inverse = torch.argsort(perm)
        restored[row, :-1] = permuted[row, :-1][inverse]
    return {"n": int(len(indices)), "max_abs_probability_difference": float((original - restored).abs().max())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train_raw = load_cache(args.cache / "train.pt", device)
    cal_raw = load_cache(args.cache / "calibration.pt", device)
    dev_raw = load_cache(args.cache / "dev.pt", device)
    stats = fit_standardization(train_raw)
    train, cal, dev = (standardize(x, stats, device) for x in (train_raw, cal_raw, dev_raw))
    qdim, odim = train["query"].shape[1], train["option"].shape[2]
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(stats, args.output / "feature_standardization.pt")
    results = {"models": {}, "K": int(train_raw["K"]), "seeds": list(SEEDS)}
    verification = {}
    distractors = {name: {} for name in ("B2_MLP", "B3_SET_CHOICE", "B4_SET_CHOICE_VERIFY")}
    permutation = {}

    for name in MODEL_NAMES:
        results["models"][name] = {"seeds": {}}
        for seed in SEEDS:
            seed_all(seed)
            model = make_model(name, qdim, odim).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            indices = torch.where(train["target"] >= 0)[0]
            # Cached tensors already fit in memory; mini-batches keep the
            # attention head independent of total record count.
            generator = torch.Generator(device="cpu").manual_seed(seed)
            for epoch in range(args.epochs):
                order = indices[torch.randperm(len(indices), generator=generator, device=indices.device)]
                for start in range(0, len(order), 512):
                    batch_idx = order[start:start + 512]
                    batch = {key: value[batch_idx] for key, value in train.items() if isinstance(value, torch.Tensor)}
                    logits, verify_logits = forward(model, name, batch)
                    loss = choice_loss(logits, batch["target"], batch["option_mask"])
                    if name == "B4_SET_CHOICE_VERIFY":
                        valid = batch["target"] >= 0
                        pred = logits.argmax(-1)
                        verify_target = (pred == batch["target"]).float()
                        loss = loss + 0.5 * nn.functional.binary_cross_entropy_with_logits(verify_logits[valid], verify_target[valid])
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            model.eval()
            with torch.no_grad():
                dev_logits, dev_verify = forward(model, name, dev)
                cal_logits, _ = forward(model, name, cal)
            dev_metrics = metric_bundle(dev_logits, dev["target"], dev["option_mask"])
            hard_metrics = metric_bundle(
                dev_logits,
                dev["target"],
                dev["option_mask"],
                row_filter=dev["hard"],
            )
            temperature = fit_temperature(cal_logits, cal["target"], cal["option_mask"])
            calibrated = metric_bundle(dev_logits / temperature, dev["target"], dev["option_mask"])
            row = {
                "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
                "train_supervised": int((train["target"] >= 0).sum()),
                "dev_all": dev_metrics, "dev_hard_recoverable": hard_metrics,
                "dev_calibrated": calibrated, "temperature": temperature,
            }
            results["models"][name]["seeds"][str(seed)] = row
            if name == "B4_SET_CHOICE_VERIFY":
                verification[str(seed)] = uncertainty_metrics(dev_verify, dev_logits, dev["target"])
            if name in distractors:
                for count in (1, 2, 4):
                    distractors[name].setdefault(str(seed), {})[str(count)] = distractor_metrics(model, name, dev, count)
            if name == "B3_SET_CHOICE":
                permutation[str(seed)] = permutation_test(model, dev)
            torch.save({"model": model.state_dict(), "name": name, "seed": seed, "temperature": temperature}, args.output / f"{name}.seed{seed}.pt")

    results["gmt_baseline"] = baseline_metric(dev)
    # B0.5 uses released normalized GMT score without changing its argmax.
    baseline_logits = dev["option"][:, :, 2]
    cal_baseline_logits = cal["option"][:, :, 2]
    b0_temp = fit_temperature(cal_baseline_logits, cal["target"], cal["option_mask"])
    results["gmt_calibrated"] = {"temperature": b0_temp, "dev": metric_bundle(baseline_logits / b0_temp, dev["target"], dev["option_mask"])}
    results["permutation_test"] = permutation
    (args.output / "offline_results.json").write_text(json.dumps(results, indent=2, sort_keys=True))
    (args.output / "verification_results.json").write_text(json.dumps(verification, indent=2, sort_keys=True))
    (args.output / "distractor_results.json").write_text(json.dumps(distractors, indent=2, sort_keys=True))
    (args.output / "CALIBRATION_RESULTS.md").write_text(render_calibration(results))
    (args.output / "OFFLINE_DECISION_RESULTS.md").write_text(render_offline(results))
    (args.output / "VERIFICATION_RESULTS.md").write_text(render_verification(verification))
    (args.output / "DISTRACTOR_RESULTS.md").write_text(render_distractors(distractors))
    print(json.dumps(results, indent=2, sort_keys=True))


def render_offline(results: dict) -> str:
    lines = ["# Offline decision results", "", f"Frozen K: `{results['K']}`", "", "| model | seed | all accuracy | hard accuracy | NLL | Brier | ECE |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name, info in results["models"].items():
        for seed, row in info["seeds"].items():
            allm, hard = row["dev_all"], row["dev_hard_recoverable"]
            lines.append(f"| {name} | {seed} | {allm['accuracy']:.6f} | {hard['accuracy'] if hard['accuracy'] is not None else 'n/a'} | {allm['nll']:.6f} | {allm['brier']:.6f} | {allm['ece']:.6f} |")
    lines += ["", "Candidate misses are retained in coverage accounting and excluded from supervised choice loss/evaluation because their correct active ID is not in frozen Top-K.", ""]
    return "\n".join(lines)


def render_calibration(results: dict) -> str:
    lines = ["# Calibration results", "", "Temperature is fitted only on JEV-calibration scenes after model training.", "", "| model | seed | temperature | dev calibrated NLL | Brier | ECE |", "|---|---:|---:|---:|---:|---:|"]
    for name, info in results["models"].items():
        for seed, row in info["seeds"].items():
            metric = row["dev_calibrated"]
            lines.append(f"| {name} | {seed} | {row['temperature']:.6f} | {metric['nll']:.6f} | {metric['brier']:.6f} | {metric['ece']:.6f} |")
    lines.append(f"| B0.5 calibrated GMT | - | {results['gmt_calibrated']['temperature']:.6f} | {results['gmt_calibrated']['dev']['nll']:.6f} | {results['gmt_calibrated']['dev']['brier']:.6f} | {results['gmt_calibrated']['dev']['ece']:.6f} |")
    return "\n".join(lines) + "\n"


def render_verification(verification: dict) -> str:
    lines = ["# Verification uncertainty diagnostics", "", "Verification is diagnostic only; it does not alter online tracking.", "", "| seed | n | AUROC | AUPRC | Brier | ECE |", "|---:|---:|---:|---:|---:|---:|"]
    for seed, row in verification.items():
        lines.append(f"| {seed} | {row['n']} | {row['auroc']} | {row['auprc']} | {row['brier']} | {row['ece']} |")
    return "\n".join(lines) + "\n"


def render_distractors(distractors: dict) -> str:
    lines = ["# Distractor test", "", "Real low-ranked candidate options are appended before NEW on frozen dev rows.", "", "| model | seed | distractors | n | accuracy | target probability |", "|---|---:|---:|---:|---:|---:|"]
    for name, seeds in distractors.items():
        for seed, counts in seeds.items():
            for count, row in counts.items():
                lines.append(f"| {name} | {seed} | {count} | {row['n']} | {row['accuracy']} | {row['target_probability']} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
