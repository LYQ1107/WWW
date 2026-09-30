#!/usr/bin/env python3
"""Plot the predeclared A3 history-length sweep after it has run."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
df = pd.read_csv(root / "audit/results/A3_history_sweep.csv")
out = root / "audit/figures"
out.mkdir(parents=True, exist_ok=True)
for metric, name, ylabel in (("AssA", "A3_history_vs_AssA.png", "AssA"),
                             ("IDF1", "A3_history_vs_IDF1.png", "IDF1")):
    fig, ax = plt.subplots(figsize=(6, 4))
    on = df[df["bank_enabled"] == True].sort_values("test_len")
    ax.plot(on["test_len"], on[metric], marker="o", label="bank on")
    off = df[df["bank_enabled"] == False]
    if not off.empty:
        ax.scatter(off["test_len"], off[metric], marker="x", s=70, label="bank off")
    ax.set_xlabel("TEST_LEN")
    ax.set_ylabel(ylabel)
    ax.set_title(f"A3 history length vs {ylabel}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / name, dpi=140)
    plt.close(fig)
