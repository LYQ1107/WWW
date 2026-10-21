"""Select the fixed Stage-A epoch/seed reports without inspecting TEST."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "manifests/omega_full24_training_protocol_v1.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text())
    models = ["native", "omega", "dino"]
    seeds = protocol["stages"]["A"]["seeds"]
    rows = []
    selected = {}
    for model in models:
        candidates = []
        for seed in seeds:
            path = args.root / f"{model}_seed{seed}" / "manifest.json"
            report = json.loads(path.read_text())
            if report.get("status") != "COMPLETE_STAGE_A_HEAD_TRAINING":
                raise ValueError(f"incomplete Stage-A report: {path}")
            for curve in report["curves"]:
                validation = curve["validation"]
                candidates.append({"model": model, "seed": seed, "epoch": curve["epoch"], "validation_loss": validation["listwise_loss"], "net_fixes": validation["net_fixes_vs_native"], "fixes": validation["fixes_vs_native"], "harms": validation["harms_vs_native"], "report": str(path)})
        best = min(candidates, key=lambda row: (row["validation_loss"], row["epoch"], row["seed"]))
        primary = next(row for row in candidates if row["seed"] == protocol["stages"]["A"]["primary_seed"] and row["epoch"] == best["epoch"])
        selected[model] = {"selected_epoch": best["epoch"], "selected_seed": best["seed"], "selected": best, "primary_seed_at_selected_epoch": primary}
        rows.extend(candidates)
    primary_omega = selected["omega"]["primary_seed_at_selected_epoch"]
    gate = primary_omega["fixes"] > primary_omega["harms"] and primary_omega["net_fixes"] > 0
    result = {
        "status": "PASS_STAGE_A_SELECTION_GATE" if gate else "FAIL_STAGE_A_SELECTION_GATE",
        "protocol_sha256": __import__("hashlib").sha256(PROTOCOL.read_bytes()).hexdigest(),
        "fit_scene_ids": protocol["scene_split"]["fit_video_ids"], "validation_scene_ids": protocol["scene_split"]["validation_video_ids"],
        "selection_rule": "minimum validation listwise loss per model across fixed epochs/seeds; Stage-B uses primary seed and selected epoch; no TEST access",
        "selected": selected, "all_candidates": rows,
        "omega_primary_identity_gate": gate,
        "next_action": "Stage-B all24 refit is permitted for rows with complete frozen evidence; GMT integration remains closed until causal validation metrics pass" if gate else "do not integrate Ω into GMT; retain native baseline and controls",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "selected": selected, "omega_primary_identity_gate": gate}), flush=True)


if __name__ == "__main__":
    main()
