#!/usr/bin/env python3
"""Scientific plots of completed TRAIN selection epochs; no model decisions."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METHODS={"FULL_WORLD_OBJECT_V3":"Full V3","FULL_NO_GEOMETRY_V3":"Full no-geometry", "HEAD_ONLY_NO_GEOMETRY_V3":"Head-only no-geometry"}


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--root",required=True,type=Path);parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(2,3,figsize=(14,8));proofs=[];drawn=0
    for method,label in METHODS.items():
        directory=args.root/"train"/method;path=directory/"train_gate.json"
        if not path.exists():path=directory/"progress.json"
        if not path.exists():continue
        data=json.loads(path.read_text());history=data["selection_history"]
        if not history:continue
        x=[row["epoch"] for row in history];val=[row["validation"] for row in history]
        split=[100*v["diagnostics"].get("same_person_split",0)/max(1,v["diagnostics"].get("same_person_cross_view_pairs",0)) for v in val]
        merge=[1000*v["diagnostics"].get("false_cross_view_merge",0)/max(1,v["diagnostics"].get("different_person_cross_view_pairs",0)) for v in val]
        axes[0,0].plot(x,[v["identity_assignment"] for v in val],"o-",label=label)
        axes[0,1].plot(x,[v["joint_birth"] for v in val],"o-",label=label)
        axes[0,2].plot(x,split,"o-",label=label)
        axes[1,0].plot(x,merge,"o-",label=label)
        axes[1,1].plot(x,[100*v["birth_proposal_recall"] for v in val],"o-",label=label)
        axes[1,2].plot(x,[100*v["birth_pair_precision"] for v in val],"o-",label=label+" precision")
        axes[1,2].plot(x,[100*v["birth_pair_recall"] for v in val],"s--",label=label+" recall")
        proofs.append({"method":method,"file":str(path),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"completed_selection_epochs":len(history)})
        drawn+=1
    labels=["Validation assignment loss","Validation joint-birth loss","Same-person split rate (%)",
            "False merges / 1,000 different-person pairs","Proposal recall on model-history-new positives (%)","Conditional birth-pair precision / recall (%)"]
    for axis,title in zip(axes.flat,labels):
        axis.set_title(title,fontsize=10);axis.set_xlabel("Completed TRAIN selection epoch")
        axis.grid(alpha=.25)
        if drawn:axis.legend(fontsize=7)
    fig.suptitle("WorldMOT V3 / fixed six-scene TRAIN validation (not TEST or official MOT scores)")
    fig.text(.02,.015,"Only completed epochs are shown. Diagnostic label mapping uses IoU 0.5; birth positives depend on causal predicted history. Loss/diagnostic improvements do not establish TEST improvement.",fontsize=8)
    fig.tight_layout(rect=(0,.04,1,.95));image=args.output/"selection_curves.png";fig.savefig(image,dpi=160);plt.close(fig)
    (args.output/"selection_curves_manifest.json").write_text(json.dumps({"source_artifacts":proofs,"plot_sha256":hashlib.sha256(image.read_bytes()).hexdigest(),"TEST_used":False,"official_MOT_metric":False},indent=2))
    print(json.dumps({"plotted_models":drawn,"output":str(image)}))


if __name__=="__main__":main()
