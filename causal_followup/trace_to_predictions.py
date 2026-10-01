#!/usr/bin/env python
"""Convert a post-filter replay trace and cache into COCO-style predictions."""
from __future__ import annotations
import json, sys
from pathlib import Path
import torch

def main():
    cache = Path(sys.argv[1]); trace = Path(sys.argv[2]); out = Path(sys.argv[3])
    preds = []
    for cp in sorted(cache.glob("*.pt")):
        payload = torch.load(cp, map_location="cpu")
        rows = payload["records"]
        tp = torch.load(trace / cp.name, map_location="cpu")
        for tr in tp["records"]:
            seq = int(tr["sequence_index"])
            if seq >= len(rows): raise RuntimeError(f"trace/cache sequence mismatch {cp.name}:{seq}")
            row = rows[seq]
            boxes = tr["pred_boxes"].tolist() if tr["pred_boxes"] is not None else []
            scores = tr["scores"].tolist() if tr["scores"] is not None else []
            classes = tr["pred_classes"].tolist() if tr["pred_classes"] is not None else []
            ids = tr["track_ids"].tolist()
            ih, iw = [float(x) for x in row["image_size"]]
            sx, sy = float(row["width"]) / max(iw, 1.0), float(row["height"]) / max(ih, 1.0)
            for b, s, c, tid in zip(boxes, scores, classes, ids):
                x1,y1,x2,y2 = b
                preds.append({"image_id": int(row["image_id"]), "category_id": int(c)+1,
                              "bbox": [x1*sx,y1*sy,(x2-x1)*sx,(y2-y1)*sy],
                              "score": float(s), "track_id": int(tid)})
    out.write_text(json.dumps(preds, separators=(",", ":")) + "\n")
    print(json.dumps({"predictions": len(preds), "output": str(out)}, indent=2))

if __name__ == "__main__": main()
