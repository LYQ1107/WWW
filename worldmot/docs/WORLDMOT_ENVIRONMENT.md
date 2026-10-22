# Environment record

The host has ten Tesla V100-DGXS-32GB devices. At the initial audit GPU1 contained an unrelated 12,133 MiB process; GPU0 and GPU2–9 were idle. No process was terminated. The project uses an isolated Python 3.11 environment under `/data1/liuyeqiang/envs/worldmot311` and does not modify the GMT environment.

VisionTrack TRAIN is already extracted at `/data/DATASETS/TRACKING/JDE/VisionTrack/train`, with its manifest at `annotations/train.json` (24 videos, 57,508 images, two views per video). The adapter accesses this path read-only.

Run `scripts/audit_environment.py` before a benchmark and keep its JSON beside the run. The audit also searches WorldMOT source for forbidden `gtr` and `detectron2` imports.
