# JEV Counterfactual Dataset Contract

## 目标

训练数据必须描述“同一个在线状态下，不同合法动作在未来 H 帧造成什么结果”，而不是把当前帧的 Hungarian winner 当作分类标签。该数据集用于 MATCH、MEMORY、REACTIVATION 三类 typed question，并且必须可重放、可审计、可按序列隔离。

## 一条 state record

推荐 JSONL 记录如下字段：

```json
{
  "schema_version": 1,
  "dataset": "VisionTrack",
  "sequence": "000xxscene",
  "frame": 123,
  "view": 1,
  "question_type": "MATCH_DECISION",
  "state_digest": "sha256:...",
  "gmt_checkpoint_sha256": "sha256:...",
  "legal_actions": ["ACCEPT_CURRENT", "REASSOCIATE", "START_NEW"],
  "state": {
    "observation": {},
    "gmt_evidence": {},
    "track_summary": {},
    "memory_summary": {},
    "disagreement": {}
  },
  "action_outcomes": {
    "ACCEPT_CURRENT": {"utility": 0.0, "metrics": {}, "rollout_digest": "..."},
    "REASSOCIATE": {"utility": 0.0, "metrics": {}, "rollout_digest": "..."},
    "START_NEW": {"utility": 0.0, "metrics": {}, "rollout_digest": "..."}
  },
  "best_actions": ["ACCEPT_CURRENT"],
  "target_probs": [0.6, 0.1, 0.3],
  "horizon": 8,
  "uses_future_gt": true
}
```

The stored state must not contain future GT. `uses_future_gt=true` is permitted only in the offline counterfactual builder and labeler; strict online consumers must reject such records.

## State representation

### Observation

- current frame/view and normalized box geometry;
- objectness/class confidence;
- current ReID embedding summary or frozen projection;
- number of current proposals and empty-set flags.

### GMT evidence

- raw association logits and activated probabilities;
- unmatched mass before `_activate_asso` drops its column;
- top-1/top-2 proposal and track scores;
- Hungarian candidate, margin and conflict count;
- temporal/spatial/camera features already used by GMT.

### Track/memory summary

- track age, hits, last seen, view count;
- recent association score statistics;
- ReID prototype/variance and memory age;
- number of stale candidates and bank capacity;
- current `id_count` only as a normalized state statistic, never as action label.

### Disagreement

- old threshold action vs association-only action;
- `WITH_BANK` off/on proposal;
- current-view vs cross-view assignment;
- detector/GMT confidence gap and calibration bucket.

## Action outcome generation

For every legal action, restore the same state snapshot and run the same frozen future evidence for H frames. The branch runner must:

1. deep-copy mutable `Instances` and tracker containers;
2. apply only the current typed action;
3. use the same detector/GMT outputs and frame cursor for every branch;
4. commit state through one adapter;
5. collect per-frame ID mapping, memory writes, reactivation and recovery events;
6. compute local and H-step metrics against future GT;
7. restore/discard the branch before evaluating the next action.

The branch runner must assert that action selection does not call detector training, alter frozen weights, or change the candidate legal set after the action is selected.

## Utility target

Store raw metrics before scalarization. At minimum:

```text
utility = w_assa * delta_assa
        + w_idf1 * delta_idf1
        - w_idsw * delta_idsw
        - w_frag * delta_frag
        - w_contam * contamination_frames
        + w_recovery * recovery_gain
        - w_runtime * extra_runtime
```

Normalize utility only within the same state and legal action set. `target_probs` should be a temperature-controlled softmax over normalized utilities with explicit tie handling; never compare utilities from different sequences without normalization metadata.

## Splits and leakage controls

- Split by sequence/scene, never by adjacent frames from the same sequence.
- Keep a separate held-out calibration split; do not fit temperature on test.
- Keep future GT and evaluator-only metadata out of state features.
- Hash frozen checkpoint, config, source commit and dataset manifest into every shard.
- Record seed, H, action order, tie-break and legal-mask version.
- Verify that a model cannot infer sequence identity from an absolute track ID or file path.

## Required validation

Before training a policy:

1. replay the same state twice and compare state/action digests;
2. permute legal action order and require probability permutation equivariance;
3. remove each action from legal set and verify its probability is zero/absent;
4. compare H=1 with the old frame-local label only as a diagnostic, not as the target definition;
5. verify branch mutation isolation by hashing original state before/after each rollout;
6. verify that a no-future-GT process cannot import the labeler module;
7. check action frequencies and long-tail question types before fitting the head.

## Training targets

The recommended curriculum is:

1. best-action cross entropy on high-margin oracle states;
2. soft utility KL on all legal actions;
3. held-out temperature/calibration fit;
4. optional outcome/reference-KL fine-tuning with sequence-level grouping.

Do not train on instantaneous “proposal matched GT” labels alone: they reward greedy ID correctness and cannot teach contamination duration, recovery or memory cost.

## Current trace labeler

`reproduction_tools/build_jev_counterfactual_dataset.py` consumes the
append-only trace from `MODEL.JEV.MODE=off` and reads future GT only while
scoring isolated offline branches. It emits strict JSONL plus a manifest:

```text
PYTHONPATH=/data1/liuyeqiang/WWW:/data1/liuyeqiang/WWW/third_party/CenterNet2:/data1/liuyeqiang/WWW/reproduction_tools \
  /home/liuyeqiang/anaconda3/envs/GMT/bin/python \
  reproduction_tools/build_jev_counterfactual_dataset.py \
  --trace outputs/jev_off_trace.jsonl \
  --annotations /data/DATASETS/TRACKING/JDE/VisionTrack/annotations/train.json \
  --gmt-checkpoint outputs/stage2_single_gpu/model_20000.pth \
  --output outputs/jev_counterfactual/train.jsonl \
  --horizon 32
```

The current implementation is a reproducible frozen-evidence mutable-state
rollout (`frozen_evidence_mutable_gmt_state_v1`). The real OFF trace records
the GMT tracker-container snapshot at each decision boundary; the offline
runner deep-copies track assignments, hit/memory counters, active/stale-bank
membership, and replay state before applying each legal action. Future GMT
decision evidence is then replayed identically for every branch, and future
GT is used only by the utility callback.

This is stronger than the old feature-only proxy, but it still deliberately
does not re-run detector or association networks and does not serialize full
Detectron2 `Instances` tensors. Formal reports must therefore label it
“frozen-evidence mutable GMT-state rollout”, not a full mutable-`Instances`
re-forward or a full TrackEval utility. The distinction is recorded in every
dataset manifest and is a required limitation until a full tensor-level branch
runner is available.
