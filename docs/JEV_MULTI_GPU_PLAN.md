# Multi-GPU experiment plan

The canonical Stage2 process remains on the original worktree/GPU and is not
restarted for v2. The branch `/data1/liuyeqiang/WWW_jev_v2` is isolated from
that runtime.

Suggested scheduling after the fixed checkpoint/cache gates:

- **GPU0:** canonical Stage2 completion and checkpoint validation;
- **GPU1:** frozen perception cache and v2 counterfactual/Oracle generation;
- **GPU2:** policy train/validation suites and calibration;
- **GPU3:** OFF/SHADOW equivalence and trace audits;
- **GPU4–5:** baseline/ablation training;
- **GPU6:** official-test inference only after `FINAL_SELECTION_LOCK.json`;
- **GPU7:** final v2 long-horizon and stress analysis.

Each job records GPU assignment, checkpoint hash, code commit, cache hash,
policy split hash, and output manifest. No DDP is used to accelerate a single
Stage2 task; independent jobs are the safe unit of parallelism.

