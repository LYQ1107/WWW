# Candidate-conditioned runtime binding and video01 smoke

Snapshot: **2026-10-07 14:29 UTC**  
Classification: **screening only; not a final selection or paper result**

## Result

The candidate-conditioned runtime path is now wired through the mutable GMT
engine. Both checkpoints completed a bounded video01 smoke through the actual
sequence:

| Method | MATCH rows | Native candidate bindings | Invalid bindings | START_NEW |
|---|---:|---:|---:|---:|
| candidate MLP | 17 | 17 | 0 | 0 |
| candidate JEV | 17 | 17 | 0 | 0 |

The selected IDs were legal native proposal IDs, and no second transformer
call was made for candidate binding. A selected candidate column is reserved
before any constrained assignment, and an illegal or duplicate binding fails
closed.

## Scope and limitation

This was `video_id=1`, sequence `00002garden`, with `max-frame=3`. It validates
the runtime data path only; it does **not** provide HOTA/AssA/IDF1/MOTA or a
full-video quality conclusion. The candidate checkpoints were trained under
source commit `a390682...`, while the runtime smoke used
`c9b7d61c...`; the explicit source-mismatch override was therefore required
and is recorded in the JSON report. A fresh frozen-source trace is still
required before any official candidate runtime claim.

The bounded GMT OFF replay sampled 33 state-feature records with maximum
absolute error `5.96e-8` and zero OFF action mismatches. The full 8995-record
parity gate was not claimed by this smoke.

## Formal video01 evidence

The preceding corrected v2 builder ran from `04:15:18Z` to `11:10:35Z`,
elapsed **06:55:17.54**, and durably wrote **8995/8995** records. Its JSONL
SHA-256 is:

```text
1da2ed9e7eec43c7cf139754d873f78a299c344ff74d65c6866b92f18f3da79
```

That artifact is evidence of completed computation, not semantic acceptance:
aftercare returned `NO_GO_RUNTIME_SEMANTICS_GATE` because runtime feature parity,
reactivation candidate parity, and repeated stability failed. No full H8 build
is authorized.

Machine-readable details are in
[`CANDIDATE_RUNTIME_SMOKE_VIDEO01_20261007.json`](../reports/JEV_RNG_V4/CANDIDATE_RUNTIME_SMOKE_VIDEO01_20261007.json)
and
[`VIDEO01_SIX_HOUR_RUNTIME_EVIDENCE_20261007.json`](../reports/JEV_RNG_V4/VIDEO01_SIX_HOUR_RUNTIME_EVIDENCE_20261007.json).

## Local raw artifacts

The complete smoke outputs remain on the runtime volume:

```text
/home/liuyeqiang/WWW_jev_rng_v4_runtime/candidate_runtime_smoke_mlp_20261007/
/home/liuyeqiang/WWW_jev_rng_v4_runtime/candidate_runtime_smoke_jev_20261007/
```

Their SHA-256 values and provenance are recorded in the JSON report above.
