# Corrected video1 recovery gate — 2026-10-06

This is an implementation and smoke-gate report, not an official tracking
result.

## What failed

The corrected video1 builder stopped at production key `(video=1, frame=214,
view=1)`. The replay retained an 80-slice history, but native GMT with
`VISION_test.yaml INPUT.VIDEO.TEST_LEN=40` had already removed the oldest frame
before proposing the current view. The old identity therefore remained in
`active_ids` instead of being promoted to the native stale bank.

The first retry after fixing that window exposed a second independent issue:
the association adapter's payload cache was keyed only by video/frame/view.
Consequently, a one-row reactivation payload could reuse a previously cached
full-frame tensor.

## Changes in `f2e9c16`

- Added the shared `sync_production_history_for_key()` helper.
- Applied native-equivalent history synchronization before ordinary proposals,
  reactivation discovery, future branch proposals, chunk warm-up, and pilot
  runtime replay.
- Recomputed `active_ids` from the synchronized window.
- Added `source_detection_indices` to the adapter payload-cache key.
- Added regression tests for both the 79-slice window at `(214,1)` and the
  full-payload/subset-payload cache collision.

## Evidence

The formal GMT short replay used the same video1 trace, frozen perception
cache, model-20000 checkpoint, and adapter as the worker. On GPU9 it processed
indices 1–430; index 429 is exactly `(1,214,1)`, and index 430 is `(1,215,0)`.
It completed with `SHORT_REPLAY_PASS`, zero emitted records (intentional
state-only gate), and no exception.

The detailed machine-readable record is
`reports/JEV_RNG_V4/VIDEO01_DYNAMIC_HISTORY_REACTIVATION_FIX.json`.

## Safety boundary

The failed video1 temporary files remain preserved. Formal v9, video6, and the
existing waiters were not stopped or migrated. No full GMT baseline or official
full tracking evaluation was rerun.

The next operation is limited to resetting the failed video1 queue entry and
starting a new GPU8 worker from the pushed `f2e9c16` source.
