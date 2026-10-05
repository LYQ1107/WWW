# Branch consolidation plan

This plan keeps the remote `main` history intact while making the research
state auditable.

## Current branches

- `jev/reviewer-proof-v2`: active formal v2 implementation and protocol gates.
- `codex/formal-research-progress-20261005`: published progress snapshot.
- `codex/formal-cache-prefetch-20261005`: published cache optimization chain.
- older `codex/*20261004*` and shard-specific branches: historical evidence and
  should not be merged blindly.

## Consolidation order

1. Finish the active v2 code and invariant tests on
   `jev/reviewer-proof-v2`.
2. Commit only source, manifests/schema examples, and documentation; never add
   checkpoints, datasets, credentials, traces, or result directories.
3. Rebase/merge the active branch into a new review branch from the current
   remote `main` after verifying `git fetch --all` and recording both tips.
4. Push the review branch and attach its GitHub URL. Do not rewrite remote
   `main` or delete historical branches.
5. After reviewer inspection, merge only the source/docs commits; keep large
   experiment evidence referenced by absolute local paths and hashes.

## Required pre-merge checks

- canonical checkpoint hash and validation JSON;
- strict same-GPU OFF gate plus diagnostic cross-GPU report labels;
- policy split has no TEST files and selection manifest is validation-only;
- final lock digest, code commit, and selection-manifest digest agree;
- `git diff --check`, Python compile, and targeted invariant tests pass;
- working-tree user files remain untracked/preserved.
