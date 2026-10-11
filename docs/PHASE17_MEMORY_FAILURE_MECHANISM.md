# WHO / Q4 hard-memory causal attribution

The full frozen-weight native cold-start factorial exactly reproduces both prior endpoints on the three reused DEV videos. The same actual frozen Stage1 cache, weight, native candidates, Hungarian constraints, lawful Bank and subsequent commits are used. B0 matches Phase XV pooled metrics; B3 matches Phase XVI pooled metrics, including official unmodified MATLAB counts. No new controller training occurred. The original reports are unchanged.

|Arm|WHO|Q4|HOTA|AssA|IDF1|IDSW|CVIDF1|CVMA|
|---|---|---|---:|---:|---:|---:|---:|---:|
|B0|original|original|74.188270|74.890708|90.521042|147|90.285890|86.781450|
|B1|hard view|original|65.814516|59.244386|75.530052|267|66.998096|68.282016|
|B2|original|hard predecessor|73.572232|73.709980|89.843700|330|89.145910|85.436722|
|B3|hard view|hard predecessor|61.764421|52.052180|71.026365|1494|67.476069|72.580118|

For IDSW the baseline-relative WHO effect is +120, Q4 effect +183, and the factorial interaction `B3−B2−B1+B0` is +1044. The joint degradation is +1347. This identifies a strong feedback interaction under this frozen implementation; it is not a unique allocation of all errors to one neural pathway or a significance claim. For AssA, corresponding effects are −15.6463pp, −1.1807pp and −6.0115pp. WHO filtering alone severely damages global identity/cross-view consistency even with much lower IDSW than B3.

Phase XVI `SafeMatchingMemory.build()` sends both the selected raw Gallery rows and independently updated trusted metadata/Q4 into `trusted.build()`. The raw Gallery remains append-only, but omitted rows no longer influence this WHO/Q4 view. Cold-start pending fallback can return a genuine visual row while trusted camera metadata is still absent. Thus having the raw vector does not restore the four history slots, masks, counts, motion metadata and Q4 predecessor distribution expected by frozen XV weights. The earliest fixed consistency anchor can also keep rejecting a legitimate later viewpoint. Filtered Q4 observations can become stale or absent while the actual track continues committing: predecessor gap/run/hits/coherence and motion support then describe a different history from native raw commits.

The new adapter switches WHO (history/mask/identity metadata/pair evidence, including rule REACT visual scores) independently of `commitment_features` (Q4). B1 replaces only WHO; B2 only Q4. All decisions execute in the original native lifecycle, so future states are free to diverge. Every actual query has original and executed logits, legal reference mapping, WHO/Q4 fingerprints and fallback source indices. First local decision changes save raw/trusted state, tensors and native snapshots; all later raw query/commit/admission paths remain reconstructable from frozen perception and journals. Detailed state snapshots are capped at32 per run, not falsely described as exhaustive.

First actual ID divergences: DEV17 B1/B3 at frame0 camera0, B2 at frame38 camera0; DEV18 B1/B3 at frame280 camera1, B2 at frame488 camera1; DEV19 B1/B3 at frame3 camera1, B2 at frame106 camera1. Frame-zero bootstrap has no qualified owner. A changed ID there is not automatically a known wrong merge: aliases and UNKNOWN must be evaluated separately from the causal fact that IDs changed. Later wrong-owner episodes retain the original qualified past owner after mixtures, disclose gaps/censoring and unknown exposure. `FIRST_DIVERGENCE.json` records full first-event candidates/actions and paired CORRECT/WRONG/UNKNOWN transitions.

Cold start and common historical prefixes have different interventions. Prefix arms retain all old observations as unvalidated legacy history and filter only future writes; they cannot recreate an entirely filtered cold-start trajectory. Their24 paired starts/96 native futures are still being independently verified at this milestone. Overlapping horizons are descriptive, not independent replicates.

Hard quarantine is not eligible for training or deployment. Evidence reliability and GlobalID ownership must remain distinct; appearance consistency is not owner certification. Next complete owner identifiability, then test bounded reversible weights preserving original Q4/metadata and all raw observations. No gate is changed to reinterpret Phase XVI as success.
