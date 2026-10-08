# Native Causal Lifecycle MiniSet v1 — bounded, conditional protocol

P0.5 is the execution gate. No record generation below starts until its attribution report is
complete. Existing Phase V historical labels never enter this dataset as native truth.

Training sequence 07 and validation sequence 06 retain their earlier roles. Collect 16 MATCH
events per sequence: split exogenous sequence duration into four equal frame bins; choose the
first four payloads in each bin with a real legal alternate identity, and within each choose
the smallest top1/top2 score margin (row index breaks ties). No GT or utility drives selection.
Store all event counts and bin sampling probabilities. Empty bins remain empty; no replacement
video or validation outcome selection.

Clone exact B2-native prefix/history/bank/metadata/trajectory RNG. Current row interventions
ACCEPT, REASSOCIATE, NEW retain other factual first actions; all use unchanged native global
solve and B2 binary validation, then common live B2 continuation. Include an unforced CONTROL
and factual reference; never use a cached factual future action map. Horizons H8/H16/H32 mean
current frame through current+H inclusive. Store frame count and truncation explicitly.
At the first two multi-row conflict groups, add bounded joint rejection interventions; do not
assume edge utilities add to Hungarian utility.

Offline target identity is IoU>=.5 one-to-one under cache0_annotation1. Use a fixed prefix
identity map; newly born IDs inherit their first known offline target for evaluation only.
Utility follows the preserved bounded native diagnostic convention: correct target observations
minus wrong target observations minus .5 identity switches minus .25 new target IDs minus .5
within-camera collisions minus known identity contamination over all affected observations.
This scalar utility is a supervision definition, not HOTA/AssA. Unknown target records stay
zero-weight/censored; candidate misses are labelled CANDIDATE_MISS. Preserve ties and branch
hashes. Future annotations never enter candidates, features, actions or commits.

MATCH fitting eligibility requires exact factual/control replay and transition parity, finite
outcomes, no leakage/duplicate/source issues, at least 12 informative train and 6 informative
validation events, and at least two unique-winning actions with ≥3 train and ≥2 validation
examples each. Also require ≥3 train and ≥2 validation opportunities where a correct identity
alternative is actually present and the factual proposal is wrong. Insufficient candidate/label
coverage blocks new candidate architecture fitting even if many forced bad actions diverge.
These support thresholds qualify only a bounded pilot, not generalization or Unified claims.

MEMORY: first four known-target READ-enriched canonical Phase VI captures per video (24/23),
ordered by event key, independent of utility; paired WRITE/SKIP with common H8/H16/H32/H64
windows. Record prototype/bank hashes, actual reads, score changes, assignment changes and
censoring. READ enrichment is separate from the unscreened event population. A longer window
cannot be selected separately for each branch. If utility remains indistinguishable, mark
BLOCKED_MEMORY_IDENTIFIABILITY; never create nonzero targets.

REACT: audit all existing 1,503 events and canonical 32/16 records. Correct-candidate recall,
single/multiple stale candidates, unknown GT and train/val utility distributions are required.
The relative production hook is absent, so a relative learned production MiniSet remains
BLOCKED_NATIVE_RELATIVE_REACT_HOOK until committed-state parity is established. Preserve
native binary space; do not add synthetic candidates or refit the failed Phase VI head.

Only eligible MATCH records become train/val MiniSet inputs. All other questions are explicitly
blocked, not absent pretending a successful three-question dataset. A loader checks semantic
candidate IDs/order, legal actions, train/val roles, finite probabilities/utilities, source hashes,
prefix/control replay, state independence and offline-only GT fields. Check records/checkpoints
into the evidence archive with manifests; official TEST and Full24 remain untouched.

WHAT DID WE LEARN? A forced bad branch changing state is insufficient learning opportunity.
The data must contain feasible corrective alternatives and multiple supported optimal actions.
