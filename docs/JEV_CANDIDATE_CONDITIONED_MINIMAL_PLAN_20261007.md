# Candidate-conditioned JEV: minimal follow-up plan

Status: planning only, not a model result and not an authorization for Full H8.
Created 2026-10-07 under source commit `4108f18f5040432f68d55872e81e9f76e9acd08f`.

## Evidence and limitation

The read-only video-1 feasibility audit found 174 reactivation events with 559
finite candidate scores.  Candidate IDs and scores are present in the fresh
native trace, but the available action-level records contain returns only for
`REACTIVATE_OLD` versus `START_NEW`; they do not contain verified
candidate-to-ground-truth identity alignment or per-candidate long-term
returns.  Candidate recall, correct rank, and candidate-level oracle headroom
must therefore remain unreported until those two missing pieces are produced.

## Minimal data/interface additions

1. Preserve the existing canonical mutable GMT state and event key
   `(video_id, frame, view, event_order, detection_index)`.
2. Persist the legal runtime candidate set in native order, candidate scores,
   proposal metadata, and a stable candidate-row key.  Never encode an
   absolute track ID as a fixed class index or learned feature.
3. For offline audit only, attach a validated identity correspondence for each
   candidate and the same future rollout utility used by the action-level
   dataset.  If correspondence is unavailable, mark the metric unavailable;
   do not infer it from the numeric ID namespace.
4. At runtime, let a shared scorer evaluate only the candidates legal under the
   current GMT proposal.  The final identity submission remains constrained by
   GMT view/matching semantics; the policy cannot invent an arbitrary track.

## Small controlled comparison

Use one fixed seed first (`20261003`) and exactly the same perception cache,
candidate sets, event keys, train/validation sequence split, targets, sample
weights, optimizer, learning rate, batch size, epochs, and independent test
sequences for all three methods:

* existing gated JEV (current action/state interface);
* ordinary candidate MLP (candidate rows scored from shared state and candidate
  evidence);
* candidate-conditioned JEV (shared state plus candidate-conditioned evidence,
  with legal-set masking).

Select architecture and hyperparameters using policy train/validation only.
Keep the tracking test sequences completely independent. Report validation NLL,
best-action/candidate accuracy where defined, Brier, ECE, utility, candidate
recall, correct rank, and feasible-set oracle headroom. Then run a small
GMT-OFF / threshold / candidate-MLP / candidate-JEV closed loop with identical
perception and evaluation mapping.

## Required gates

Before any Full H8 authorization, the selected interface must pass fresh
artifact provenance, MATCH/MEMORY/REACTIVATION runtime parity, reactivation
candidate parity, numerical stability, checkpoint-wrapper loading, and a
held-out closed-loop gate.  Any parity failure is `BLOCKED` evidence, not model
failure.  A changed action/state interface invalidates inherited GO status and
requires the full small-gate sequence again.

## Explicit non-goals

This plan does not authorize stopping or migrating the current video1 v2 job,
does not authorize a full-data rebuild, and does not copy
`ACCEPT_CURRENT`/`REASSOCIATE`/`START_NEW` labels into per-candidate supervision.
