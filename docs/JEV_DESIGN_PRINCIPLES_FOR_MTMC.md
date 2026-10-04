# Design principles for JEV in MTMC tracking

JEV is a typed state-transition policy layer around GMT. It is not a detector,
an identity classifier, or a candidate-ID predictor.

1. **Perception and policy are separated.** Detector boxes, scores, ReID
   features, and image metadata are inputs. JEV receives online state features
   and a typed question.
2. **GMT owns identity proposals.** Hungarian/association logic proposes
   track identities. JEV can choose `ACCEPT_CURRENT`, `REASSOCIATE`, or
   `START_NEW`, but never emits an arbitrary track ID.
3. **Actions are question-masked.** Memory and reactivation actions are not
   legal in a match question. Probabilities are normalized only over runtime
   legal actions.
4. **Only online information enters the controller.** No GT ID, future
   trajectory, evaluator metric, absolute sequence ID, absolute track ID, or
   file path may be a feature.
5. **Persistent state is part of the problem.** Memory writes, stale-bank
   membership, collisions, switches, and fragmentation are state transitions,
   not post-hoc bookkeeping.
6. **Counterfactual branches start from one pre-action state.** Each branch is
   a deep copy. A branch cannot mutate its sibling or the source state.
7. **Association is globally coupled.** A rejected edge is masked and the full
   current assignment is recomputed once; local second-best selection is not a
   formal implementation.
8. **Selection is sequence-disjoint and val-only.** Architecture, hidden size,
   calibration, and thresholds are selected on policy validation sequences.
   Official test is read only after the final-selection lock.

The v2 implementation keeps these principles explicit in code contracts and
manifests so that a positive tracking number can be audited back to a fixed
checkpoint, state schema, cache, and policy split.

