# Persistent World Query design

`PersistentWorldQueryTransformer` keeps a fixed `Q_max=128` slot bank.  Each
state has an embedding, normalized xyz/velocity, diagonal covariance,
appearance memory, existence, per-camera visibility, age, missed count,
last-seen time, slot ID, and runtime global ID.

Each decoder layer performs query self-attention, camera-agnostic observation
cross-attention with a learned relative-geometry bias, and a feed-forward
query update.  Assignment logits include an explicit null column.  Birth and
visibility heads are separate from association logits.  `constrained_assign`
is a post-decoder validity constraint: no observation has two owners, and a
query has at most one owner in each camera but may own observations from both
cameras.

When an observation is not assigned and its birth logit passes the threshold,
the runtime allocates an unused slot and a new global ID.  Missing assigned
queries are retained through `max_missed` and then archived; an archived ID is
never allocated again.  Projection-only boxes are never marked as real
observations.

