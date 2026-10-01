# Causal Audit v2 code audit

This audit was written before the v2 replay. It describes the released GMT
association path and the first follow-up replay without changing either
`causal/` or `causal_followup/`.

## A. The old `_frame_error` endpoint is global

`causal_followup/replay_engine.py::_frame_error(frame)` loops over every view,
then every GT-matched detection in that frame. For each matched detection it
looks up the current track ID, maps that ID through the mutable
`gt_history` majority (`_canonical(pid)`), and increments `wrong` when that
majority GT differs from the detection's GT. It therefore measures:

* all GT-matched observations in the future frame;
* across all views;
* after collapsing each predicted ID to a current majority identity.

It does not filter on the event's treated `gt_id`, `p_correct`, or `p_wrong`.
Consequently the old C1b/C2b `error_rate = wrong / total` is a global frame
error endpoint, not persistence of the treated identity. V2 retains this
quantity only as a secondary endpoint.

## B. The old C1b-C repair is partial

`_repair_identity_state(event)` in the old replay engine locates observations
for the event GT whose public `instances[seq].track_ids[det]` equals
`p_baseline` in the preceding 20 frames. It then:

1. decrements `gt_history[p_wrong][gt_id]`;
2. removes matching re-identification rows from `id_reid_dict[p_wrong]`;
3. appends copied rows with `track_ids = p_correct` to
   `id_reid_dict[p_correct]`;
4. decrements/increments `id_count_dict`;
5. removes/adds IDs in `poss_ids` using the bank-size threshold.

The old implementation does not change the public
`instances[seq].track_ids`. That preserves emitted history, but it also means
that GMT's actual sliding-window association input remains unchanged. It also
does not update `old_reids.old_reids` when the memory-bank proxy contains the
wrong or correct ID. Thus `old_reids`/memory-bank state can remain inconsistent
with the repaired `id_reid_dict`.

The v2 repair keeps public history immutable and adds an audit-only
`history_id_overrides[(sequence_index, detection_index)]`. The v2 history
filter returns a copy of an overridden instance for future association while
leaving the stored/public instance untouched. Relevant memory-bank entries are
rebuilt from the repaired `id_reid_dict`; if that cannot be done reliably, the
event is explicitly marked `C1c-bankoff` rather than silently patched.

## C. GMT reads sliding-history `track_ids`

In `gtr/modeling/meta_arch/gtr_rcnn.py`, both `run_first_tracker_plus` and
`run_global_tracker_plus` construct the association candidates as follows:

```text
ids = cat(history_instance.track_ids)
unique_ids = torch.unique(ids)
id_inds = (unique_ids[None, :] == ids[:, None]).float()
traj_score = torch.mm(association_scores, id_inds)
```

`run_global_tracker_plus` then performs Hungarian matching against
`unique_ids`, and the optional memory-bank path uses the same ID-bearing
history. Therefore a repair that changes only `id_reid_dict`,
`id_count_dict`, `gt_history`, or `poss_ids` does not repair the candidate
identity evidence actually consumed by future association. V2 applies the
internal history override before GMT receives `history`, so the effective
path is:

```text
history track_ids
  -> ids
  -> unique_ids
  -> id_inds
  -> traj_score
```

The public emitted `instances` remain unchanged; only the copied history fed
to this association path is overridden.

## D. Snapshot implications

`gtr/audit/state_snapshot.py` deep-copies `instances`,
`id_count_dict`, `id_reid_dict`, module globals (`poss_ids`, `old_ids`,
`old_reids`), and Python/NumPy/Torch RNG streams. The v2 engine stores its
`history_id_overrides` and quarantine set in the snapshot `extra` payload, so
branch restoration also restores the audit-only internal-history view.

## E. V2 assertions

After each full repair v2 records assertions for:

* repaired `p_wrong`/`p_correct` counts in the internal W=20 history;
* `id_count_dict` versus `len(id_reid_dict[id])` for both IDs;
* duplicate IDs within each frame/view;
* collisions where a repaired frame/view already contains `p_correct`;
* relevant `old_reids` IDs and feature/count consistency.

Collisions are labeled `repair_conflict`; unrebuildable memory-bank state is
labeled `C1c-bankoff`. Neither condition is silently counted as a successful
repair.
