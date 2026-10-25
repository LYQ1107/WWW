"""Posthoc birth, NULL and cumulative identity-contamination diagnostics.

This sidecar receives already delivered IDs. It does not run a tracker or alter
its history. All labels must come from the caller's clean posthoc GT mapping.
"""
from collections import Counter, defaultdict


class TemporalErrorAudit:
    def __init__(self):
        self.seen_ids = set()
        self.cumulative_labels = defaultdict(Counter)
        self.last_seen_step = {}
        self.step_index = 0

    def observe(self, observations, labels, assignments, prior_active,
                prior_clean_labels, current_objects):
        counts = Counter()
        events = []
        current = defaultdict(list)
        unique_history = defaultdict(list)
        for gid in prior_active:
            label = prior_clean_labels.get(gid)
            if label is not None:
                unique_history[label].append(gid)
        for index, label in enumerate(labels):
            if label is not None:
                current[label].append(index)
        counts["clean_detected_person_observations"] = sum(label is not None for label in labels)
        # The caller first asserts complete key/bbox equality across all methods.
        counts["clean_detected_person_observations_retained_in_output"] = counts["clean_detected_person_observations"]
        counts["clean_detected_person_observations_dropped_after_association"] = 0
        for index, (observation, label) in enumerate(zip(observations, labels)):
            if label is None:
                continue
            gid = assignments[observation.key]
            old_labels = self.cumulative_labels.get(gid, {})
            born = gid not in self.seen_ids
            counts["clean_observations_assigned_to_new_global_id" if born else
                   "clean_observations_assigned_to_existing_global_id"] += 1
            reason = None
            if born and len(unique_history[label]) == 1:
                counts["new_id_despite_unique_clean_active_history"] += 1
                reason = "BIRTH_DESPITE_UNIQUE_CLEAN_HISTORY"
            elif old_labels and label not in old_labels:
                counts["historical_id_reused_for_previously_different_clean_identity"] += 1
                reason = "HISTORICAL_ID_REUSED_FOR_DIFFERENT_PERSON"
            if old_labels and len(old_labels) > 1:
                counts["clean_observations_reusing_cumulatively_mixed_id"] += 1
            if reason:
                events.append({"error": reason, "observation_indices": [index],
                               "prior_global_id_labels": sorted(old_labels),
                               "matching_prior_global_ids": unique_history[label]})
        if current_objects is None:
            counts["timestamps_without_serialized_NULL_state"] += 1
        else:
            counts["timestamps_with_serialized_NULL_state"] += 1
            objects = {obj["global_id"]: obj for obj in current_objects}
            for label, indices in current.items():
                # Duplicate clean histories do not give a unique NULL target.
                if len(unique_history[label]) != 1:
                    continue
                gid = unique_history[label][0]
                counts["unique_clean_history_objects_with_current_detection"] += 1
                obj = objects.get(gid)
                if obj is not None and obj["missed"] > 0:
                    counts["unique_clean_history_NULL_despite_current_detection"] += 1
                    events.append({"error": "NULL_DESPITE_UNIQUE_CLEAN_CURRENT_PERSON",
                                   "observation_indices": indices,
                                   "null_global_id": gid, "missed_after_decision": obj["missed"]})
        grouped = defaultdict(Counter)
        for observation, label in zip(observations, labels):
            if label is not None:
                grouped[assignments[observation.key]][label] += 1
        for gid, values in grouped.items():
            history = self.cumulative_labels[gid]
            was_mixed = len(history) > 1
            history.update(values)
            if len(history) > 1 and not was_mixed:
                counts["cumulative_identity_contamination_onsets"] += 1
            if len(history) > 1:
                counts["clean_observations_assigned_to_cumulatively_mixed_id"] += sum(values.values())
        self.seen_ids.update(assignments.values())
        for gid in assignments.values():
            self.last_seen_step[gid] = self.step_index
        active_now = {gid for gid, step in self.last_seen_step.items()
                      if self.step_index - step <= 12} if current_objects is None else {
            obj["global_id"] for obj in current_objects}
        counts["active_cumulatively_labelled_object_steps"] += sum(
            bool(self.cumulative_labels.get(gid)) for gid in active_now)
        counts["active_cumulatively_mixed_object_steps"] += sum(
            len(self.cumulative_labels.get(gid, {})) > 1 for gid in active_now)
        self.step_index += 1
        return counts, events


def increment_quarter(quarters, before, after, frame_index, total_frames):
    """Accumulate monotone frame counts by relative video progress, not GT time."""
    quarter = min(3, 4 * frame_index // max(1, total_frames))
    for key, value in after.items():
        delta = value - before.get(key, 0)
        if delta < 0:
            raise ValueError("NONMONOTONE_DIAGNOSTIC_COUNT")
        quarters[quarter][key] += delta
