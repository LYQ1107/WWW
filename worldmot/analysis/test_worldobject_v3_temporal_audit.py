from collections import Counter
from types import SimpleNamespace
import pytest

from worldobject_v3_temporal_audit import TemporalErrorAudit, increment_quarter


def frame(audit, labels, gids, active=(), clean=None, objects=None):
    observations = [SimpleNamespace(key=(i % 2 + 1, i)) for i in range(len(labels))]
    assignments = {o.key: gid for o, gid in zip(observations, gids)}
    return audit.observe(observations, labels, assignments, set(active), clean or {}, objects)


def test_false_birth_and_null_are_outcomes_with_clean_unique_history():
    audit = TemporalErrorAudit()
    frame(audit, ["A", "B"], [1, 2], objects=[{"global_id": 1, "missed": 0}, {"global_id": 2, "missed": 0}])
    counts, events = frame(audit, ["A"], [3], [1, 2], {1: "A", 2: "B"},
                          [{"global_id": 1, "missed": 1}, {"global_id": 2, "missed": 1}, {"global_id": 3, "missed": 0}])
    assert counts["new_id_despite_unique_clean_active_history"] == 1
    assert counts["unique_clean_history_NULL_despite_current_detection"] == 1
    assert {event["error"] for event in events} == {
        "BIRTH_DESPITE_UNIQUE_CLEAN_HISTORY", "NULL_DESPITE_UNIQUE_CLEAN_CURRENT_PERSON"}
    assert counts["clean_detected_person_observations_dropped_after_association"] == 0


def test_duplicate_clean_histories_do_not_create_a_unique_null_or_birth_error():
    audit = TemporalErrorAudit()
    frame(audit, ["A", "A"], [1, 2])
    counts, events = frame(audit, ["A"], [3], [1, 2], {1: "A", 2: "A"},
                          [{"global_id": 1, "missed": 1}, {"global_id": 2, "missed": 1}, {"global_id": 3, "missed": 0}])
    assert not counts["new_id_despite_unique_clean_active_history"]
    assert not counts["unique_clean_history_objects_with_current_detection"]
    assert not events


def test_wrong_history_recovery_and_contamination_are_not_erased_by_a_clean_later_frame():
    audit = TemporalErrorAudit()
    frame(audit, ["B"], [2])
    counts, events = frame(audit, ["A"], [2], [2], {2: "B"}, [{"global_id": 2, "missed": 0}])
    assert counts["historical_id_reused_for_previously_different_clean_identity"] == 1
    assert counts["cumulative_identity_contamination_onsets"] == 1
    assert events[0]["prior_global_id_labels"] == ["B"]
    later, _ = frame(audit, ["B"], [2], [2], {2: "B"}, [{"global_id": 2, "missed": 0}])
    assert later["clean_observations_reusing_cumulatively_mixed_id"] == 1
    assert later["active_cumulatively_mixed_object_steps"] == 1
    assert not later["cumulative_identity_contamination_onsets"]


def test_first_frame_different_people_sharing_one_id_is_a_single_contamination_onset():
    audit = TemporalErrorAudit()
    counts, _ = frame(audit, ["A", "B"], [1, 1])
    assert counts["cumulative_identity_contamination_onsets"] == 1
    assert counts["clean_observations_assigned_to_cumulatively_mixed_id"] == 2


def test_previously_unmatched_output_id_is_not_relabelled_as_new_birth():
    audit = TemporalErrorAudit()
    frame(audit, [None], [7])
    counts, _ = frame(audit, ["A"], [7], [3], {3: "A"})
    assert counts["clean_observations_assigned_to_existing_global_id"] == 1
    assert not counts["new_id_despite_unique_clean_active_history"]


def test_deleted_object_is_not_counted_as_a_serialized_null():
    audit = TemporalErrorAudit()
    frame(audit, ["A"], [1])
    counts, _ = frame(audit, ["A"], [2], [1], {1: "A"}, [{"global_id": 2, "missed": 0}])
    assert counts["unique_clean_history_objects_with_current_detection"] == 1
    assert not counts["unique_clean_history_NULL_despite_current_detection"]


def test_legacy_object_retention_is_twelve_processing_steps_and_null_is_unobserved():
    audit = TemporalErrorAudit()
    first, _ = frame(audit, ["A"], [1])
    assert first["timestamps_without_serialized_NULL_state"] == 1
    for _ in range(12):
        counts, _ = frame(audit, [], [])
        assert counts["active_cumulatively_labelled_object_steps"] == 1
    expired, _ = frame(audit, [], [])
    assert expired["active_cumulatively_labelled_object_steps"] == 0
    assert not expired["timestamps_with_serialized_NULL_state"]


def test_quarter_counts_partition_the_complete_video_with_unequal_quarter_sizes():
    quarters = [Counter() for _ in range(4)]
    cumulative = Counter()
    for index in range(9):
        before = cumulative.copy()
        cumulative.update({"paired_timestamps": 1, "errors": int(index % 2 == 0)})
        increment_quarter(quarters, before, cumulative, index, 9)
    assert [quarter["paired_timestamps"] for quarter in quarters] == [3, 2, 2, 2]
    assert sum(quarters, Counter()) == cumulative
    with pytest.raises(ValueError, match="NONMONOTONE"):
        increment_quarter(quarters, {"errors": 2}, {"errors": 1}, 0, 9)
