"""CPU invariants for frozen-evidence mutable GMT branch isolation."""

from jev_mutable_branch import FrozenEvidenceGMTBranchRunner, TraceTrackerState


def main():
    events = [
        {
            "question": "MATCH_DECISION",
            "off_action": "ACCEPT_CURRENT",
            "legal_actions": ["ACCEPT_CURRENT", "REASSOCIATE", "START_NEW"],
            "context": {
                "video_id": 1,
                "event_order": 0,
                "frame": 1,
                "view": 0,
                "decision_scope": "match",
                "proposal_track_id": 10,
                "alternate_track_id": 20,
            },
        },
        {
            "question": "MEMORY_DECISION",
            "off_action": "WRITE_MEMORY",
            "legal_actions": ["WRITE_MEMORY", "SKIP_MEMORY"],
            "context": {
                "video_id": 1,
                "event_order": 1,
                "frame": 1,
                "view": 0,
                "decision_scope": "memory",
                "track_id": 10,
            },
        },
        {
            "question": "MATCH_DECISION",
            "off_action": "ACCEPT_CURRENT",
            "legal_actions": ["ACCEPT_CURRENT", "START_NEW"],
            "context": {
                "video_id": 1,
                "event_order": 2,
                "frame": 2,
                "view": 0,
                "decision_scope": "match",
                "proposal_track_id": 10,
            },
        },
    ]
    targets = {0: 7, 1: 7, 2: 7}
    runner = FrozenEvidenceGMTBranchRunner(
        events, lambda event: targets[int(event["context"]["event_order"])]
    )
    state = TraceTrackerState.from_event(events[0])
    source_before = state.snapshot()
    outcomes = runner.run(
        state,
        0,
        events[0],
        events[0]["legal_actions"],
        horizon=2,
    )
    assert state.snapshot() == source_before
    assert set(outcomes) == set(events[0]["legal_actions"])
    assert all("rollout_digest" in outcome for outcome in outcomes.values())
    assert outcomes["ACCEPT_CURRENT"]["future_identity_consistency"] >= 0.0
    runner.apply(state, events[0], "ACCEPT_CURRENT", target=7, position=0)
    runner.apply(state, events[1], "WRITE_MEMORY", target=7, position=1)
    assert state.memory_lengths[10] == 1

    # Immediate correctness must be evaluated from the pre-action state.  If
    # an existing ID is already mapped to a different target, applying ACCEPT
    # must not overwrite that mapping before scoring the current event.
    wrong_events = [
        {
            "question": "MATCH_DECISION",
            "legal_actions": ["ACCEPT_CURRENT", "START_NEW"],
            "context": {
                "video_id": 2,
                "event_order": 0,
                "frame": 1,
                "view": 0,
                "decision_scope": "match",
                "proposal_track_id": 10,
            },
        }
    ]
    wrong_runner = FrozenEvidenceGMTBranchRunner(wrong_events, lambda _: 99)
    wrong_state = TraceTrackerState(track_targets={10: 7}, target_tracks={7: 10})
    wrong = wrong_runner.run(
        wrong_state, 0, wrong_events[0], wrong_events[0]["legal_actions"], horizon=1
    )
    assert wrong["ACCEPT_CURRENT"]["immediate_identity"] < 0.0
    assert wrong["ACCEPT_CURRENT"]["sample_weight"] == 1.0

    # No memory target and no affected future event is uninformative; it must
    # be a zero-weight tie rather than a free SKIP_MEMORY label.
    memory_events = [
        {
            "question": "MEMORY_DECISION",
            "legal_actions": ["WRITE_MEMORY", "SKIP_MEMORY"],
            "context": {
                "video_id": 3,
                "event_order": 0,
                "frame": 1,
                "view": 0,
                "decision_scope": "memory",
                "track_id": 10,
            },
        }
    ]
    memory_runner = FrozenEvidenceGMTBranchRunner(memory_events, lambda _: None)
    memory = memory_runner.run(
        TraceTrackerState(), 0, memory_events[0], memory_events[0]["legal_actions"], horizon=1
    )
    assert memory["WRITE_MEMORY"]["sample_weight"] == 0.0
    assert memory["SKIP_MEMORY"]["sample_weight"] == 0.0
    print("JEV mutable branch invariants: PASS")


if __name__ == "__main__":
    main()
