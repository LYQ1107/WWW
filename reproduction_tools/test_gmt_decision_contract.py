"""CPU regression checks for the original GMT decision semantics."""

from gtr.modeling.gmt_decision_contract import (
    gmt_off_match_action,
    gmt_threshold,
    match_legal_actions,
    reassociation_candidate,
)


def main():
    assert gmt_threshold(0.2, 5, not_mult_thresh=True) == 0.2
    assert gmt_threshold(0.2, 5, not_mult_thresh=False) == 1.0
    assert gmt_off_match_action(0.2, 0.2) == "START_NEW"
    assert gmt_off_match_action(0.20001, 0.2) == "ACCEPT_CURRENT"
    assert match_legal_actions(False) == ["ACCEPT_CURRENT", "START_NEW"]
    assert reassociation_candidate([0.9, 0.8, 0.7], 0, [1]) == 2
    assert reassociation_candidate([0.9], 0) is None
    print("GMT decision contract invariants: PASS")


if __name__ == "__main__":
    main()

