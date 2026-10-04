"""CPU invariants for decision-policy metrics."""

from jev_metrics import summarize


def main():
    records = [
        {
            "legal_actions": ["ACCEPT_CURRENT", "START_NEW"],
            "target_probs": [1.0, 0.0],
            "best_actions": ["ACCEPT_CURRENT"],
        },
        {
            "legal_actions": ["ACCEPT_CURRENT", "START_NEW"],
            "target_probs": [0.0, 1.0],
            "best_actions": ["START_NEW"],
        },
    ]
    predictions = [
        {"legal_actions": ["START_NEW", "ACCEPT_CURRENT"], "probabilities": [0.1, 0.9]},
        {"legal_actions": ["ACCEPT_CURRENT", "START_NEW"], "probabilities": [0.2, 0.8]},
    ]
    result = summarize(records, predictions)
    assert result["records"] == 2
    assert result["best_action_accuracy"] == 1.0
    assert result["nll"] > 0.0
    assert len(result["risk_coverage"]) == 4
    print("JEV metric invariants: PASS")


if __name__ == "__main__":
    main()

