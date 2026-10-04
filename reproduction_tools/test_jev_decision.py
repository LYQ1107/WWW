"""CPU invariants for the shared typed JEV decision controller."""

import torch

from gtr.modeling.jev_decision import (
    ACTION_NAMES,
    JEVDecisionController,
    QUESTION_NAMES,
    legal_actions_for,
)


def main():
    assert legal_actions_for("MATCH_DECISION", can_reassociate=True) == [
        "ACCEPT_CURRENT",
        "REASSOCIATE",
        "START_NEW",
    ]
    assert legal_actions_for("MATCH_DECISION", can_reassociate=False) == [
        "ACCEPT_CURRENT",
        "START_NEW",
    ]
    assert legal_actions_for("REACTIVATION_DECISION", has_old_track=False) == [
        "START_NEW",
    ]

    torch.manual_seed(7)
    model = JEVDecisionController(
        state_dim=10,
        hidden_dim=32,
        question_dim=16,
        action_dim=16,
        use_option_interaction=True,
    ).eval()
    features = torch.randn(3, 10)
    questions = ["MATCH_DECISION", "MEMORY_DECISION", "REACTIVATION_DECISION"]
    actions = [
        legal_actions_for(questions[0]),
        legal_actions_for(questions[1]),
        legal_actions_for(questions[2], has_old_track=False),
    ]
    result = model(features, questions, actions)
    assert result["probs"].shape == (3, 3)
    assert torch.allclose(result["probs"].sum(dim=1), torch.ones(3), atol=1e-6)
    assert torch.all(result["probs"] >= 0)

    # Runtime masks must remove REACTIVATE_OLD rather than leave a dead class.
    assert result["legal_mask"][2].tolist() == [True, False, False]
    assert result["legal_actions"][2, 0].item() == ACTION_NAMES.index("START_NEW")

    # Permuting a legal action set must permute its probabilities, not change
    # the semantic score assigned to each action.
    one = model(features[:1], [questions[0]], [actions[0]])["probs"][0]
    permuted_actions = [["START_NEW", "ACCEPT_CURRENT", "REASSOCIATE"]]
    permuted = model(features[:1], [questions[0]], permuted_actions)["probs"][0]
    mapping = {name: one[index].item() for index, name in enumerate(actions[0])}
    for index, name in enumerate(permuted_actions[0]):
        assert abs(permuted[index].item() - mapping[name]) < 1e-5

    # The scorer is trainable and is shared across all three question types.
    train_result = model.train()(features, questions, actions)
    loss = -torch.log(train_result["probs"][:, 0]).mean()
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
    print("JEV decision invariants: PASS")


if __name__ == "__main__":
    main()
