"""CPU invariants for the JEV baseline family."""

import torch

from gtr.modeling.jev_baselines import (
    FixedThresholdPolicy,
    FixedSlotMLP,
    GlobalLearnedThreshold,
    IndependentMLPHeads,
    LogisticGate,
    SharedEncoderSeparateHeads,
    StateConditionedThreshold,
)
from gtr.modeling.jev_decision import legal_actions_for


def main():
    features = torch.tensor(
        [
            [0.9, 0.1, 0.2, 0.3, 0.0],
            [0.1, 0.8, 0.2, 0.3, 0.0],
            [0.1, 0.2, 0.8, 0.3, 0.0],
            [0.1, 0.2, 0.3, 0.8, 0.0],
        ]
    )
    questions = [
        "MATCH_DECISION",
        "MATCH_DECISION",
        "MEMORY_DECISION",
        "REACTIVATION_DECISION",
    ]
    actions = [
        legal_actions_for(questions[0]),
        legal_actions_for(questions[1]),
        legal_actions_for(questions[2]),
        legal_actions_for(questions[3]),
    ]

    fixed = FixedThresholdPolicy(0.5).eval()
    fixed_result = fixed(features, questions, actions)
    fixed_choices = fixed_result["legal_actions"].gather(
        1, fixed_result["probs"].argmax(dim=1, keepdim=True)
    ).squeeze(1).tolist()
    assert fixed_choices == [0, 1, 3, 5], fixed_choices
    assert torch.allclose(
        fixed_result["probs"].sum(dim=1), torch.ones(len(features)), atol=1e-6
    )

    # Every learned baseline has the same masked interface and is trainable.
    models = [
        GlobalLearnedThreshold(),
        StateConditionedThreshold(features.shape[1]),
        LogisticGate(features.shape[1]),
        FixedSlotMLP(features.shape[1], hidden_dim=16),
        IndependentMLPHeads(features.shape[1], hidden_dim=16),
        SharedEncoderSeparateHeads(features.shape[1], hidden_dim=16),
    ]
    for model in models:
        result = model(features, questions, actions)
        assert result["probs"].shape == (4, 3)
        assert torch.allclose(result["probs"].sum(dim=1), torch.ones(4), atol=1e-6)
        loss = -torch.log(result["probs"][:, 0]).mean()
        loss.backward()
        assert any(parameter.grad is not None for parameter in model.parameters())

    # Fixed-slot policies must be semantic, not position based.
    permuted = [actions[0][::-1]] + actions[1:]
    original = models[2](features, questions, actions)["probs"][0]
    changed = models[2](features, questions, permuted)["probs"][0]
    original_by_name = {
        name: original[index].item() for index, name in enumerate(actions[0])
    }
    for index, name in enumerate(permuted[0]):
        assert abs(changed[index].item() - original_by_name[name]) < 1e-6

    print("JEV baseline invariants: PASS")


if __name__ == "__main__":
    main()
