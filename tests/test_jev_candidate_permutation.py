import torch

from gtr.jev.decision_head import SetChoiceHead


def test_set_choice_preserves_semantic_probabilities_under_candidate_permutation():
    torch.manual_seed(7)
    head = SetChoiceHead(query_dim=11, option_dim=17)
    head.eval()
    query = torch.randn(8, 11)
    options = torch.randn(8, 5, 17)
    mask = torch.ones(8, 5, dtype=torch.bool)
    with torch.no_grad():
        original = torch.softmax(head(query, options, mask), dim=-1)
        permutation = torch.tensor([2, 0, 4, 1, 3])
        permuted = options[:, permutation]
        permuted_probabilities = torch.softmax(head(query, permuted, mask), dim=-1)
        restored = permuted_probabilities[:, torch.argsort(permutation)]
    assert torch.max(torch.abs(original - restored)).item() < 1e-5
