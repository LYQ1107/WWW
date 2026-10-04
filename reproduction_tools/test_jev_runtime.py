"""CPU invariants for the online JEV adapter and commit boundary."""

from pathlib import Path
import tempfile

import torch
from torch import nn

from gtr.modeling.jev_runtime import DecisionTraceWriter, JEVCommitAdapter, JEVRuntimePolicy


class AlwaysStart(nn.Module):
    def forward(self, state, questions, legal_actions):
        names = list(legal_actions[0])
        ids = {name: index for index, name in enumerate((
            "ACCEPT_CURRENT", "REASSOCIATE", "START_NEW", "WRITE_MEMORY",
            "SKIP_MEMORY", "REACTIVATE_OLD"))}
        action_ids = torch.tensor([[ids[name] for name in names]], dtype=torch.long)
        logits = torch.tensor([[-5.0 if name != "START_NEW" else 5.0 for name in names]])
        return {"probs": torch.softmax(logits, dim=1), "legal_actions": action_ids,
                "legal_mask": torch.ones_like(action_ids, dtype=torch.bool)}


def main():
    features = torch.zeros(1, 8)
    legal = ["ACCEPT_CURRENT", "REASSOCIATE", "START_NEW"]
    off = JEVRuntimePolicy("off")
    off_decision = off.decide(features, "MATCH_DECISION", legal, off_action="ACCEPT_CURRENT")
    assert off_decision.committed_action == "ACCEPT_CURRENT"
    assert off_decision.proposed_action == "ACCEPT_CURRENT"

    controller = AlwaysStart()
    with tempfile.TemporaryDirectory() as directory:
        trace = DecisionTraceWriter(Path(directory) / "shadow.jsonl")
        shadow = JEVRuntimePolicy("shadow", controller, trace)
        decision = shadow.decide(features, "MATCH_DECISION", legal, off_action="ACCEPT_CURRENT")
        trace.close()
        assert decision.proposed_action == "START_NEW"
        assert decision.committed_action == "ACCEPT_CURRENT"
        line = (Path(directory) / "shadow.jsonl").read_text()
        assert '"future_gt_access": false' in line
        assert "ground_truth" not in line
        assert '"event_order": 0' in line

    jev = JEVRuntimePolicy("jev", controller)
    assert jev.decide(features, "MATCH_DECISION", legal, off_action="ACCEPT_CURRENT").committed_action == "START_NEW"
    try:
        JEVRuntimePolicy("oracle")
    except ValueError as exc:
        assert "unsupported online" in str(exc)
    else:
        raise AssertionError("online adapter accepted oracle mode")

    state = {"commits": []}
    adapter = JEVCommitAdapter()
    committed = adapter.commit(state, off_decision, lambda current, action: {"commits": current["commits"] + [action]})
    assert state == {"commits": []}
    assert committed == {"commits": ["ACCEPT_CURRENT"]}
    print("JEV runtime invariants: PASS")


if __name__ == "__main__":
    main()
