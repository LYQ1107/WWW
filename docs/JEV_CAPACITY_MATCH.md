# Capacity matching protocol

Every learned baseline and JEV variant uses the same state features, question
and legal-action masks, counterfactual labels, split, and training budget.
`reproduction_tools/match_policy_capacity.py` records trainable parameters,
estimated linear MACs, batch-one CPU latency, and CUDA peak memory when
available.

The matching rule searches hidden widths and minimizes absolute trainable
parameter difference to the selected JEV target. The formal gate is relative
parameter difference below 2% (target below 1% where a width permits it).
The report must also include MAC/FLOP difference, decision latency, peak VRAM,
and any extra association calls caused by REASSOCIATE.

The current CPU smoke probe used `state_dim=64` and `hidden_dim=64` and is only
a tooling check. It is not the final capacity report and does not justify a
performance comparison.

