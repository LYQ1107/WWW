# WorldMOT World Object G3 — temporal persistence

This is the temporal-persistence companion name for
`WORLDOBJECT_G3_WORLD_OBJECT_PERSISTENCE.md`. The registered TEST result is
the causal persistent-bank route, not the oracle diagnostic.

Across TEST (22 scenes, 29,019 synchronized timestamps), the predicted bank
has 332,781 identity observations, 85,155 predicted identity switches and
31,958 wrong world-object merges. The oracle-history diagnostic has zero
switches by construction and is not used for formal TEST inference.

The complete failure audit is `results/worldobject_test/G3_temporal_identity_failures.json`;
the compact route summary is `results/worldobject_test/G3_oracle_vs_predicted_all24.csv`
(the filename is retained for compatibility and its `split` field is TEST).

