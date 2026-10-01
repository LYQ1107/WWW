# Causal Audit v2 mechanism evidence matrix

The old `STATE_CAUSAL_GO = NO-GO` is immutable. This is a post-hoc mechanism gate.

| Test | Primary endpoint | Result | Gate implication |
|---|---|---|---|
| C2c isolated injection | target GT error | support horizons `[]` | FAIL |
| C1c full internal repair | target GT error | support horizons `[2, 5, 10, 20]` | PASS |

```json
{
  "mechanism_validated": "NO",
  "C2c_propagation_support_horizons": [],
  "C1c_recovery_support_horizons": [
    2,
    5,
    10,
    20
  ],
  "C2c_event_count": 256,
  "C1c_event_count": 71,
  "old_STATE_CAUSAL_GO_preserved": "NO-GO",
  "rule": "C2c target-error lower CI >0 and >=2/3 scenes positive at one of +2/+5/+10/+20; C1c target-error upper CI <0 and >=2/3 scenes negative at one of +2/+5/+10/+20",
  "seed": 20260930,
  "bootstrap_resamples": 10000
}
```
