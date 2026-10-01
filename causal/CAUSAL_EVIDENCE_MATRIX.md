# Causal evidence matrix

The primary endpoint is future identity error. Current event frames are excluded. Bootstrap intervals resample frozen events with seed 20260930.

## C1 Oracle correction (expected negative)

|   horizon |   events |   mean_difference |    ci95_low |   ci95_high |   scene_same_direction_fraction | direction_expected   | paired_ci_excludes_zero_expected_direction   |
|----------:|---------:|------------------:|------------:|------------:|--------------------------------:|:---------------------|:---------------------------------------------|
|         1 |       60 |         0.0833333 |  0.00833333 |    0.166667 |                        0.333333 | negative             | False                                        |
|         2 |       55 |         0.0727273 | -0.00909091 |    0.154545 |                        0.333333 | negative             | False                                        |
|         5 |       57 |         0.0614035 | -0.0263158  |    0.149123 |                        0.333333 | negative             | False                                        |
|        10 |       54 |         0.111111  |  0.00925926 |    0.222222 |                        0.333333 | negative             | False                                        |
|        20 |       56 |         0.116071  |  0.0267857  |    0.214286 |                        0.333333 | negative             | False                                        |

## C2 Error injection (expected positive)

|   horizon |   events |   mean_difference |   ci95_low |   ci95_high |   scene_same_direction_fraction | direction_expected   | paired_ci_excludes_zero_expected_direction   |
|----------:|---------:|------------------:|-----------:|------------:|--------------------------------:|:---------------------|:---------------------------------------------|
|         1 |      503 |          0.384692 |   0.343936 |    0.426466 |                               1 | positive             | True                                         |
|         2 |      504 |          0.376984 |   0.335317 |    0.419643 |                               1 | positive             | True                                         |
|         5 |      510 |          0.376471 |   0.334314 |    0.418627 |                               1 | positive             | True                                         |
|        10 |      490 |          0.362245 |   0.319388 |    0.405102 |                               1 | positive             | True                                         |
|        20 |      472 |          0.355932 |   0.3125   |    0.399364 |                               1 | positive             | True                                         |

## Gate

**CONDITIONAL GO**

Only one causal intervention meets the predeclared evidence direction. Stop before implementing JEV-GMT and report the ambiguity.

The event schedules were frozen from baseline decisions before either intervention outcome was read.
