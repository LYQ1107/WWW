# C1 Oracle correction

Events in frozen manifest: 71
Events applied in intervention run: 70
Events observed in sham run: 71

The intervention is compared with its same-event sham. The current event frame is excluded; future identity error is the fraction of matched observations for the same scene/GT at exactly t+k whose selected ID differs from the frozen target ID.

## Paired horizon summary

|   horizon |   events |   sham_error |   intervention_error |   difference_intervention_minus_sham | bootstrap_ci95                              |   same_direction_fraction |   cross_view_difference |
|----------:|---------:|-------------:|---------------------:|-------------------------------------:|:--------------------------------------------|--------------------------:|------------------------:|
|         1 |       60 |     0.558333 |             0.641667 |                            0.0833333 | [0.008333333333333333, 0.16666666666666666] |                  0.166667 |                0.368421 |
|         2 |       55 |     0.618182 |             0.690909 |                            0.0727273 | [-0.00909090909090909, 0.15454545454545454] |                  0.163636 |                0.411765 |
|         5 |       57 |     0.666667 |             0.72807  |                            0.0614035 | [-0.02631578947368421, 0.14912280701754385] |                  0.175439 |                0.411765 |
|        10 |       54 |     0.472222 |             0.583333 |                            0.111111  | [0.009259259259259259, 0.2222222222222222]  |                  0.222222 |                0.411765 |
|        20 |       56 |     0.491071 |             0.607143 |                            0.116071  | [0.026785714285714284, 0.21428571428571427] |                  0.178571 |                0.315789 |

Bootstrap intervals resample events with seed 20260930. No result is used to select events.

## Scene-level paired effects

| scene       |   horizon |   difference |
|:------------|----------:|-------------:|
| 00001garden |         1 |   -0.111111  |
| 00001garden |         2 |   -0.125     |
| 00001garden |         5 |   -0.125     |
| 00001garden |        10 |   -0.0625    |
| 00001garden |        20 |   -0.0625    |
| 00003garden |         1 |    0.269231  |
| 00003garden |         2 |    0.318182  |
| 00003garden |         5 |    0.214286  |
| 00003garden |        10 |    0.357143  |
| 00003garden |        20 |    0.384615  |
| 00005garden |         1 |    0.0657895 |
| 00005garden |         2 |    0.0416667 |
| 00005garden |         5 |    0.0428571 |
| 00005garden |        10 |    0.046875  |
| 00005garden |        20 |    0.0571429 |


## Tracking metrics

| condition    |    HOTA |    AssA |    MOTA |    IDF1 |   number_predictions |   sequence_count |
|:-------------|--------:|--------:|--------:|--------:|---------------------:|-----------------:|
| intervention | 83.1823 | 82.3426 | 79.2982 | 82.8022 |                47949 |                6 |
| sham         | 83.0679 | 81.9243 | 79.6219 | 82.6338 |                48005 |                6 |

## Future error streak

| condition   |   count |    mean |   median |   max |
|:------------|--------:|--------:|---------:|------:|
| effect      |      68 | 2.70588 |      3   |     5 |
| sham        |      68 | 2.70588 |      2.5 |     5 |
