# C2 Error injection

Events in frozen manifest: 939
Events applied in intervention run: 535
Events observed in sham run: 939

The intervention is compared with its same-event sham. The current event frame is excluded; future identity error is the fraction of matched observations for the same scene/GT at exactly t+k whose selected ID differs from the frozen target ID.

## Paired horizon summary

|   horizon |   events |   sham_error |   intervention_error |   difference_intervention_minus_sham | bootstrap_ci95                             |   same_direction_fraction |   cross_view_difference |
|----------:|---------:|-------------:|---------------------:|-------------------------------------:|:-------------------------------------------|--------------------------:|------------------------:|
|         1 |      503 |    0.0308151 |             0.415507 |                             0.384692 | [0.34393638170974156, 0.42646620278329983] |                  0.389662 |               0.0819672 |
|         2 |      504 |    0.0327381 |             0.409722 |                             0.376984 | [0.3353174603174603, 0.41964285714285715]  |                  0.382937 |               0.0737705 |
|         5 |      510 |    0.0411765 |             0.417647 |                             0.376471 | [0.33431372549019606, 0.41862745098039217] |                  0.382353 |               0.0901639 |
|        10 |      490 |    0.0408163 |             0.403061 |                             0.362245 | [0.3193877551020408, 0.4051020408163265]   |                  0.367347 |               0.0813008 |
|        20 |      472 |    0.0423729 |             0.398305 |                             0.355932 | [0.3125, 0.399364406779661]                |                  0.364407 |               0.0806452 |

Bootstrap intervals resample events with seed 20260930. No result is used to select events.

## Scene-level paired effects

| scene       |   horizon |   difference |
|:------------|----------:|-------------:|
| 00001garden |         1 |     0.317518 |
| 00001garden |         2 |     0.308824 |
| 00001garden |         5 |     0.316547 |
| 00001garden |        10 |     0.311594 |
| 00001garden |        20 |     0.277778 |
| 00003garden |         1 |     0.328358 |
| 00003garden |         2 |     0.345588 |
| 00003garden |         5 |     0.318519 |
| 00003garden |        10 |     0.296992 |
| 00003garden |        20 |     0.287879 |
| 00005garden |         1 |     0.456897 |
| 00005garden |         2 |     0.435345 |
| 00005garden |         5 |     0.444915 |
| 00005garden |        10 |     0.43379  |
| 00005garden |        20 |     0.443925 |


## Tracking metrics

| condition    |    HOTA |    AssA |    MOTA |    IDF1 |   number_predictions |   sequence_count |
|:-------------|--------:|--------:|--------:|--------:|---------------------:|-----------------:|
| intervention | 80.1667 | 76.7494 | 77.95   | 79.2956 |                47732 |                6 |
| sham         | 83.0679 | 81.9243 | 79.6219 | 82.6338 |                48005 |                6 |

## Future error streak

| condition   |   count |     mean |   median |   max |
|:------------|--------:|---------:|---------:|------:|
| effect      |     528 | 1.98106  |        0 |     5 |
| sham        |     528 | 0.242424 |        0 |     5 |
