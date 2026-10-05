# GMT Stage2 result

## Checkpoint validation

- Path: `/data1/liuyeqiang/WWW/outputs/stage2_single_gpu/model_20000.pth`
- SHA256: `cd72823824d16c86ed27c2dfc8323de610aa9f6c0c0b29249aa3de609deabce8`
- Iteration: `20000`
- Scheduler/global iteration: `20000 / 20000`
- Model keys: `411`
- Model finiteness: `PASS`
- Optimizer state: `PRESENT`
- Checkpoint reload: `PASS`

## Training metrics

The final logged training values are `total_loss=1.2422004495747387`,
`loss_asso=0.6270861029624939`, and `reid_loss=0.46364010870456696`. The
last-100 total-loss mean is `0.8710850674601442`.

These are GMT training losses, not MOT/COCO tracking metrics. The canonical OFF
tracking baseline is reported separately in
`docs/STAGE2_OFF_TRACKING_RESULTS.md`.

## Reproducibility boundary

All formal JEV artifacts must bind to this checkpoint digest. No `model_4500`
proxy result may be promoted to final evidence or used as official-test
authority.
