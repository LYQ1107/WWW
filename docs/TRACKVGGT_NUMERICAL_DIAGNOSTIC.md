# Bounded frozen-backbone numerical check before training

Real TRAIN5 two-view interface probes retain the original checkpoint and source.
VGGT-Det's attention module and the chunked incoming-attention statistic agree
within about1e-7 absolute error. However, full official-backbone repeated FP16
forwards differ even with the collection hook removed: v4 maximum layer4/11/17/23
differences are .015824/.022341/.169898/.362587. Thus a failed bitwise comparison
cannot currently be attributed specifically to the hook. Training has not begun.
Original failing probes and numerical_diagnostic.json remain retained.

Before changing any backend, the bounded proposed check is: force Torch's math
SDPA and deterministic cuDNN for BOTH instrumented and plain reference forwards,
with the same checkpoint, real image group and FP16 precision. Compare every
retained layer, attention and one further uninstrumented repeat. The problem is
repeatability of the frozen feature extractor; this changes no AG/QD structure,
weights, training labels, selected events or GMT tracker. Default-backend failures
are not silently relabeled. If the check passes, record this backend choice for
every new VGGT teacher/ROI control, and charge its actual extraction cost. No
old full-video result, deployment speed or numerical backend cause is inferred.
