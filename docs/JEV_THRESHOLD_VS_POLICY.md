# Threshold versus policy baselines

The minimum baseline family is:

1. fixed legacy threshold;
2. global learned scalar threshold;
3. linear state-conditioned threshold;
4. nonlinear state-conditioned threshold;
5. question-conditioned threshold;
6. question-conditioned generic MLP;
7. parameter-matched shared encoder with separate heads;
8. action-conditioned scorer without question conditioning;
9. question-conditioned fixed head;
10. full typed JEV shared scorer.

The threshold baselines may decide whether to accept or reject a GMT proposal,
but they do not receive a candidate-ID output head. All models use identical
legal-action masking, records, sample weights, sequence split, optimizer
budget, calibration procedure, and evaluation metrics.

The central diagnostic is whether a full JEV advantage remains after the
nonlinear/question-conditioned threshold family is capacity-matched. If not,
the typed state-transition claim is NO-GO even if a raw JEV number improves.

