# G2 full TRAIN multiview formation summary

The joint world-object builder was evaluated on all 24 TRAIN scenes after the immutable Ω cache was complete. GT was used only post-hoc to classify each fused hypothesis:

- Hypotheses: **409,807**; fused two-camera hypotheses: **147,195**.
- Correct same-person fusions: **113,199**.
- Wrong different-person fusions: **33,865**.
- Remaining fused cases without a unique clean identity: **131**.
- Correct-vs-wrong among uniquely classified fused pairs: **76.97% / 23.03%**.
- Candidate cross-view pairs in the TRAIN threshold sweep: **6,055,068**.

The TRAIN-only compatibility sweep trades same-person recall against different-person false-merge rate (for example, at threshold 0.50 the diagnostic recall is 69.04% and the different-person false-merge rate is 3.51%). These thresholds are not TEST-selected and are not a final MOT result. The scene spread is large, with crowd-heavy Scenes 3, 12–16, 20, 23, and 24 driving most wrong fusions.
