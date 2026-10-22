# World alignment

The first valid synchronized reconstruction defines a local world reference. For later frames, ORB matches on the first shared camera are sampled in the two dense pointmaps; confidence and finite-point checks reject unusable matches. A deterministic RANSAC Umeyama fit estimates current-to-previous-world `Sim(3)` and records correspondence count, inliers, residuals, and scale.

If the fit does not pass the fixed inlier gate, current observations are marked unaligned and are not written into identity memory. Existing tracks are only predicted and their uncertainty grows. The last valid world pointmaps remain the alignment reference so a later frame can recover without pretending that a failed coordinate system is globally valid.
