# Long-horizon analysis

Short-window accuracy alone can reward a current-frame heuristic. The v2
labeler therefore preserves raw outcomes over multiple horizons (for example
1, 4, 8, 16, and 32 frames, subject to the dataset cadence):

- correct identity duration;
- ID switches and fragmentation;
- collisions;
- memory contamination and duration;
- recovery latency;
- false reactivation;
- new-ID fragmentation;
- window IDF1 and an AssA proxy.

The report must show horizon curves for GMT/OFF, each threshold baseline,
generic MLP, and JEV, plus match-only, memory-only, reactivation-only, and
full-policy ablations. A result that improves only immediate acceptance but
not a persistent-state metric is not evidence for the claimed state-transition
mechanism.

