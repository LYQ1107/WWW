WorldMOT V5 identity analysis reporting

This reporting layer changes no model, checkpoint, inference rule, training budget or GT. It consumes the completed all110 identity-attribution artifact and original legal14 native scores. Until those artifacts exist, the analysis JSON is explicitly pending, the CSVs contain headers only, and the narrative contains no measured event rates.

The renderer requires all five canonical native scopes, matching prediction hashes for every one of22 scenes,29,019 paired times and548,583 frozen candidates per method, and the unchanged original annotation hash. Each new model must have an actual full24 final gate with the registered initialization,606,468 parameters and common12/16 epochs. Every final epoch must cover all24 scenes and28,754 paired times. Quarter counters must sum to full-scene counters; scene counters must sum to their legal14 or invalid8 groups. Reliable and unreliable shared-input geometry strata must partition the same legal14 cross-view opportunities.

The report keeps these units distinct:

| Quantity | Denominator or interpretation |
|---|---|
| Current output coverage | All frozen detector candidates, including audited duplicate suppression |
| Same-person shared-ID rate | All clean same-person cross-view pairs |
| Same-person ID split | Clean same-person cross-view pairs with two delivered different IDs; missing output is reported separately |
| Different-person false merge | All clean different-person cross-view pairs |
| Simultaneous Duplicate Birth | Clean same-person cross-view pairs assigned different previously unseen IDs, over all common clean same-person pair opportunities; one dropped output cannot become a successful paired avoidance event |
| Extra simultaneous newborn IDs | Additional distinct new IDs for the same clean person at the same timestamp, per1,000 retained clean observations; this differs from a historical ID plus one new ID |
| False Birth proxy | New-ID observations despite a unique pure active history; displayed per1,000 retained clean observations |
| Wrong historical reuse | Clean observations reusing an ID with different prior clean identity history, per1,000 retained clean observations |
| NULL | Unique pure active histories remaining missed despite a current clean person, over serialized unique-history opportunities |
| LOST recovery correctness | Pure-identity correct recoveries over observations that actually reused a prior LOST ID; this is conditional correctness, not recovery recall |
| Detection-gap retention | Same ID after a2–30 or longer observation gap; GT absence may be occlusion, out-of-view or missing annotation |
| Cumulative mixed-ID output | Clean retained observations assigned to IDs cumulatively associated with multiple clean GT people; this is not a direct embedding measurement |
| Write coincidence | Wrong-reused IDs written at the current timestamp; the timestamp does not identify the camera/observation that caused the write |

A zero opportunity count yields N/A, never an invented0% or100%. Every rate retains its numerator, denominator and unit in CSV. Paired contrasts retain opportunities, corrections, regressions and signed net changes. A positive diagnostic net cannot override negative native HOTA, AssA or CVIDF1. Video-quarter rates aggregate exposures before division, rather than averaging scene percentages.

The generated-input checks cover pending output, zero-opportunity recovery, weighted exposures, canonical-scope rejection, changed prediction hashes, full24/common-budget rejection, geometry/quarter conservation, and narrative/CSV preservation. Additional delivered-output checks verify symmetric Duplicate Birth corrections/regressions, reject suppressed-output pseudo-corrections, and distinguish a historical ID plus one new ID from simultaneous multiple-ID birth. They read no real TEST data and run no model.

The current reporting outputs are:

- worldmot/results/v5_identity_analysis.json
- worldmot/results/v5_identity_event_rates.csv
- worldmot/results/v5_identity_event_contrasts.csv
- worldmot/results/v5_identity_video_quarters.csv
- worldmot/docs/V5_FULL_TEST_IDENTITY_ANALYSIS.md

The standard summary entrypoint generates these files. Invalid8 event diagnostics remain separately labelled and never become official native metrics. B1−B0 is a coupled policy replacement; B2−B1 also changes capacity/training budget; B3−B4 is the matched explicit-geometry comparison. Single-seed outcomes do not establish statistical significance.
