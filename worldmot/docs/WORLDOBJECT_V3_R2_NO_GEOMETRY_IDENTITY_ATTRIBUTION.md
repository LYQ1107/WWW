# R2 Full No-Geometry: completed TEST identity attribution

These are **read-only outcome diagnostics**, not new official tracking metrics
or a source of R2 parameter selection. All 22 immutable prediction files were
SHA-verified before annotations were opened. Predictions and original GT are
unchanged. The primary counts below cover the 14 legal full TEST videos
(19,103 synchronous times); the other eight videos are reported separately.

The official all22-inference/legal14-evaluation result is HOTA **60.4318**,
AssA **58.2867**, IDF1 **70.1622**, CVIDF1 **67.6049**, MOTA **75.0228**,
TrackEval IDSW **2,130**. The final all24-TRAIN checkpoint SHA-256 is
`b08badfc634c9652a34366951c93207fa321deb7d1a2b240a76798f11607053f`.

The identical legal14 GMT reference has HOTA 66.0646, AssA 69.5787,
IDF1 80.1121, CVIDF1 78.0613, MOTA 72.9636 and IDSW 6,044. Thus the
No-Geometry CVIDF1 deficit is **10.4564 percentage points**, despite fewer
local ID switches. TP/FP/FN differ from GMT by only -3/+3/+3. Detection
recall does not explain the association deficit.

| Read-only diagnostic on legal14 | Actual count |
|---|---:|
| Clean IoU=.5 matched output observations | 165,351 |
| Same-person cross-view pairs | 54,452 |
| Same-person pairs delivered with different IDs | 11,662 / 54,452 (21.42%) |
| Clean detected GT people split over multiple IDs | 123 / 151 (81.46%) |
| Extra person-ID fragments, summed per GT person | 678 |
| Different-person cross-view false merges | 1,515 |
| False merges reusing an existing ID | 1,514 |
| False merges creating a joint new ID | 1 |
| Clean-labelled output IDs cumulatively used for multiple GT people | 177 / 545 |
| Minority-person observations within those mixed IDs | 6,481 |
| New-ID observations despite unique clean matching active history | 165 |
| Unique matching-history object steps left NULL despite current detection | 745 / 87,677 |
| Per-camera detection-gap returns within 30 steps that change ID | 324 / 1,792 |
| Per-camera detection-gap returns over 30 steps that change ID | 138 / 161 |
| Global detection-gap returns recovering none of the previous IDs | 309 / 1,306 |

The 678 extra fragments are per-person graph incidences, not 678 unique newly
created tracker objects: wrongly shared IDs can be incident to multiple people.
The diagnostic clean join differs from official matching, so its 165,351
matched detections and 2,453 diagnostic local ID changes must not replace
TrackEval's 165,335 TP or 2,130 IDSW.

IDSW counts local changes of a matched trajectory. Two persistent IDs, one in
each camera, can remain locally stable while representing the same person
incorrectly across cameras. A wrongly reused ID can also remain stable after
an error. CVIDF1 evaluates consistent global identity agreement over both
views and the complete video; these persistent splits and wrong reuses count
against it. Lower IDSW is therefore compatible with much worse CVIDF1.

Cumulative contamination is defined by all clean GT identities ever assigned
to an ID, independently of the tracker's last-clean-label sidecar. The fraction
of active clean-labelled object steps belonging to already mixed IDs is
23.08%, 39.19%, 44.25%, and 42.32% in the four relative video quarters.
There are 61,926 clean observations reusing an already mixed ID. This is
evidence of persistent output-ID contamination, not a direct measurement of
the learned embedding's contents, and not proof that contamination alone
causes the full CVIDF1 deficit.

“False Birth” and “NULL despite detection” above are final-decision outcome
proxies. They do not by themselves diagnose a raw Birth or NULL probability
error. The separate TRAIN VAL traced replay observes probabilities, actual
geometry rejection, optimizer proposals, chosen groups and state updates.
Detection gaps are separated by GT presence in the machine-readable report;
GT absence is not itself proof of physical occlusion.

Full V3+Ω and Head-Only remain separate registered models. This completed
No-Geometry analysis neither evaluates Ω's independent benefit nor changes
R2. Any state-update repair is prepared on an isolated R3 branch using
TRAIN-only evidence; no R3 long training is authorized here.

Machine-readable evidence: [worldobject_v3_no_geometry_identity_attribution.json](../results/worldobject_v3_no_geometry_identity_attribution.json).
