# Full association-TRAIN causal identity audit

Status: INCOMPLETE. Complete scenes: [4, 5, 6, 7, 8, 9, 13, 17, 20, 21, 24]. Pending scenes: [12].

Each sequence starts at group1 and runs continuously to the actual final group using the frozen GMT checkpoint and seed20261014. DecisionTrace observes the original Hungarian/threshold/memory methods. GT is read only after full inference and independent uninstrumented OFF40 equality. Existing400-prefix collectors must agree. No VGGT/DINO, future or GT tracker features. Detailed examples are bounded1000/scene; complete lightweight query indices retain coverage. Runtime is offline audit cost, not deployment FPS.

Raw GT conflicts in association scenes20/24 are preserved. Entire camera-frames containing duplicate or invalid GT are unreliable during post-run proposal matching; subsequent reference purity includes their unknown labels. These are conservatively masked diagnostics, not repaired official evaluations.

|Scene|Groups|Known queries|Known wrong/missed|Pure positive full|Top8 numerator /denominator|Impure histories|Unknown|Status|
|---|---:|---:|---:|---:|---|---:|---:|---|
|4|901|3557|243|3457|3432 /3457|2907|308|COMPLETE|
|5|1769|6912|52|6879|6879 /6879|1331|91|COMPLETE|
|6|601|2571|29|2532|2532 /2532|1525|68|COMPLETE|
|7|443|1693|64|1558|1558 /1558|734|324|COMPLETE|
|8|523|1969|25|1891|1891 /1891|1322|514|COMPLETE|
|9|1942|8678|171|8600|8600 /8600|2044|192|COMPLETE|
|12|1200|NOT_MEASURED|NOT_MEASURED|NOT_MEASURED|NOT_MEASURED|NOT_MEASURED|NOT_MEASURED|INCOMPLETE|
|13|1621|16795|28|15691|15691 /15691|9445|1430|COMPLETE|
|17|1200|11002|42|10025|10023 /10025|9236|1514|COMPLETE|
|20|1001|23599|157|19956|19627 /19956|23746|5500|COMPLETE|
|21|1001|6952|9|5619|5619 /5619|7117|1747|COMPLETE|
|24|1201|57322|707|37245|34539 /37245|69993|38451|COMPLETE|

Counts overlap and describe the frozen native history. Known wrong includes pure known wrong links and rejected queries with a pure positive, not unknown selections. A ranking error can already be corrected by Hungarian. Missing pure positive can coexist with an impure track containing a correct observation. Candidate coverage divides by known queries with a pure full-history positive. Correctable error rate divides by reliably known primary wrong/missed queries. Neither is a HOTA/CVIDF1 upper bound.

Actual raw/GTA ranks, Hungarian pairs, strict thresholds, memory reactivation, new-ID allocation and previous-group output anchors are separate in each sealed summary. Descriptor averages are not assigned GT from source tags. Final output IDs are permutation invariant; query counts cannot establish an end-to-end metric gain.
