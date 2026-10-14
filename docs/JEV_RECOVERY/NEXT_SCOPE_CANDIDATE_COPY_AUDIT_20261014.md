# Historical candidate-copy audit (2026-10-14)

A metadata-only audit found a historical `Wildtrack_dataset` copy. It did not open image, label, video, model, or archive payloads and did not train or run native tracking.

- Candidate root: `/data1/liuyeqiang/WWW_jev_phase14_runtime/20261010_v1/external_wildtrack_v1`
- Status: `CANDIDATE_COPY_FOUND_AUTHORIZATION_UNRESOLVED`
- Existing files by suffix: `.json=325, .png=640`
- Existing filesystem bytes (stat sum): `1772978015`
- Metadata bytes read: `913711` (limit 67108864)
- Authorized dataset path: `false`
- Authorization/provenance status: `UNRESOLVED`
- New retained bytes: `0`
- B-stage training/native intervention: `NOT_AUTHORIZED`

The local files contain a previous audit binding and a completed member download state, but those facts do not establish that the current user or project may use the copy. The official access record must be supplied before any image/label use or independent experiment.
