# Phase X native gallery repair

All 221 frozen Phase IX records (220 distinct boundaries) pass. All 94 records that failed the birth-only reconstruction remain in the event set. No old snapshot or checkpoint was overwritten.

The repair uses a serialized production prefix before current get_asso/MATCH, after current perceptions have been loaded. It preserves actual ordered Gallery Instances, history storage, hits, bank containers, ID counter and trajectory/Python/NumPy/Torch RNG. The fork resumes the production sliding loop, so it does not reconstruct Gallery from hits or the old Replay memory list.

Every fixed boundary preserves candidate order, raw association scores, Hungarian proposal and legal masks. Restored state64/evidence12 are within 1e-6; complete prefix and current native commit containers, observation events and RNG match exactly. Each full source video also has separate OFF/compat full-state and identity equality. Known video12 identity5 bank reactivation at frame839 and Gallery state at frame851 are separately recorded in the JSON reports.

This establishes the state repair, not validity of old H32 utilities or any learned architecture. Those remain gated until all frozen counterfactual branches run again. Large prefixes and full event traces remain on the server; compact reports bind their checks by SHA256.
