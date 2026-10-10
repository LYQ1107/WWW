# Phase XVI evidence availability and recovery

The original raw Gallery and hits remain unchanged. Annotation is an offline observer only. A person appearing in an ID history is different from that Global ID having a certified past owner. The owner anchor is established only at the earliest past moment with >=3 known observations, >=0.8 coverage and one GT, then remains immutable. Histories mixed before such an anchor remain unanchored.

P0 includes19 complete native replays and72 H32 branches at all24 original TRAIN prefixes under all three XV checkpoints. The additional six early provenance windows are retained engineering qualification, not independent evidence. A-G flags overlap; the primary category precedence is E/C/D/F/B/A/G. Unknown observations cannot be labelled wrong. A denotes no observed target content, with UNKNOWN-history ambiguity disclosed.

C includes actual stale Bank membership and a potential promotion proxy: an inactive poss-ID with enough Gallery observations. The latter does not assert the Bank was offered; promotion additionally needs MATCH DEFER and nonempty instances_old. D records an ID excluded by the frozen lifecycle, not an illegal candidate fabrication or a demonstrated implementation bug. Actual offered refs are retained per query.

Default REACT is untrained: actual Bank IDs are offered only after MATCH DEFER, then max(last, global mean, own-camera mean) cosine is compared through the same one-to-one solver with0.75 terminal. The fourth other-camera token is used by the WHO model but excluded from this fallback. No Bank candidate or detection is fabricated. Every query logs actual frozen logits and both summary score scopes.

Primary V3 TRAIN diagnostic:

{
  "counts": {
    "ASSOCIATE_EXISTING": 122299,
    "REACTIVATE": 4,
    "START_NEW": 48,
    "all_queries": 122403,
    "anchored_correct_owner_exists": 109609,
    "current_GT_known": 117147,
    "high_risk_GT_known_queries": 48214,
    "high_risk_queries": 53470,
    "legal_pure_correct_available": 71122,
    "observed_correct_content_exists": 117078,
    "owner_anchored_summary_hidden_queries": 63325,
    "payloads": 9644,
    "qualified_raw_content_exists": 116844,
    "raw_UNKNOWN_observations_at_query": 117968522,
    "raw_known_observations_at_query": 2383549895,
    "selected_UNKNOWN_history": 48089,
    "selected_certified_correct": 68933,
    "selected_certified_wrong": 57,
    "selected_globally_mixed": 47536,
    "summary_hidden_queries": 76629
  },
  "high_risk_primary": {
    "A_ABSENT_FROM_ALL_HISTORY": 69,
    "B_PRESENT_IN_RAW_GALLERY": 3,
    "E_CANDIDATE_PRESENT_BUT_WRONG_SELECTION": 2189,
    "F_HISTORY_CONTAMINATED": 42279,
    "G_AMBIGUOUS": 5547,
    "C_PRESENT_IN_STALE_BANK": 3383
  },
  "high_risk_multi_label": {
    "A_ABSENT_FROM_ALL_HISTORY": 69,
    "B_PRESENT_IN_RAW_GALLERY": 38004,
    "C_PRESENT_IN_STALE_BANK": 3383,
    "D_CANDIDATE_ELIGIBILITY_FAILURE": 0,
    "E_CANDIDATE_PRESENT_BUT_WRONG_SELECTION": 2189,
    "F_HISTORY_CONTAMINATED": 47815,
    "G_AMBIGUOUS": 47853
  },
  "native_action_consequences": {
    "commits": 122351,
    "START_NEW": 48,
    "REACTIVATE": 4,
    "anchored_owner_wrong_observations": 13227,
    "new_cross_GT_mixing": 182,
    "previously_observed_GT_birth_proxy": 10,
    "false_split_birth_with_legal_pure_correct_ID": 0
  },
  "cases": 4,
  "actual_untrained_REACT_diagnostics": {
    "actual_REACT_queries": 52,
    "GT_known": 35,
    "actual_candidates_nonempty": 19,
    "pure_correct_stale_offered": 0,
    "pure_correct_selected": 0,
    "owner_raw075_support_but_three_summary_below075": 0,
    "other_camera_slot_only_crosses075_for_owner": 0
  },
  "videos": [
    12,
    13,
    14,
    16
  ],
  "TRAIN14_in_all_query_denominator": true,
  "strict_TRAIN_counts_without_TRAIN14": {
    "ASSOCIATE_EXISTING": 90015,
    "REACTIVATE": 4,
    "START_NEW": 39,
    "all_queries": 90101,
    "anchored_correct_owner_exists": 82019,
    "current_GT_known": 85852,
    "high_risk_GT_known_queries": 36598,
    "high_risk_queries": 40847,
    "legal_pure_correct_available": 51174,
    "observed_correct_content_exists": 85793,
    "owner_anchored_summary_hidden_queries": 42224,
    "payloads": 7707,
    "qualified_raw_content_exists": 85618,
    "raw_UNKNOWN_observations_at_query": 107127932,
    "raw_known_observations_at_query": 1872578514,
    "selected_UNKNOWN_history": 36493,
    "selected_certified_correct": 49254,
    "selected_certified_wrong": 45,
    "selected_globally_mixed": 36021,
    "summary_hidden_queries": 51414
  },
  "strict_TRAIN_action_consequences_without_TRAIN14": {
    "commits": 90058,
    "START_NEW": 39,
    "REACTIVATE": 4,
    "anchored_owner_wrong_observations": 12137,
    "new_cross_GT_mixing": 144,
    "previously_observed_GT_birth_proxy": 7,
    "false_split_birth_with_legal_pure_correct_ID": 0
  },
  "cluster_counts": {
    "high_risk_known": 810,
    "high_risk_owner_anchored_summary_loss": 496,
    "owner_anchored_summary_loss": 926
  },
  "observed_correct_content_rate": 0.9994109964403698,
  "anchored_owner_content_rate": 0.935653495181268,
  "selected_correct_on_GT_known_rate": 0.5884316286375237,
  "high_risk_GT_known_queries": 48214,
  "query_counts_are_not_independent_clusters": true
}

Bounded retrieval trial eligible: True. This is an information-availability gate; it is not safe-recovery or learned-JEV success. All-query TRAIN denominators retain video14 as a disclosed duplicate-GT diagnostic; strict_TRAIN_counts_without_TRAIN14 and qualifying cluster counts exclude it. Continuous cosine gaps need positive-retrieval and false-activation controls before native benefit can be claimed. Development results never set training labels, thresholds or eligibility.
