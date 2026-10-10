"""Freeze admission and negative-control causal protocol before interventions."""
from jev_phase16_common import *


def main():
    protect()
    path = REPORTS/'P2_P3_PROTOCOL.json'
    assert not path.exists()
    assert read(REPORTS/'MULTI_PROTOTYPE_RESULTS.json')['selected_reader'] is None
    cases = []
    for video in TRAIN:
        manifest = read(XV/'train_commitment_prefixes_v1'/f'video{video:02d}/RESULT.json')
        for e in manifest['counterfactual_prefixes']:
            cases.append(dict(video=video,key=e['key'],row=e['row'],prefix=e['prefix'],native_state=e['starting_state_SHA256']))
    strict = [e for e in cases if e['video'] != 14]
    h64 = sorted(strict,key=lambda e:(e['video'],e['key'],e['row']))[:8]
    protocol = dict(status='FROZEN_BEFORE_P2_P3_RESULTS',binding=binding(scope='GT-free admission and bounded causal diagnostics; no training'),
        scientific_version='V2_recovery_plus_safe_memory',major_hypotheses_used=['V1_frozen_evidence_readers','V2_safe_memory'],
        P1_status='NO_GO_PROTOTYPE_DEPLOYMENT',prototype_training_allowed=False,
        rules=['always_write','recent_only','conservative_provisional','delayed_confirmation','bounded_quarantine'],
        selected_rule='bounded_quarantine',selected_without_DEV_or_outcome_optimization=True,
        constants=dict(anchor_cosine=.85,nearby_normalized_box_mean_L1=.05,nearby_max_scene_gap=2,
            confirmation_observations=3,pending_limit=4,quarantine_limit=4),
        anchors='Earliest GT-free consistent same-camera triple, fixed once established; not purity certification',
        existing_prefix='All original past writes retained as CONFIRMED_LEGACY_UNVALIDATED in every arm; no GT cleanup',
        statuses=['CONFIRMED','PROVISIONAL','QUARANTINED','CONFIRMED_LEGACY_UNVALIDATED'],
        raw_invariants=['actual committed ID','native raw Gallery','hits','Bank','candidate ID legality','ordinary posterior update'],
        safe_view='Admitted raw-row pointers for four WHO slots and independent trusted Q4 predecessor state; same soft raw posterior reliability',
        fallback='Cold IDs retain real past pending or last quarantined row; fallback explicitly untrusted; no invented detection or birth',
        no_new_WRITE_head=True,GT_or_future_actor_inputs=False,
        feature_budget='No extra stored raw observations; trusted per-camera sum/last/EMA <=6 vectors perID. Anchor/pending/quarantine store raw row indices. Temporary original Gallery means unchanged.',
        metadata_budget='Append-only status/geometry metadata O(raw observations); not a bounded total-memory deployment claim',
        shadow_scope=dict(videos=TRAIN+DEV,source='P0 frozen V3 actual commits + exact cache features',learning_updates=0),
        factorial=dict(A='original decision + original unconditional memory',B='original decision + bounded quarantine memory',
            C='E reader at one targeted first decision + original memory',D='E reader at one targeted first decision + bounded quarantine memory'),
        evidence_factor='Failed P1 E is a fixed negative-control, single-target first-decision intervention only; no persistent reader expansion or promotion',
        targeted_row_only=True,H=[8,16,32],H64_prefixes=h64,all_XV_24_prefixes=cases,
        same_original_prefix_RNG=True,mutated_future_never_synchronized=True,
        future_policy='Frozen XV V3 native policy and original four-slot decoder in all arms; B/D apply admission to all actual future writes',
        candidate_sets='Same native constructor/lifecycle; future sets may diverge causally after actual commits',
        safety_gate=dict(min_independent_benefit_clusters=8,forbid_worse_new_mixing_at_any_strict_prefix_H=True,
            forbid_worse_wrong_owner_observations_or_episode_duration=True,forbid_worse_false_split_birth=True,
            TRAIN14='duplicate GT diagnostic excluded',independent_cluster='video, anchored target GT, view, frame//64; repeated prefixes not independent'),
        continuation='Training requires qualified evidence, admission safety and labels. P1 failed frozen gate: no learned reader Tiny/Pilot/formal. Full-image memory-only evaluation only if native admission safety qualifies; else explicitly notrun/null.',
        gates_not_relaxed=True)
    save(path,protocol)
    print('PHASE16_P2_P3_PROTOCOL_FROZEN',len(cases),len(h64),flush=True)


if __name__=='__main__':main()
