"""Attribute executed native commitment effects; never pool different policies."""
import collections
import gzip
from jev_phase15_common import *


def rows(path):
    with gzip.open(path, 'rt') as stream: return [json.loads(line) for line in stream]


def original_ids(video):
    case = old_case('multi_question', 20261009, video) if video in DEV else (
        OUT / 'train_commitment_prefixes_v1' / f'video{video:02d}')
    ann = read(ANNOTATIONS); images = {im['id']: im for im in ann['images'] if im['video_id'] == video}
    grouped = collections.defaultdict(list)
    for p in read(case / 'RAW_PREDICTIONS.json'):
        im = images[p['image_id']]; grouped[im['frame_id'] - 1, im['view_id'] - 1].append(p['track_id'])
    return grouped


def certify_original_policy_baseline(branch, video, frozen):
    trace = rows(branch['committed_future_trace']['path']); grouped = collections.defaultdict(dict)
    for event in trace: grouped[tuple(event['key'])][event['row']] = event['identity']
    boundary = tuple(branch['prefix']['key'][1:]); end = boundary[0] + branch['H']['32']['scene_frames_observed']
    count = 0
    for frame in range(boundary[0], end):
        for view in [0, 1]:
            key = frame, view
            if key < boundary: continue
            actual = [ref for _, ref in sorted(grouped[key].items())]
            assert actual == frozen[key], ('unchanged pi_multi future did not replay original IDs', video, key)
            count += len(actual)
    return {'status': 'PASS', 'future_committed_detections_exact': count,
            'comparison': 'unaltered first decision + pi_multi must equal frozen original native future'}


def corrected_propagation_censoring(branch):
    """Do not interpret an observation gap or UNKNOWN as certified correction."""
    trace = rows(branch['committed_future_trace']['path'])
    corrected = {}
    for horizon, metrics in branch['H'].items():
        end = branch['prefix']['key'][1] + metrics['scene_frames_observed']
        intervals = []
        for item in metrics['identity_consequences']['error_propagation_intervals']:
            item = dict(item)
            correction = any(e['key'][0] == item['end'] + 1 and e['key'][0] < end and
                e['key'][1] == item['view'] and e['GT_OFFLINE_ONLY'] == item['GT_OFFLINE_ONLY'] and
                e['selected_tag'] == 'CERTIFIED_CORRECT' for e in trace)
            item['right_censored'] = item['right_censored'] or not correction
            item['termination_certificate'] = 'next-frame pure certified target' if correction else 'gap/UNKNOWN/horizon end remains censored'
            intervals.append(item)
        corrected[horizon] = intervals
    return corrected


def main():
    protect(); complete = []; pending = []; branch_records = []; comparisons = []; oracles = []
    original = {}; positive = collections.defaultdict(set); safe_support = collections.defaultdict(set)
    policy_counts = collections.defaultdict(collections.Counter)
    for video in DEV + TRAIN:
        for policy in ['pi_multi_frozen20k', 'pi_fixed_frozen20k']:
            folder = OUT / 'causal_commitment_v1' / policy / f'video{video:02d}'
            final = folder / 'RESULT.json'
            if not final.exists(): pending.append({'video': video, 'policy': policy}); continue
            summary = read(final); assert summary['status'] == 'COMPLETE'
            complete.append(ref(final)); grouped = collections.defaultdict(dict)
            for descriptor in summary['branches']:
                assert sha(descriptor['path']) == descriptor['SHA256']
                branch = read(descriptor['path']); item = branch['prefix']
                key = tuple(item['key']), item['row']; grouped[key][branch['action']] = branch, descriptor
                if branch['status'] == 'COMPLETE':
                    assert sha(branch['committed_future_trace']['path']) == branch['committed_future_trace']['SHA256']
                branch_records.append({'video': video, 'scope': 'TRAIN' if video in TRAIN else 'DEVELOPMENT_DIAGNOSTIC',
                    'policy': policy, 'key': item['key'], 'row': item['row'], 'action': branch['action'],
                    'status': branch['status'], 'native_branch': descriptor,
                    'actor_source_commit': branch['binding']['source_commit'], 'starting_state_SHA256': branch['starting_state_SHA256'],
                    'H': branch.get('H'), 'GT_in_model_inputs': False,
                    'offline_research_intervention': True,
                    'corrected_observation_gap_censoring': corrected_propagation_censoring(branch) if branch['status'] == 'COMPLETE' else None})
                policy_counts[policy]['executed' if branch['status'] == 'COMPLETE' else 'not_legal'] += 1
            for key, actions in grouped.items():
                baseline, baseline_ref = actions['SELECT_JEV_BEST_ID']
                item = baseline['prefix']; group = (video, item['GT_OFFLINE_ONLY'], item['key'][2], item['key'][1] // 64)
                if policy == 'pi_multi_frozen20k':
                    if video not in original: original[video] = original_ids(video)
                    certify_original_policy_baseline(baseline, video, original[video])
                    policy_counts[policy]['original_future_parity_prefixes'] += 1
                candidates = []
                for action, (branch, descriptor) in actions.items():
                    if branch['status'] != 'COMPLETE': continue
                    assert branch['starting_state_SHA256'] == baseline['starting_state_SHA256']
                    assert branch['intervention']['first_causal_tensors_SHA256'] == baseline['intervention']['first_causal_tensors_SHA256']
                    assert branch['intervention']['first_original_scores_SHA256'] == baseline['intervention']['first_original_scores_SHA256']
                    deltas = {}
                    for horizon in ['8', '16', '32']:
                        b = baseline['H'][horizon]; r = branch['H'][horizon]
                        before = b['identity_consequences']['counts']; after = r['identity_consequences']['counts']
                        fields = ['new_cross_GT_gallery_mix', 'false_birth', 'false_split_birth',
                                  'past_pure_owner_wrong_observations', 'writes_into_already_polluted_history',
                                  'pure_fragment_hops', 'polluted_history_switches', 'history_corrective_switches',
                                  'cross_camera_ID_mismatch_observations', 'UNKNOWN_selected']
                        deltas[horizon] = {'future_CLEAR_IDSW': r['future_CLEAR_IDSW'] - b['future_CLEAR_IDSW'],
                                           **{name: after.get(name, 0) - before.get(name, 0) for name in fields}}
                    adverse_merge = any(deltas[h]['new_cross_GT_gallery_mix'] > 0 for h in deltas)
                    adverse_birth = any(deltas[h]['false_birth'] > 0 for h in deltas)
                    benefit = deltas['32']['future_CLEAR_IDSW'] < 0 and not adverse_merge and not adverse_birth
                    if action == 'KEEP_PREVIOUS_COMMITTED_ID' and item['pure_previous_continuation_candidate']:
                        safe_support[policy].add(group)
                        if benefit: positive[policy].add(group)
                    if benefit: policy_counts[policy]['favorable_legal_actions'] += 1
                    if adverse_merge: policy_counts[policy]['actions_increasing_certified_merge'] += 1
                    if adverse_birth: policy_counts[policy]['actions_increasing_false_birth'] += 1
                    comparisons.append({'video': video, 'scope': 'TRAIN' if video in TRAIN else 'DEVELOPMENT_DIAGNOSTIC',
                        'policy': policy, 'key': item['key'], 'row': item['row'], 'category': item['category'],
                        'action': action, 'baseline': baseline_ref, 'branch': descriptor, 'deltas': deltas,
                        'safe_previous_certification_before_current_action': item['pure_previous_continuation_candidate'],
                        'H32_fewer_IDSW_no_observed_merge_or_birth_increase': benefit,
                        'genuine_future_and_same_start_and_RNG': True})
                    if not adverse_merge and not adverse_birth: candidates.append((branch['H']['32']['future_CLEAR_IDSW'], action, descriptor))
                best = min(candidates, key=lambda x: (x[0], x[1])) if candidates else None
                oracles.append({'video': video, 'scope': 'TRAIN' if video in TRAIN else 'DEVELOPMENT_DIAGNOSTIC',
                    'policy': policy, 'key': item['key'], 'row': item['row'], 'baseline_H32_IDSW': baseline['H']['32']['future_CLEAR_IDSW'],
                    'best_executed_action_H32_IDSW': best[0] if best else None,
                    'posthoc_offline_oracle_action': best[1] if best else None, 'oracle_branch': best[2] if best else None,
                    'posthoc_summary_is_not_a_deployable_action_selector': True,
                    'actual_executed_actions_were_chosen_without_future_frames': True})
    done = not pending; source = binding(seed=20261009, dataset=ref(ANNOTATIONS),
        evaluator='actual frozen native branches + unmodified TrackEval CLEAR',
        scope='conditional action attribution; overlapping windows are not full-video gains')
    result = {'status': 'COMPLETE' if done else 'RUNNING', 'binding': source, 'case_manifests': complete,
        'pending': pending, 'branches': branch_records, 'paired_comparisons': comparisons,
        'policy_specific_counts': {p: dict(c) for p, c in policy_counts.items()},
        'future_values_are_policy_specific': True, 'different_policies_pooled_as_one_value': False,
        'unaltered_pi_multi_matches_original_future': True if original else None,
        'no_past_ID_rewriting_or_candidate_fabrication': True,
        'full_video_HOTA_or_official_CVIDF1_claimed_from_prefixes': False}
    save(REPORTS / 'COMMITMENT_NATIVE_CAUSAL_BRANCHES.json', result)
    save(REPORTS / 'COMMITMENT_ORACLE_UPPER_BOUND.json', {'status': result['status'], 'binding': source,
        'cases': oracles, 'offline_upper_bound_only': True, 'supervision_eligibility': 'TRAIN only; DEVELOPMENT never enters gradients',
        'action_choices_before_execution_did_not_use_future': True,
        'best_outcome_selection_after_execution_is_for_research_bound_only': True})
    gate = {'status': 'FROZEN_EVIDENCE_DECISION' if done else 'PENDING_COMPLETE_BRANCH_MATRIX', 'binding': source,
        'all_declared_TRAIN_and_DEVELOPMENT_cases_complete': done,
        'policy_results': {p: {'certified_safe_64frame_episode_clusters': len(safe_support[p]),
                               'beneficial_safe_clusters': len(positive[p]),
                               'eight_cluster_feasibility_gate': 'PASS' if done and len(positive[p]) >= 8 else 'NO_GO' if done else 'PENDING'}
                           for p in ['pi_multi_frozen20k', 'pi_fixed_frozen20k']},
        'cluster_definition': 'video + offline GT + camera + 64-frame block; adjacent prefixes are not independent evidence',
        'safety': 'no certified merge or false birth increase at H8/H16/H32; mixed-owner risk and coverage separately recorded',
        'mechanism_GO': done and any(len(positive[p]) >= 8 for p in positive),
        'learned_Commitment_model_success': 'NOT_RUN', 'scientific_tracking_GO': 'NOT_YET_ESTABLISHED',
        'if_gate_fails': 'inspect legal candidate support, pollution and recognition/continuity label identifiability; continue evidence-driven repair, never manufacture a KEEP certificate',
        'conditional_gate_is_not_full_video_or_unique_JEV_value_proof': True}
    save(REPORTS / 'COMMITMENT_FEASIBILITY_GO_NO_GO.json', gate)
    print('PHASE15_CAUSAL_AGGREGATION', result['status'], len(complete), len(pending), gate['policy_results'], flush=True)


if __name__ == '__main__': main()
