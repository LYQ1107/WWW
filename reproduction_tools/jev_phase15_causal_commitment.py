"""Lawful single-decision interventions followed by actual frozen native policies.

Certified candidate selection is an offline research upper bound, never a
deployable policy. GT labels do not enter the neural model's causal tensors.
"""
import argparse
import collections
import gzip
import time
import traceback
import torch
from jev_phase15_common import *
from jev_phase13_runtime import build_tracker, cache_inputs, run
from jev_phase14_artifacts import load_dense
from jev_phase14_native_risk import NativeRisk
from jev_phase15_causal_metrics import FrozenClearFuture
from gtr.modeling.jev_phase14.memory import CachedIdentityMemory
from gtr.modeling.jev_native_state import fingerprint
from gtr.modeling.jev_stage2.assignment import lawful_choice

ACTIONS = ['SELECT_JEV_BEST_ID', 'KEEP_PREVIOUS_COMMITTED_ID',
           'SELECT_ANOTHER_CERTIFIED_CANDIDATE', 'DEFER', 'NATIVE_CONSERVATIVE_FALLBACK']


class IllegalIntervention(Exception):
    pass


def force_one_row(logits, legal, row, target):
    """A constrained native assignment with per-row DEFER and capacity one."""
    k = legal.shape[-1]
    if row < 0 or row >= len(logits): raise IllegalIntervention('requested detection is absent')
    if target < -1: raise IllegalIntervention('unsupported terminal semantics')
    if target >= 0 and (target >= k or not bool(legal[row, target])):
        raise IllegalIntervention('requested existing action is absent/illegal')
    result = logits.clone(); result[row] = -1e6
    result[row, target if target >= 0 else k] = 1e6
    if target >= 0:
        other = torch.arange(len(logits), device=logits.device) != row
        result[other, target] = -1e6
    chosen = lawful_choice(result, legal)
    assert chosen[row] == target and len([x for x in chosen if x >= 0]) == len(set(x for x in chosen if x >= 0))
    return result


class SingleNativeAction(torch.nn.Module):
    def __init__(self, first_policy, subsequent_policy, event, action):
        super().__init__(); self.first_policy = first_policy; self.subsequent_policy = subsequent_policy
        self.event = event; self.action = action; self.context = None; self.refs = []
        self.triggered = False; self.intervention = None

    def forward(self, x):
        key = (self.context['frame'], self.context['view']); wanted = tuple(self.event['key'][1:])
        if key != wanted or self.triggered: return self.subsequent_policy(x)
        assert int(x['question_type'][0, 0]) == 0
        original = self.first_policy(x); scores = original[0]; legal = x['legal'][0]; row = self.event['row']
        choice = lawful_choice(scores, legal); target = choice[row]
        if self.action == 'KEEP_PREVIOUS_COMMITTED_ID':
            ref = self.event['previous_native_id']
            if ref not in self.refs: raise IllegalIntervention('previous ID is not a legal active MATCH-stage candidate; no candidate fabricated')
            target = self.refs.index(ref)
        elif self.action == 'SELECT_ANOTHER_CERTIFIED_CANDIDATE':
            gt = self.event['GT_OFFLINE_ONLY']
            query = next(q for q in self.event['queries'] if q['task'] == 0)
            assert query['refs'] == self.refs
            certified = [j for j, ref in enumerate(self.refs) if bool(legal[row, j]) and
                         set(map(int, query['candidate_history_GT_OFFLINE_ONLY'][str(ref)])) == {gt} and j != choice[row]]
            if not certified: raise IllegalIntervention('no another certified pure candidate exists; no synthetic candidate')
            target = max(certified, key=lambda j: float(scores[row, j]))
        elif self.action == 'DEFER': target = -1
        elif self.action == 'NATIVE_CONSERVATIVE_FALLBACK':
            values = x['pair_evidence'][0, :, :, [0, 1, 2]].max(-1).values
            fallback = torch.cat([values, values.new_full((len(values), 1), .75)], 1)
            target = lawful_choice(fallback, legal)[row]
        elif self.action != 'SELECT_JEV_BEST_ID': raise ValueError(self.action)
        if self.action != 'SELECT_JEV_BEST_ID':
            result = force_one_row(scores, legal, row, target)[None]
        else: result = original
        self.triggered = True
        self.intervention = {'key': self.event['key'], 'row': row, 'action': self.action,
            'legal_refs': self.refs.copy(), 'requested_MATCH_column': target,
            'requested_existing_ID': self.refs[target] if target >= 0 else None,
            'first_frozen_jev_assignment': choice, 'first_causal_tensors_SHA256': fingerprint(x),
            'first_original_scores_SHA256': fingerprint(scores),
            'forced_current_scores_SHA256': fingerprint(result[0]),
            'same_camera_one_to_one': True, 'DEFER_is_not_forced_START_NEW': True,
            'oracle_candidate_uses_offline_GT_certificate': self.action == 'SELECT_ANOTHER_CERTIFIED_CANDIDATE'}
        return result


def risk_at_horizon(risk, boundary, end):
    """Bounded observed consequences; pure past owners and UNKNOWN stay separate."""
    votes = collections.defaultdict(collections.Counter)
    for identity, history in risk.prefix_votes.items(): votes[identity].update(history)
    owners = {identity: next(iter(history)) for identity, history in votes.items() if len(history) == 1}
    latest = dict(risk.prefix_latest); ever = set(risk.prefix_ever); counts = collections.Counter()
    durations = collections.defaultdict(list); active = {}; trace = []
    for event in risk.trace:
        frame, view = event['key']
        if frame >= end: break
        identity = event['identity']; gt = event['GT_OFFLINE_ONLY']; action = event['action']
        if gt is None: counts['unassessed_GT_observations'] += 1; continue
        before = votes[identity]; previous = latest.get((gt, view))
        if previous and previous[1] != identity:
            counts['observed_same_GT_identity_changes'] += 1
            if set(votes[previous[1]]) == {gt} and set(before) == {gt}: counts['pure_fragment_hops'] += 1
            elif len(before) > 1: counts['polluted_history_switches'] += 1
            elif set(before) == {gt} and len(votes[previous[1]]) > 1: counts['history_corrective_switches'] += 1
            else: counts['ambiguous_or_other_changes'] += 1
        elif previous: counts['same_GT_identity_continuations'] += 1
        counts['new_cross_GT_gallery_mix'] += bool(before) and gt not in before
        counts['writes_into_already_polluted_history'] += len(before) > 1
        counts['false_birth'] += action == 'START_NEW' and gt in ever
        counts['false_split_birth'] += action == 'START_NEW' and bool(event['pure_correct_refs'])
        counts['UNKNOWN_selected'] += event['selected_tag'] == 'SELECTED_UNKNOWN'
        other = latest.get((gt, 1 - view))
        if other and frame - other[0] <= 40:
            counts['cross_camera_continuity_observations'] += 1
            counts['cross_camera_ID_mismatch_observations'] += other[1] != identity
        if identity not in owners and not before: owners[identity] = gt
        assessed = identity in owners; wrong = assessed and owners[identity] != gt
        counts['past_pure_owner_assessed_observations'] += int(assessed)
        counts['past_pure_owner_wrong_observations'] += int(wrong)
        counts['past_mixed_owner_UNKNOWN_observations'] += int(not assessed)
        episode = (gt, view)
        if wrong:
            state = active.get(episode)
            if state is None or frame != state[-1] + 1:
                if state: durations[episode].append(state)
                active[episode] = [frame]
            else: state.append(frame)
        elif episode in active: durations[episode].append(active.pop(episode))
        before[gt] += 1; ever.add(gt); latest[gt, view] = (frame, identity)
        trace.append(event)
    intervals = [{'GT_OFFLINE_ONLY': gt, 'view': view, 'start': values[0], 'end': values[-1],
                  'observed_wrong_frames': len(values), 'right_censored': False}
                 for (gt, view), groups in durations.items() for values in groups]
    intervals += [{'GT_OFFLINE_ONLY': gt, 'view': view, 'start': values[0], 'end': values[-1],
                   'observed_wrong_frames': len(values), 'right_censored': True}
                  for (gt, view), values in active.items()]
    return {'counts': dict(counts), 'error_propagation_intervals': intervals,
        'identity_error_scope': 'fixed certified pure prefix owner; new identity owner from first observed GT; mixed prefix owners remain UNKNOWN, not silently correct',
        'fragment_hop_is_not_CLEAR_IDSW': True, 'cross_camera_diagnostic_is_not_official_CVIDF1': True}


def main(video, continuation, duplicate_gt_diagnostic=False):
    protect(); assert video in TRAIN + DEV
    training_case = OUT / 'train_commitment_prefixes_v1' / f'video{video:02d}'
    replay = (training_case / 'RESULT.json') if video in TRAIN else (
        OUT / 'native_replay_v2/multi_question_seed20261009' / f'video{video:02d}' / 'RESULT.json')
    assert read(replay)['status'] == 'PASS'
    manifest = read(replay); source = binding(seed=20261009, dataset=ref(ANNOTATIONS),
        evaluator='unchanged TrackEval CLEAR plus explicitly scoped causal history risk',
        scope='actual mutated-state H8/H16/H32 futures; complete original history retained')
    first, first_ck, _ = load_old_policy('multi_question', 20261009)
    variant = 'multi_question' if continuation == 'pi_multi_frozen20k' else 'fixed_question'
    subsequent, next_ck, _ = load_old_policy(variant, 20261009)
    source['checkpoints'] = {'current_decision': first_ck, 'subsequent_policy': next_ck}
    source['perception_input_provenance'] = perception_provenance(video)
    torch.set_num_threads(1); torch.manual_seed(20261009)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    values, total_frames, reader = cache_inputs(video)
    metric = FrozenClearFuture('multi_question', 20261009, video, case_override=training_case if video in TRAIN else None,
                               duplicate_gt_diagnostic=duplicate_gt_diagnostic)
    source['strict_CLEAR_causal_gate_eligible'] = not duplicate_gt_diagnostic
    source['raw_GT_duplicate_audit'] = metric.evaluator
    event_path = training_case / 'SWITCH_EVENTS.jsonl.gz' if video in TRAIN else (
        OUT / 'failure_reconstruction_v1/multi_question_seed20261009' / f'video{video:02d}' / 'SWITCH_EVENTS.jsonl.gz')
    with gzip.open(event_path, 'rt') as stream: all_events = [json.loads(line) for line in stream]
    by_event = {(tuple(e['key']), e['row']): e for e in all_events}
    folder = OUT / ('causal_commitment_v2_duplicate_diagnostic' if duplicate_gt_diagnostic else 'causal_commitment_v1') / continuation / f'video{video:02d}'
    folder.mkdir(parents=True, exist_ok=True); results = []; begin = time.monotonic()
    for item in manifest['counterfactual_prefixes']:
        boundary = tuple(item['key'][1:]); event = by_event[tuple(item['key']), item['row']]
        assert sha(item['prefix']['path']) == item['prefix']['SHA256']
        baseline = None
        for action in ACTIONS:
            name = f'frame{boundary[0]:06d}_view{boundary[1]}_row{item["row"]}_{action}'
            path = folder / (name + '.json')
            if path.exists():
                result = read(path)
                assert result['binding'] == source
            else:
                storage_guard(); prefix = load_dense(item['prefix']['path'], map_location='cuda:0')
                initial = fingerprint(prefix); assert initial == item['starting_state_SHA256']
                wrapper = SingleNativeAction(first, subsequent, event, action)
                model = build_tracker(video, policy=wrapper, variant='full', temperature=(1., 1.), react_learned=False)
                executor = model.jev_stage2_executor; executor.memory_factory = CachedIdentityMemory
                original_scores = executor.scores
                def scores(batch, refs, context, task):
                    wrapper.context = context; wrapper.refs = list(refs)
                    return original_scores(batch, refs, context, task)
                executor.scores = scores
                risk = NativeRisk(video, reader, prefix)
                risk.prefix_votes = {k: dict(v) for k, v in risk.votes.items()}
                risk.prefix_latest = dict(risk.latest); risk.prefix_ever = set(risk.ever)
                executor.observer = risk.before; executor.commit_observer = risk.after
                stop = min(total_frames - 1, boundary[0] + 31)
                unchanged = {key: list(ids) for key, ids in metric.ids(prefix['instances']).items() if key < boundary}
                result = {'binding': source, 'prefix': item, 'action': action, 'subsequent_policy': continuation,
                    'starting_state_SHA256': initial, 'same_saved_RNG': True, 'deployment_use_allowed': False,
                    'neural_model_GT_inputs': False, 'intervention_OFFLINE_research_only': True,
                    'candidate_sets_and_native_lifecycle_unchanged': True, 'GT_annotation_rewritten': False}
                try:
                    with torch.no_grad(): raw, _ = run(model, values, total_frames, stop=stop, prefix=prefix)
                    assert wrapper.triggered, 'requested actual detection never reached'
                    ids = metric.ids(raw)
                    assert all(ids[key] == old for key, old in unchanged.items()), 'historical committed identity changed'
                    expected_payloads = 2 * (stop - boundary[0] + 1) - boundary[1]
                    assert risk.counts['payloads'] == expected_payloads
                    horizons = {}
                    for horizon in [8, 16, 32]:
                        end = min(total_frames, boundary[0] + horizon)
                        horizons[str(horizon)] = dict(metric.horizon(ids, boundary, horizon, total_frames),
                            identity_consequences=risk_at_horizon(risk, boundary, end))
                    trace = folder / (name + '.jsonl.gz')
                    with gzip.open(trace, 'wt') as stream:
                        for row in risk.trace: stream.write(json.dumps(row, allow_nan=False) + '\n')
                    observed = next(row for row in risk.trace if row['key'] == list(boundary) and row['row'] == event['row'])
                    result.update(status='COMPLETE', intervention=wrapper.intervention,
                        actual_first_committed_ID=observed['identity'], actual_first_native_action=observed['action'],
                        H= horizons, committed_future_trace=ref(trace), actual_mutated_future=True,
                        every_past_camera_history_preserved=True, final_identity_memory_SHA256=fingerprint(executor.memory.state_dict()),
                        future_committed_IDs_SHA256=fingerprint({key: value for key, value in ids.items() if key >= boundary}))
                except IllegalIntervention as e:
                    result.update(status='NOT_RUN_ACTION_NOT_LEGAL', reason=str(e), H=None, actual_mutated_future=False)
                save(path, result); del prefix, model, risk, wrapper; torch.cuda.empty_cache()
            if action == 'SELECT_JEV_BEST_ID':
                assert result['status'] == 'COMPLETE'; baseline = result
            assert baseline is not None and result['starting_state_SHA256'] == baseline['starting_state_SHA256']
            if result['status'] == 'COMPLETE':
                assert result['intervention']['first_causal_tensors_SHA256'] == baseline['intervention']['first_causal_tensors_SHA256']
                assert result['intervention']['first_original_scores_SHA256'] == baseline['intervention']['first_original_scores_SHA256']
            results.append(ref(path)); print('PHASE15_NATIVE_CAUSAL_BRANCH', video, continuation, boundary, action, result['status'],
                {h: r['future_CLEAR_IDSW'] for h, r in (result.get('H') or {}).items()}, flush=True)
            save(folder / 'PROGRESS.json', {'status': 'RUNNING', 'done': len(results),
                 'total': len(manifest['counterfactual_prefixes']) * len(ACTIONS), 'last': [list(boundary), action],
                 'seconds': time.monotonic() - begin})
    save(folder / 'RESULT.json', {'status': 'COMPLETE', 'binding': source, 'video': video,
         'subsequent_policy': continuation, 'branches': results, 'replay_manifest': ref(replay),
         'seconds': time.monotonic() - begin, 'GT_or_future_in_online_policy': False})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--video', type=int, required=True)
    p.add_argument('--continuation', choices=['pi_multi_frozen20k', 'pi_fixed_frozen20k'], required=True)
    p.add_argument('--duplicate-gt-diagnostic',action='store_true')
    args = p.parse_args()
    try: main(args.video, args.continuation, args.duplicate_gt_diagnostic)
    except Exception: failure('causal_commitment', traceback.format_exc()); raise
