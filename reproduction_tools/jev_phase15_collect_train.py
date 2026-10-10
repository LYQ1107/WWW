"""Natural frozen Multi predicted TRAIN states; no teacher-forced identities."""
import argparse
import collections
import gzip
import time
import traceback
import torch
from jev_phase15_common import *
from jev_phase13_runtime import build_tracker, cache_inputs, run
from jev_phase14_native_risk import NativeRisk
from jev_phase14_artifacts import save_dense
from audit_jev_stage2_gta_free import phase13_prefix
from run_jev_phase10_closed_loop import raw_predictions, metrics
from gtr.modeling.jev_phase14.memory import CachedIdentityMemory
from gtr.modeling.jev_native_state import fingerprint
from gtr.modeling.jev_stage2.assignment import lawful_choice


def main(video):
    protect(); storage_guard(); assert video in TRAIN
    torch.set_num_threads(1); torch.manual_seed(20261009)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    policy, checkpoint, training_result = load_old_policy('multi_question', 20261009)
    source = binding(seed=20261009, checkpoints=[checkpoint], dataset=ref(ANNOTATIONS),
        evaluator='actual full TRAIN native predicted histories + post-run TrackEval',
        scope='TRAIN only, no new optimizer updates; offline prefix selection')
    source['perception_input_provenance'] = perception_provenance(video)
    values, frames, reader = cache_inputs(video); risk = NativeRisk(video, reader)
    model = build_tracker(video, policy=policy, variant='full', temperature=(1., 1.), react_learned=False)
    executor = model.jev_stage2_executor; executor.memory_factory = CachedIdentityMemory
    folder = OUT / 'train_commitment_prefixes_v1' / f'video{video:02d}'; folder.mkdir(parents=True, exist_ok=True)
    assert not (folder / 'RESULT.json').exists()
    current_native = [None]; selected = []; selections = collections.Counter(); event_records = []
    pending = {}; begin = time.monotonic(); last_progress = [0.]
    journal = gzip.open(folder / 'COMMITS.jsonl.gz', 'wt')

    def before_native(**d): current_native[0] = d

    def before(**d):
        risk.before(**d)
        if d['task'] != 0 or not len(d['logits']): return
        c = d['context']; key = (c['frame'], c['view']); targets = risk.labels.current(*key)
        choice = lawful_choice(d['logits'], d['batch']['legal'][0]); refs = d['refs']
        for row, col in enumerate(choice):
            gt = targets[row]
            if gt is None: continue
            previous = risk.latest.get((gt, key[1]))
            if previous is None: continue
            old = previous[1]; new = refs[col] if col >= 0 else None
            if old == new: continue
            old_history = dict(risk.votes[old]); new_history = dict(risk.votes[new]) if new is not None else {}
            if set(old_history) == set(new_history) == {gt}: category = 'PURE_FRAGMENT_HOP'
            elif len(new_history) > 1: category = 'POLLUTED_HISTORY_SWITCH'
            elif set(new_history) == {gt} and len(old_history) > 1: category = 'CORRECTIVE_SWITCH'
            elif new is None: category = 'FALSE_BIRTH_OR_NATIVE_RECOVERY'
            else: category = 'AMBIGUOUS_UNKNOWN'
            if selections[category] >= 2 or len(selected) >= 6: continue
            if any(tuple(e['key'][1:]) == key for e in selected): continue
            query = {'task': 0, 'refs': refs.copy(), 'legal': d['batch']['legal'][0, row].cpu().tolist(),
                     'option_scores': d['logits'][row].cpu().tolist(),
                     'candidate_history_GT_OFFLINE_ONLY': {str(t): dict(risk.votes[t]) for t in refs}}
            previous_legal = old in refs and query['legal'][refs.index(old)]
            event = {'key': [video, *key], 'row': row, 'category': category, 'previous_native_id': old,
                     'native_id': new, 'GT_OFFLINE_ONLY': gt, 'queries': [query],
                     'previous_ID_in_legal_MATCH_set': bool(previous_legal),
                     'pure_previous_continuation_candidate': set(old_history) == {gt} and bool(previous_legal)}
            storage_guard(); path = folder / f'PREFIX_{key[0]:06d}_{key[1]}.pth.xz'
            state = phase13_prefix(model, current_native[0]); digest = fingerprint(state)
            save_dense(path, state, reserve=30 * 2**30)
            selected.append({k: v for k, v in event.items() if k != 'queries'})
            selected[-1].update(prefix=ref(path), starting_state_SHA256=digest,
                actual_frozen_policy='pi_multi_seed20261009_20k', before_current_decision=True, GT_or_future_in_state=False)
            event_records.append(event); pending[key, row] = event
            selections[category] += 1
            print('PHASE15_TRAIN_PREFIX', video, key, row, category, flush=True)

    def after(**d):
        risk.after(**d); key = (d['frame'], d['view']); ids = d['instances'][-1].track_ids.cpu().tolist()
        assert len(ids) == len(set(ids))
        for row, (identity, action) in enumerate(zip(ids, d['events'])):
            event = pending.get((key, row))
            if event is not None:
                event['native_id'] = identity; event['actual_native_action'] = action['action']
                for item in selected:
                    if tuple(item['key'][1:]) == key and item['row'] == row:
                        item['native_id'] = identity; item['actual_native_action'] = action['action']
        journal.write(json.dumps({'key': [video, *key], 'ids': ids, 'events': d['events'], 'id_count': d['id_count']}) + '\n')
        if time.monotonic() - last_progress[0] >= 15:
            save(folder / 'PROGRESS.json', {'status': 'RUNNING', 'key': [video, *key], 'frames': frames,
                 'prefixes': len(selected), 'source_commit': source['source_commit'], 'seconds': time.monotonic() - begin})
            last_progress[0] = time.monotonic()
    executor.observer = before; executor.commit_observer = after; model.jev_native_prefix_observer = before_native
    try:
        with torch.no_grad(): raw, _ = run(model, values, frames)
    finally: journal.close()
    assert risk.counts['payloads'] == 2 * frames - 1
    predictions = raw_predictions(raw, risk.labels.images); save(folder / 'RAW_PREDICTIONS.json', predictions)
    # GT evaluators run only after all predictions have been committed/frozen.
    strict, evaluator = metrics(folder / 'RAW_PREDICTIONS.json', [video], folder / 'strict_eval')
    with gzip.open(folder / 'SWITCH_EVENTS.jsonl.gz', 'wt') as stream:
        for event in event_records: stream.write(json.dumps(event, allow_nan=False) + '\n')
    save(folder / 'RESULT.json', {'status': 'PASS', 'binding': source, 'video': video, 'frames': frames,
        'strict_online_metrics': strict, 'raw_predictions': ref(folder / 'RAW_PREDICTIONS.json'),
        'real_evaluator': ref(evaluator / 'metrics.json'), 'training_result': training_result,
        'natural_native_risk': risk.summary(), 'counterfactual_prefixes': selected,
        'selected_event_ledger': ref(folder / 'SWITCH_EVENTS.jsonl.gz'),
        'actual_mutated_state': True, 'teacher_forced_IDs': False, 'GT_actor_inputs': False,
        'selected_prefix_counts': dict(selections), 'new_optimizer_updates': 0,
        'selection_uses_future_utility': False, 'seconds': time.monotonic() - begin,
        'no_Commitment_learnability_or_quality_claim': True})
    save(folder / 'PROGRESS.json', {'status': 'PASS', 'prefixes': len(selected), 'frames': frames})
    print('PHASE15_TRAIN_PREFIX_COLLECTION_PASS', video, len(selected), strict, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--video', type=int, required=True); args = p.parse_args()
    try: main(args.video)
    except Exception: failure('train_prefix_collection', traceback.format_exc()); raise
