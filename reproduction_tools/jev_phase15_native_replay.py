"""Replay frozen native actors and recover real switch inputs without mutation."""
import argparse
import collections
import gzip
import time
import traceback
import torch
from jev_phase15_common import *
from jev_phase13_runtime import build_tracker, cache_inputs, run
from jev_phase14_artifacts import save_dense
from audit_jev_stage2_gta_free import phase13_prefix
from gtr.modeling.jev_phase14.memory import CachedIdentityMemory
from gtr.modeling.jev_native_state import fingerprint


def gzip_rows(path):
    with gzip.open(path, 'rt') as stream:
        return [json.loads(line) for line in stream]


def main(variant, seed, video):
    protect(); storage_guard(); assert video in DEV
    case = old_case(variant, seed, video); original = read(case / 'RESULT.json')
    for rel, digest in original['binding']['source_SHA256'].items():
        if rel.startswith('gtr/'):
            assert sha(ROOT / rel) == digest, ('historical actor computation changed', rel)
    old_questions = collections.defaultdict(list)
    for q in gzip_rows(case / 'QUESTIONS.jsonl.gz'):
        old_questions[tuple(q['key'][1:])].append(q)
    old_commits = {tuple(q['key'][1:]): q for q in gzip_rows(case / 'COMMITS.jsonl.gz')}
    reconstruction = read(OUT / 'failure_reconstruction_v1' / f'{variant}_seed{seed}' / f'video{video:02d}' / 'RESULT.json')
    switches = collections.defaultdict(list)
    for e in gzip_rows(reconstruction['event_stream']['path']): switches[tuple(e['key'][1:])].append(e)
    selected = {tuple(e['key'][1:]): e for e in reconstruction['selected_counterfactual_prefixes']}
    # Keep native full prefixes only for the predeclared worst-seed interventions.
    if variant != 'multi_question' or seed != 20261009: selected = {}
    folder = OUT / 'native_replay_v1' / f'{variant}_seed{seed}' / f'video{video:02d}'
    folder.mkdir(parents=True, exist_ok=True)
    assert not (folder / 'RESULT.json').exists(), 'completed native replay is immutable'
    torch.set_num_threads(1); torch.manual_seed(20261009)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    policy, checkpoint, training_result = load_old_policy(variant, seed)
    source = binding(seed=seed, checkpoints=[checkpoint], dataset=ref(ANNOTATIONS),
                     evaluator='exact native replay versus saved per-question and per-commit journals',
                     scope='complete development video; frozen Phase14 actor; no training')
    source['inference_torch_seed'] = 20261009
    values, frames, reader = cache_inputs(video); assert frames == original['frames']
    model = build_tracker(video, policy=policy, variant='full', temperature=(1., 1.), react_learned=False)
    executor = model.jev_stage2_executor; executor.memory_factory = CachedIdentityMemory
    counts = collections.Counter(); max_logit_difference = [0.]; final = [None]
    prefix_refs = []; details = []; observed_questions = collections.Counter(); current_native = [None]
    journal = gzip.open(folder / 'SWITCH_REPLAY.jsonl.gz', 'wt'); begin = time.monotonic(); last_progress = [0.]

    def before_native(**d):
        key = (d['frame'], d['view']); current_native[0] = d
        if key not in selected: return
        storage_guard(); path = folder / f'PREFIX_{key[0]:06d}_{key[1]}.pth.xz'
        native = phase13_prefix(model, d)
        digest = fingerprint(native)
        if not path.exists(): save_dense(path, native, reserve=30 * 2**30)
        prefix_refs.append(dict(selected[key], prefix=ref(path), starting_state_SHA256=digest,
            actual_frozen_policy=f'pi_multi_seed{seed}_20k', before_current_decision=True,
            GT_or_future_in_state=False))

    def question(**d):
        c = d['context']; key = (c['frame'], c['view']); task = d['task']
        if not len(d['logits']): return
        index = observed_questions[key]; observed_questions[key] += 1
        q = old_questions[key][index]; rows = c.get('rows', list(range(len(d['logits']))))
        assert q['task'] == task and q['refs'] == d['refs'] and q['rows'] == rows
        assert q['legal'] == d['batch']['legal'][0].cpu().tolist()
        expected = torch.tensor(q['logits'], device=d['logits'].device, dtype=d['logits'].dtype)
        delta = float((d['logits'] - expected).abs().max())
        max_logit_difference[0] = max(max_logit_difference[0], delta)
        assert delta == 0., ('saved native logit mismatch', key, task, delta)
        counts['questions_exact'] += 1
        if task != 0 or key not in switches: return
        with torch.no_grad(): output = policy.details(d['batch'])
        assert torch.equal(output['logits'][0], d['logits'])
        native = current_native[0]; assert (native['frame'], native['view']) == key
        from gtr.modeling.meta_arch.gtr_rcnn import poss_ids, old_reids
        state = {'identity_memory': executor.memory.state_dict(),
                 'possible_ids': sorted(poss_ids.poss_ids),
                 'bank_ids': old_reids.old_reids[0].track_ids.cpu().tolist() if old_reids.old_reids else [],
                 'id_count': native['id_count'], 'hits': dict(native['hits'])}
        row_events = []
        for e in switches[key]:
            qi = rows.index(e['row'])
            record = {'key': [video, *key], 'row': e['row'], 'category_OFFLINE_ONLY': e['category'],
                'refs': list(d['refs']), 'legal': d['batch']['legal'][0, qi].cpu().tolist(),
                'WHO_logits': output['choice_logits'][0, qi].cpu().tolist(),
                'Availability_logit': float(output['availability_logits'][0, qi]),
                'Trust_logits': output['trust_logits'][0, qi].cpu().tolist(),
                'final_option_scores': d['logits'][qi].cpu().tolist(),
                'previous_native_ID_OFFLINE_ANALYSIS_ONLY': e['previous_native_id'],
                'final_saved_committed_ID': e['native_id'],
                'memory_and_bank_state_SHA256': fingerprint(state),
                'causal_tensor_record_index': len(details),
                'Gallery_lengths': {str(t): len(native['galleries'].get(t, [])) for t in d['refs']},
                'identity_history_metadata': {str(t): executor.memory.meta.get(t) for t in d['refs']},
                'bank_IDs': state['bank_ids'], 'future_information_in_actor_inputs': False}
            journal.write(json.dumps(record, allow_nan=False) + '\n'); row_events.append(record)
            counts['switch_rows_with_WHO_Availability_Trust'] += 1
        # Full numeric visual tokens, masks, metadata and option evidence reside
        # in the lossless server artifact; no GT tensor enters these inputs.
        details.append({'key': [video, *key], 'rows': rows, 'refs': list(d['refs']),
                        'inputs': {k: v.detach().cpu().clone() for k, v in d['batch'].items()},
                        'switch_rows': [e['row'] for e in switches[key]], 'row_evidence': row_events})

    def after(**d):
        from gtr.modeling.meta_arch.gtr_rcnn import poss_ids, old_reids
        key = (d['frame'], d['view']); current = d['instances'][-1]; ids = current.track_ids.cpu().tolist()
        old = old_commits[key]
        assert ids == old['ids'] and d['id_count'] == old['id_count'] and d['events'] == old['events'], ('commit mismatch', key)
        assert {str(t): int(d['hits'][t]) for t in ids} == old['hits']
        assert {str(t): len(d['galleries'][t]) for t in ids} == old['Gallery_lengths']
        assert sorted(poss_ids.poss_ids) == old['possible_ids']
        bank = old_reids.old_reids[0].track_ids.cpu().tolist() if old_reids.old_reids else []
        assert bank == old['old_reids_IDs']
        assert len(model._jev_trajectory_rng._trajectory_draws) == old['trajectory_RNG_draws']
        assert len(ids) == len(set(ids)); counts['native_payloads_exact'] += 1; counts['native_detections_exact'] += len(ids)
        if key == (frames - 1, 1):
            final[0] = fingerprint(dict(instances=d['instances'], galleries=d['galleries'], hits=d['hits'],
                id_count=d['id_count'], possible_ids=poss_ids.poss_ids, old_reids=old_reids.old_reids,
                memory=executor.memory.state_dict(), RNG=model._jev_trajectory_rng.getstate(),
                draws=model._jev_trajectory_rng._trajectory_draws))
        if time.monotonic() - last_progress[0] >= 15:
            save(folder / 'PROGRESS.json', {'status': 'RUNNING', 'key': [video, *key], 'frames': frames,
                 'counts': dict(counts), 'source_commit': source['source_commit'], 'seconds': time.monotonic() - begin})
            last_progress[0] = time.monotonic()
            print('PHASE15_REPLAY_PROGRESS', variant, seed, video, key, dict(counts), flush=True)

    executor.observer = question; executor.commit_observer = after; model.jev_native_prefix_observer = before_native
    try:
        with torch.no_grad(): raw, _ = run(model, values, frames)
    finally: journal.close()
    assert counts['native_payloads_exact'] == 2 * frames - 1
    assert counts['switch_rows_with_WHO_Availability_Trust'] == reconstruction['exact_CLEAR_switches']
    assert final[0] == original['final_complete_native_state_SHA256'], 'final native state differs'
    for key, qs in old_questions.items(): assert observed_questions[key] == len(qs), ('missing query', key)
    storage_guard(); tensor_path = folder / 'CAUSAL_SWITCH_INPUTS.pth.xz'
    save_dense(tensor_path, {'records': details, 'GT_input_tensors': False, 'source': source}, reserve=30 * 2**30)
    result = {'status': 'PASS', 'binding': source, 'variant': variant, 'seed': seed, 'video': video,
        'frames': frames, 'counts': dict(counts), 'maximum_saved_vs_replayed_logit_difference': max_logit_difference[0],
        'original_actor_source_commit': original['binding']['source_commit'], 'original_native_result': ref(case / 'RESULT.json'),
        'original_training_result': training_result, 'every_native_ID_and_lifecycle_commit_exact': True,
        'every_saved_option_score_exact': True, 'final_full_native_state_SHA256': final[0],
        'final_state_matches_frozen_Phase14': True, 'causal_visual_inputs': ref(tensor_path),
        'recovered_head_journal': ref(folder / 'SWITCH_REPLAY.jsonl.gz'), 'counterfactual_prefixes': prefix_refs,
        'prefix_selection_frozen_before_future_outcomes': True, 'GT_actor_inputs': False,
        'full_FPS': None, 'seconds_with_logging_and_serialization': time.monotonic() - begin}
    save(folder / 'RESULT.json', result); save(folder / 'PROGRESS.json', {'status': 'PASS', 'counts': dict(counts)})
    print('PHASE15_NATIVE_REPLAY_PASS', variant, seed, video, reconstruction['exact_CLEAR_switches'], flush=True)


if __name__ == '__main__':
    a = argparse.ArgumentParser(); a.add_argument('--variant', required=True)
    a.add_argument('--seed', type=int, required=True); a.add_argument('--video', type=int, required=True)
    args = a.parse_args()
    try: main(args.variant, args.seed, args.video)
    except Exception:
        failure('native_replay', traceback.format_exc()); raise
