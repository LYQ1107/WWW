"""Reconstruct every saved CLEAR switch; separate history risk from necessity.

All annotations below are offline diagnostics. They are never policy inputs.
The unmodified TrackEval preprocessing and continuation-priority CLEAR matcher
determine events. Saved logits must reproduce the actual native assignment.
"""
import argparse
import collections
import gzip
import time
import numpy as np
from scipy.optimize import linear_sum_assignment
from jev_phase14_forensics import offline_alignment, native_choice, load_gzip, iou
from jev_phase15_common import *

CATEGORIES = ['PURE_FRAGMENT_HOP', 'POLLUTED_HISTORY_SWITCH', 'CORRECTIVE_SWITCH',
              'WRONG_EXISTING_MERGE', 'FALSE_BIRTH', 'RECOVERY_SWITCH', 'AMBIGUOUS_UNKNOWN']


def exact_clear(out, aligned, annotations, video):
    sys.path.insert(0, str(ROOT / 'TrackEval'))
    for name, kind in [('float', float), ('int', int), ('bool', bool)]:
        if name not in np.__dict__: setattr(np, name, kind)
    import trackeval
    manifest = read(out / 'strict_eval/tracking_eval_runtime_state/native/prepared/manifest.json')
    cfg = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    cfg.update(GT_FOLDER=manifest['trackeval_gt'], TRACKERS_FOLDER=manifest['trackeval_trackers'],
               TRACKERS_TO_EVAL=['GMT'], BENCHMARK='VisionTrack', SPLIT_TO_EVAL='test',
               SKIP_SPLIT_FOL=True, SEQ_INFO=manifest['seq_lengths'], PRINT_CONFIG=False)
    dataset = trackeval.datasets.MotChallenge2DBox(cfg)
    images = {(im['frame_id'] - 1, im['view_id'] - 1): im for im in annotations['images'] if im['video_id'] == video}
    byimage = collections.defaultdict(list)
    image_ids = {im['id'] for im in images.values()}
    for a in annotations['annotations']:
        if a['image_id'] in image_ids: byimage[a['image_id']].append(a)
    events = []; counts = []
    for sequence in manifest['sequences']:
        view = int(sequence.rsplit('View', 1)[1]) - 1
        original = dataset.get_raw_seq_data('GMT', sequence)
        native_ids = [a.copy() for a in original['tracker_ids']]
        native_boxes = [a.copy() for a in original['tracker_dets']]
        data = dataset.get_preprocessed_seq_data(original, 'pedestrian')
        official = trackeval.metrics.CLEAR({'PRINT_CONFIG': False}).eval_sequence(data)
        previous = np.full(data['num_gt_ids'], np.nan); previous_step = previous.copy()
        last_native = {}; last_frame = {}; last_gt_consistency = {}; start = len(events)
        for frame, (gs, ts, similarities) in enumerate(zip(data['gt_ids'], data['tracker_ids'], data['similarity_scores'])):
            if not len(gs) or not len(ts): continue
            score = 1000 * (ts[None, :] == previous_step[gs[:, None]]) + similarities
            score[similarities < .5 - np.finfo(float).eps] = 0
            rr, cc = linear_sum_assignment(-score)
            valid = score[rr, cc] > np.finfo(float).eps; rr = rr[valid]; cc = cc[valid]
            ps, diagnostic_gt = aligned[frame, view]
            targets = byimage[images[frame, view]['id']]
            for r, c in zip(rr, cc):
                match = np.all(np.isclose(native_boxes[frame], data['tracker_dets'][frame][c], rtol=0, atol=1e-9), axis=1)
                found = native_ids[frame][match]
                assert len(found) == 1, (sequence, frame, found)
                native = int(found[0]); rows = [j for j, p in enumerate(ps) if p['track_id'] == native]
                assert len(rows) == 1
                row = rows[0]; clear_gt = None
                if targets:
                    overlap = iou([data['gt_dets'][frame][r]], [a['bbox'] for a in targets])[0]
                    identity = np.flatnonzero(overlap >= .999)
                    if len(identity) == 1: clear_gt = int(targets[int(identity[0])]['instance_id'])
                consistent = clear_gt is not None and clear_gt == diagnostic_gt[row]
                gt_key = int(gs[r])
                if not np.isnan(previous[gt_key]) and previous[gt_key] != ts[c]:
                    events.append({'frame': frame, 'view': view, 'row': row, 'native_id': native,
                                   'previous_native_id': last_native[gt_key], 'previous_CLEAR_frame': last_frame[gt_key],
                                   'CLEAR_GT_relabelled': gt_key, 'GT_OFFLINE_ONLY': clear_gt,
                                   'diagnostic_alignment_GT_OFFLINE_ONLY': diagnostic_gt[row],
                                   'current_alignment_agrees_with_CLEAR': consistent,
                                   'previous_alignment_agreed_with_CLEAR': last_gt_consistency[gt_key]})
                last_native[gt_key] = native; last_frame[gt_key] = frame; last_gt_consistency[gt_key] = consistent
            previous[gs[rr]] = ts[cc]; previous_step[:] = np.nan; previous_step[gs[rr]] = ts[cc]
        assert len(events) - start == official['IDSW'], (sequence, len(events) - start, official['IDSW'])
        counts.append({'sequence': sequence, 'reconstructed_IDSW': len(events) - start, 'official_IDSW': int(official['IDSW'])})
    assert len(events) == read(out / 'RESULT.json')['strict_online_metrics']['IDSW']
    return events, counts


def audit_case(variant, seed, video, annotations):
    out = old_case(variant, seed, video); saved = read(out / 'RESULT.json')
    for name, digest in [('RAW_PREDICTIONS.json', saved['raw_predictions']['SHA256']),
                         ('COMMITS.jsonl.gz', saved['commits_SHA256']), ('QUESTIONS.jsonl.gz', saved['questions_SHA256'])]:
        assert sha(out / name) == digest, str(out / name)
    raw = read(out / 'RAW_PREDICTIONS.json'); aligned = offline_alignment(raw, annotations, video)
    switches, clear_counts = exact_clear(out, aligned, annotations, video)
    keys = {(s['frame'], s['view'], s['row']): s for s in switches}
    commits = load_gzip(out / 'COMMITS.jsonl.gz'); queries = collections.defaultdict(list)
    for q in load_gzip(out / 'QUESTIONS.jsonl.gz'): queries[tuple(q['key'][1:])].append(q)
    votes = collections.defaultdict(collections.Counter); seen = set(); trajectory = collections.defaultdict(list)
    for (frame, view), (ps, gts) in sorted(aligned.items()):
        for p, gt in zip(ps, gts):
            if gt is not None: trajectory[gt, view].append([frame, p['track_id']])
    other = 1 - commits[0]['key'][2]; initial, initial_gt = aligned[0, other]
    for p, gt in zip(initial, initial_gt):
        if gt is not None: votes[p['track_id']][gt] += 1; seen.add(gt)
    events = []; counts = collections.Counter(); examples = collections.defaultdict(list)
    for commit in commits:
        frame, view = commit['key'][1:]; ps, gts = aligned[frame, view]
        assert commit['ids'] == [p['track_id'] for p in ps]
        row_queries = collections.defaultdict(list)
        for q in queries[frame, view]:
            choice = native_choice(q['logits'], q['legal'])
            for qi, (row, selected) in enumerate(zip(q['rows'], choice)):
                expected_action = 'ASSOCIATE_EXISTING' if q['task'] == 0 else 'REACTIVATE'
                actual = commit['events'][row]
                assert (q['refs'][selected] if selected >= 0 else -1) == (actual['identity'] if actual['action'] == expected_action else -1)
                row_queries[row].append({'task': q['task'], 'refs': q['refs'], 'legal': q['legal'][qi],
                    'option_scores': q['logits'][qi], 'selected_column': selected,
                    'candidate_history_GT_OFFLINE_ONLY': {str(t): dict(votes[t]) for t in q['refs']}})
        # Snapshot classifications before any current-camera row writes GT votes.
        for row, (native, gt, action) in enumerate(zip(commit['ids'], gts, commit['events'])):
            switch = keys.get((frame, view, row))
            if switch is None: continue
            previous = switch['previous_native_id']; current_history = dict(votes[native]); previous_history = dict(votes[previous])
            certified = switch['current_alignment_agrees_with_CLEAR'] and switch['previous_alignment_agreed_with_CLEAR']
            gt = switch['GT_OFFLINE_ONLY']; pure_new = gt is not None and set(current_history) == {gt}
            pure_previous = gt is not None and set(previous_history) == {gt}
            new_other_person = bool(current_history) and gt is not None and gt not in current_history
            old_polluted = len(previous_history) > 1; new_polluted = len(current_history) > 1
            tags = []
            if pure_new and pure_previous: tags.append('BOTH_PURE_SAME_GT')
            if old_polluted: tags.append('PREVIOUS_HISTORY_ALREADY_POLLUTED')
            if new_polluted: tags.append('SELECTED_HISTORY_ALREADY_POLLUTED')
            if new_other_person: tags.append('FIRST_CURRENT_GT_INTRODUCTION_IN_SELECTED_HISTORY')
            if not certified: category = 'AMBIGUOUS_UNKNOWN'
            elif action['action'] == 'START_NEW': category = 'FALSE_BIRTH' if gt in seen else 'AMBIGUOUS_UNKNOWN'
            elif new_other_person: category = 'WRONG_EXISTING_MERGE'
            elif pure_new and old_polluted: category = 'CORRECTIVE_SWITCH'
            elif action['action'] == 'REACTIVATE': category = 'RECOVERY_SWITCH'
            elif pure_new and pure_previous: category = 'PURE_FRAGMENT_HOP'
            elif new_polluted: category = 'POLLUTED_HISTORY_SWITCH'
            else: category = 'AMBIGUOUS_UNKNOWN'
            first = next((q for q in row_queries[row] if q['task'] == 0), None)
            previous_legal = bool(first and previous in first['refs'] and first['legal'][first['refs'].index(previous)])
            safe = bool(certified and pure_previous and previous_legal)
            future = [x for x in trajectory.get((gt, view), []) if frame <= x[0] <= frame + 32] if gt is not None else []
            event = dict(switch, key=[video, frame, view], variant=variant, seed=seed,
                category=category, overlapping_tags=tags, GT_scope='offline diagnostics only',
                previous_history_GT_OFFLINE_ONLY=previous_history, selected_history_GT_OFFLINE_ONLY=current_history,
                previous_ID_in_legal_MATCH_set=previous_legal, pure_previous_continuation_candidate=safe,
                action=action, final_committed_ID=native, queries=row_queries[row],
                following_same_GT_IDs_OFFLINE_ONLY=future,
                native_bank={'old_reids_IDs': commit['old_reids_IDs'], 'possible_ids': commit['possible_ids']},
                native_hits=commit['hits'], native_Gallery_lengths=commit['Gallery_lengths'],
                WHO_Availability_Trust_and_visual_tokens='PENDING_EXACT_NATIVE_REPLAY',
                corrective_scope='clean candidate replaces demonstrably mixed old history; future benefit/necessity still requires paired native intervention')
            events.append(event); counts[category] += 1
            counts['certified_pure_previous_and_legal'] += int(safe)
            counts['previous_ID_not_legal_MATCH_candidate'] += int(not previous_legal)
            if len(examples[category]) < 3:
                examples[category].append({k: v for k, v in event.items() if k not in ['queries', 'native_hits', 'native_Gallery_lengths', 'following_same_GT_IDs_OFFLINE_ONLY']})
        for native, gt in zip(commit['ids'], gts):
            if gt is not None: votes[native][gt] += 1; seen.add(gt)
    assert len(events) == len(switches) == sum(counts[c] for c in CATEGORIES)
    folder = OUT / 'failure_reconstruction_v1' / f'{variant}_seed{seed}' / f'video{video:02d}'
    folder.mkdir(parents=True, exist_ok=True); path = folder / 'SWITCH_EVENTS.jsonl.gz'
    with gzip.open(path, 'wt') as h:
        for event in events: h.write(json.dumps(event, allow_nan=False) + '\n')
    # Freeze bounded chronological strata before native future outcomes exist.
    selected = []; used = set()
    strata = [lambda e: e['category'] == 'PURE_FRAGMENT_HOP',
              lambda e: e['category'] == 'POLLUTED_HISTORY_SWITCH',
              lambda e: e['category'] == 'CORRECTIVE_SWITCH',
              lambda e: e['category'] in ['FALSE_BIRTH', 'WRONG_EXISTING_MERGE', 'AMBIGUOUS_UNKNOWN']]
    for predicate in strata:
        taken = 0
        for event in events:
            key = tuple(event['key'])
            if not predicate(event) or key in used or taken >= 2: continue
            selected.append({k: event[k] for k in ['key', 'row', 'category', 'previous_native_id', 'native_id',
                              'GT_OFFLINE_ONLY', 'previous_ID_in_legal_MATCH_set', 'pure_previous_continuation_candidate']})
            used.add(key); taken += 1
    summary = {'status': 'COMPLETE_SAVED_CLEAR_RECONSTRUCTION_NATIVE_REPLAY_PENDING',
        'binding': binding(seed=seed, checkpoints=[saved['trained']['checkpoint']], dataset=ref(ANNOTATIONS),
                           evaluator='unmodified TrackEval CLEAR with exact saved preprocessing', scope='complete historical development video; offline only'),
        'variant': variant, 'seed': seed, 'video': video, 'frames': saved['frames'],
        'exact_CLEAR_switches': len(events), 'per_camera_CLEAR': clear_counts, 'counts': dict(counts),
        'event_stream': ref(path), 'saved_native_result': ref(out / 'RESULT.json'),
        'actor_source_commit': saved['binding']['source_commit'], 'selected_counterfactual_prefixes': selected,
        'examples': dict(examples), 'every_CLEAR_IDSW_classified_exactly_once': True,
        'every_saved_option_assignment_matches_native_commit': True,
        'selection_uses_future_utility': False, 'offline_upper_bound_only': True}
    save(folder / 'RESULT.json', summary)
    return summary


def main():
    protect(); assert read(REPORTS / 'PHASE14_FROZEN_EVIDENCE.json')['status'] == 'PASS'
    annotations = read(ANNOTATIONS); cases = []; begin = time.monotonic()
    methods = [('multi_question', 20261009), ('fixed_question', 20261009), ('set_transformer', 20261009),
               ('multi_question', 20261008), ('multi_question', 20261010)]
    for variant, seed in methods:
        for video in DEV:
            p = OUT / 'failure_reconstruction_v1' / f'{variant}_seed{seed}' / f'video{video:02d}' / 'RESULT.json'
            result = read(p) if p.exists() else audit_case(variant, seed, video, annotations)
            assert sha(result['event_stream']['path']) == result['event_stream']['SHA256']
            cases.append(result); print('PHASE15_SWITCH_RECONSTRUCTION', variant, seed, video,
                                      result['exact_CLEAR_switches'], result['counts'], flush=True)
            save(OUT / 'failure_reconstruction_v1/PROGRESS.json', {'status': 'RUNNING', 'completed': len(cases),
                 'total': 15, 'last': [variant, seed, video], 'seconds': time.monotonic() - begin})
    worst = [r for r in cases if r['variant'] == 'multi_question' and r['seed'] == 20261009]
    assert sum(r['exact_CLEAR_switches'] for r in worst) == 824
    counts = collections.Counter()
    for r in worst: counts.update(r['counts'])
    ledger = {'status': 'SAVED_RECONSTRUCTION_PASS_NATIVE_REPLAY_PENDING', 'binding': binding(dataset=ref(ANNOTATIONS)),
              'cases': [{k: v for k, v in r.items() if k not in ['examples', 'binding']} for r in cases],
              'worst_seed_CLEAR_IDSW': 824, 'worst_seed_counts': dict(counts),
              'each_event_in_one_primary_category': True,
              'CORRECTIVE_SWITCH_is_history_certificate_not_proven_future_benefit': True,
              'deployment_GT_or_future_inputs': False, 'long_training_gate': 'PENDING_NATIVE_REPLAY_AND_COUNTERFACTUAL'}
    for name in ['SWITCH_EVENT_LEDGER', 'PHASE15_SWITCH_EVENT_LEDGER']: save(REPORTS / (name + '.json'), ledger)
    atlas = {'status': ledger['status'], 'binding': binding(seed=20261009), 'counts': dict(counts),
             'videos': worst, 'scope': '824 exact CLEAR changes; neither all harmful nor all safely avoidable'}
    for name in ['SEED20261009_FAILURE_ATLAS', 'PHASE15_SEED20261009_FAILURE_ATLAS']: save(REPORTS / (name + '.json'), atlas)
    comparison = {'status': ledger['status'], 'binding': binding(), 'categories': CATEGORIES,
        'definitions': {'CORRECTIVE_SWITCH': 'pure current-GT target replacing old history with proven cross-GT pollution; paired future establishes utility',
                        'PURE_FRAGMENT_HOP': 'both before-current histories pure same offline GT, native existing association',
                        'UNKNOWN': 'uncertified GT, mismatch with CLEAR alignment, or insufficient past evidence'},
        'worst_seed_counts': dict(counts), 'all_IDSW_are_harmful': False,
        'KEEP_is_always_correct': False, 'safe_option_support_is_separate_from_intervention_effect': True}
    save(REPORTS / 'PHASE15_CORRECTIVE_VS_HARMFUL_SWITCH.json', comparison)
    save(OUT / 'failure_reconstruction_v1/PROGRESS.json', {'status': 'SAVED_RECONSTRUCTION_PASS_NATIVE_REPLAY_PENDING', 'completed': 15, 'total': 15})


if __name__ == '__main__':
    try: main()
    except Exception as e: failure('switch_forensics', e); raise
