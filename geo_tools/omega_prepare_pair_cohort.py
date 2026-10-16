"""Freeze real pairs before any Omega metric; reuse sealed audits, never rerun GMT.

No detector/GT inference occurs here. Hard events use existing offline TRAIN
labels only. The inference cohort is written separately from these labels.
"""
import argparse
import collections
import json
from pathlib import Path
from PIL import Image
from omega_common import DATA, FIT, ROOT, jsonl, provenance, save_new, sha256, verified_native


def analyze_scene(scene, metadata):
    directory, receipt = verified_native(scene)
    events = {tuple(e['key'][1:]): e for e in jsonl(directory / 'events.jsonl.gz')}
    counters = collections.Counter()
    errors = []
    seen = set()
    for call in jsonl(directory / 'actual_decisions.jsonl.gz'):
        if call['stage'] == 'run_memory_tracker':
            continue
        for source in call['queries']:
            key = tuple(source)
            if key in seen:
                raise ValueError('duplicate primary query')
            seen.add(key)
            e = events[key]
            q = metadata[source[0]]
            assert e['frame'] == q['frame_id'] and e['camera'] == q['view_id']
            counters['all_primary_queries'] += 1
            counters['known_wrong_or_missed'] += bool(e['known_wrong'])
            if not e['known_wrong']:
                continue
            current_refs = []
            for ref in call['references']:
                r = metadata[ref['source'][0]]
                if r['video_id'] != scene or r['frame_id'] > q['frame_id']:
                    raise ValueError('invalid/future native reference')
                if r['frame_id'] == q['frame_id'] and r['view_id'] != q['view_id']:
                    current_refs.append(ref)
            sync_tracks = {r['track'] for r in current_refs}
            sync_positive = sync_tracks.intersection(e['positive_tracks'])
            counters['errors_with_any_synchronous_cross_camera_reference'] += bool(current_refs)
            counters['errors_with_synchronous_pure_positive'] += bool(sync_positive)
            counters['errors_with_synchronous_pure_positive_in_top8'] += bool(sync_positive.intersection(e['top8_tracks']))
            counters['errors_without_any_synchronous_cross_camera_reference'] += not bool(current_refs)
            counters['errors_without_synchronous_pure_positive'] += not bool(sync_positive)
            errors.append(dict(key=e['key'], frame=e['frame'], camera=e['camera'],
                               query_gt_identity=e['query_gt_identity'],
                               actual_primary_state=e['actual_primary_state'],
                               accepted_primary_track=e['accepted_primary_track'],
                               final_output_track=e['final_output_track'],
                               positive_tracks=e['positive_tracks'], top8_tracks=e['top8_tracks'],
                               synchronous_references=current_refs,
                               synchronous_pure_positive_tracks=sorted(sync_positive),
                               native_scope_status='SYNCHRONOUS_REFERENCE_AVAILABLE' if current_refs else 'GEOMETRY_NOT_APPLICABLE',
                               positive_comparison_status='SYNCHRONOUS_PURE_POSITIVE_AVAILABLE' if sync_positive else 'NO_CERTIFIED_SYNCHRONOUS_PURE_POSITIVE',
                               geometry_reliability='NOT_MEASURED',
                               geometry_candidate_discrimination='NOT_EXECUTED'))
    if seen != set(events):
        raise ValueError('event/call membership mismatch')
    summary = json.loads((directory / 'summary.json').read_text())['counts']
    assert counters['all_primary_queries'] == summary['N_total']
    assert counters['known_wrong_or_missed'] == summary['N_wrong']
    # Explicit zeros only concern actually counted native metadata, never Omega measurements.
    names = ['all_primary_queries', 'known_wrong_or_missed',
             'errors_with_any_synchronous_cross_camera_reference', 'errors_with_synchronous_pure_positive',
             'errors_with_synchronous_pure_positive_in_top8',
             'errors_without_any_synchronous_cross_camera_reference', 'errors_without_synchronous_pure_positive']
    return dict(scene=scene, counts={k: counters[k] for k in names}, native_source=receipt), errors


def choose_hard(errors, existing):
    # One representative per scene/camera/person first; deterministic time spacing
    # of >=30 synchronized groups from every selected hard event for that person.
    by_scene = collections.defaultdict(lambda: collections.defaultdict(list))
    for e in errors:
        if e['key'][0] in FIT and e['synchronous_pure_positive_tracks']:
            by_scene[e['key'][0]][e['query_gt_identity']].append(e)
    queues = {}
    for scene, groups in by_scene.items():
        picked = []
        for identity, rows in sorted(groups.items()):
            prior = []
            for e in sorted(rows, key=lambda x: (x['frame'], x['key'])):
                if all(abs(e['frame'] - f) >= 30 for f in prior):
                    prior.append(e['frame'])
                    picked.append(e)
        # Round-robin people before additional episodes of a person.
        rounds = collections.defaultdict(list)
        by_person = collections.defaultdict(list)
        for e in picked:
            by_person[e['query_gt_identity']].append(e)
        for person, rows in sorted(by_person.items()):
            for index, e in enumerate(rows):
                rounds[index].append(e)
        queues[scene] = [e for i in sorted(rounds) for e in rounds[i]]
    selected = []
    while len(selected) < 30 and any(queues.values()):
        for scene in FIT:
            if len(selected) >= 30:
                break
            while queues.get(scene):
                e = queues[scene].pop(0)
                pair = (scene, e['frame'])
                if pair not in existing:
                    existing.add(pair)
                    selected.append(e)
                    break
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'manifests/omega_fixed_pair_cohort_v1.json')
    parser.add_argument('--scope-output', type=Path, default=ROOT / 'results/omega_native_synchronous_scope_v1.json')
    args = parser.parse_args()
    annotation = DATA / 'annotations/train.json'
    assert sha256(annotation) == '9093c36204bae482c2f74997f531bfe2d8a46464f4541093703ef22db0068c85'
    data = json.loads(annotation.read_text())
    metadata = {x['id']: x for x in data['images']}
    reports, errors = [], []
    for scene in (*FIT, 12):
        report, rows = analyze_scene(scene, metadata)
        reports.append(report)
        errors.extend(rows)
        print(json.dumps(dict(scene=scene, **report['counts'])), flush=True)
    # Ordered first24 are all GT-independent uniform times across every fit scene.
    # Expansion is fixed now but remains conditional on the first24 interface gate.
    groups = collections.defaultdict(dict)
    for r in data['images']:
        if r['video_id'] in FIT:
            groups[r['video_id']].setdefault(r['frame_id'], {})[r['view_id']] = r
    primary, uniform_extra = [], []
    for scene in FIT:
        frames = sorted(groups[scene])
        indices = [round((len(frames)-1)*i/5) for i in range(6)]
        chosen = [frames[i] for i in indices]
        primary.extend((scene, f) for f in [chosen[0], chosen[-1]])
        uniform_extra.extend((scene, f) for f in chosen[1:-1])
    for scene in (4, 24):
        item = (scene, sorted(groups[scene])[round((len(groups[scene])-1)*2/5)])
        primary.append(item)
        uniform_extra.remove(item)
    uniform_keys = primary + uniform_extra
    existing = set(uniform_keys)
    hard = choose_hard(errors, existing)
    hard_keys = [(e['key'][0], e['frame']) for e in hard]
    pairs = []
    for i, (scene, frame) in enumerate(uniform_keys + hard_keys):
        records = groups[scene][frame]
        if sorted(records) != [1, 2]:
            raise ValueError('expected exactly two physical cameras')
        images = []
        for cam, row in sorted(records.items()):
            path = DATA / 'train' / row['file_name']
            with Image.open(path) as im:
                if im.size != (row['width'], row['height']):
                    raise ValueError('actual/annotation image size mismatch')
            images.append(dict(camera_id=cam, image_id=row['id'], path=str(path),
                               sha256=sha256(path), original_width=row['width'], original_height=row['height']))
        pairs.append(dict(pair_id=f'scene{scene}_group{frame}', scene=scene, group=frame,
                          stage='initial24' if i < 24 else 'conditional_expansion',
                          sampling='uniform_without_GT' if i < len(uniform_keys) else 'TRAIN_hard_diagnostic',
                          images=images))
    assert len(primary) == 24 and len(uniform_keys) == 66 and len(pairs) <= 96
    cohort = dict(schema='omega-fixed-pair-cohort/v1', status='FROZEN_BEFORE_OMEGA_METRICS',
                  fit_scenes=list(FIT), diagnostic_scene12='NATIVE_SCOPE_ONLY_NO_OMEGA_PAIR_SELECTION',
                  strict_DEV5='NOT_SELECTED_NOT_USED', official_TEST='NOT_SELECTED_NOT_USED',
                  annotation=dict(path=str(annotation), sha256=sha256(annotation)),
                  first_stage_pairs=24, uniform_pairs=66, hard_pairs=len(hard), total_pairs=len(pairs),
                  selection='six uniform frame-index quantiles per fit scene; first24=two endpoints/scene plus fixed .4 quantiles of4/24; up to30 synchronous-pure-positive hard TRAIN pairs scene/person round-robin with30-group within-person spacing',
                  correlation_unit='scene/GT-person and30-group-spaced hard episodes across both cameras; uniform samples can still correlate',
                  expansion_gate='real Omega first24 interface and reliable background evidence required',
                  pair_input_contains_GT=False, pairs=pairs,
                  native_source_receipts=[r['native_source'] for r in reports if r['scene'] in FIT],
                  provenance=provenance())
    counts = {k: sum(r['counts'][k] for r in reports) for k in reports[0]['counts']}
    scope = dict(schema='omega-native-synchronous-scope/v1', status='COMPLETE_NATIVE_METADATA_REUSE',
                 scope='Existing primary solver references only. A same-time reference is necessary, not proof of geometric validity or correction. Historical-only pure positives remain in the denominator.',
                 no_GMT_inference=True, no_new_GT_matching=True, no_Omega_inference=True,
                 all12_counts=counts,
                 fit11_counts={k: sum(r['counts'][k] for r in reports if r['scene'] != 12) for k in counts},
                 scene_reports=reports, errors=errors, selected_hard_offline_labels=hard,
                 geometry_reliable_error_count='NOT_MEASURED', actual_Omega_corrections='NOT_EXECUTED',
                 provenance=provenance())
    save_new(args.output, cohort)
    save_new(args.scope_output, scope)
    save_new(ROOT / 'results/omega_pair_cohort.json', dict(status=cohort['status'], manifest=str(args.output),
               sha256=sha256(args.output), first_stage_pairs=24, total_pairs=len(pairs),
               uniform_pairs=66, hard_pairs=len(hard), source_scope_sha256=sha256(args.scope_output)))
    print(json.dumps(dict(cohort=str(args.output), sha256=sha256(args.output), pairs=len(pairs))))


if __name__ == '__main__':
    main()
