"""Frozen TRAIN Rule commits versus raw GMT scores on the same live state."""
import argparse
import collections


def paired_commit_counts(sample):
    counts = collections.Counter(rows=len(sample['native_ids']))
    for row, (actual, target, person, baseline_col) in enumerate(zip(
            sample['native_ids'], sample['targets'].tolist(),
            sample['GT_labels_OFFLINE_ONLY'], sample['baseline_accepted_cols'])):
        if baseline_col >= 0 and actual != sample['refs'][baseline_col]:
            counts['changed_accepted_reference_commits'] += 1
        if target < 0:
            counts['UNKNOWN_target'] += 1
            continue
        target_id = sample['refs'][target]
        counts['unique_active_rows'] += 1
        counts['actual_unique_target_commits'] += actual == target_id
        counts['eligible_unique_rows'] += bool(sample['eligible'][row].any())
        if baseline_col < 0:
            counts['UNKNOWN_reference_Bank_or_birth'] += 1
            continue
        baseline_id = sample['refs'][baseline_col]
        if sample['owners'].get(baseline_id) is None:
            counts['UNKNOWN_reference_owner'] += 1
            continue
        baseline_correct = baseline_id == target_id
        counts['certified_accepted_reference_rows'] += 1
        counts['reference_correct'] += baseline_correct
        counts['reference_wrong_person'] += not baseline_correct
        if baseline_correct and actual == target_id:
            counts['correct_reference_preserved'] += 1
        elif not baseline_correct and sample['owners'][baseline_id] != person and actual == target_id:
            counts['N01'] += 1
        elif baseline_correct and actual > sample['id_count_before']:
            counts['N10'] += 1
            counts['N10_new_identity_fragment'] += 1
        elif baseline_correct and sample['owners'].get(actual) is not None and sample['owners'][actual] != person:
            counts['N10'] += 1
            counts['N10_wrong_certified_person'] += 1
        elif actual != baseline_id:
            counts['UNKNOWN_changed_commit_owner_or_alias'] += 1
    return counts


def main(video):
    from gmt_jev_runtime import metadata, REPORTS, read, save, ref
    from gmt_jev_opportunity import collect
    assert video in [12, 13, 14]
    coverage_path = REPORTS / 'FULL_TRAIN_OPPORTUNITY_AUDIT.json'
    coverage = read(coverage_path)
    assert coverage['status'] == 'COMPLETE_ALL_REGISTERED_TRAIN_TEMPORAL_COVERAGE'
    assert [case['video'] for case in coverage['cases']] == [12, 13, 14]
    train = collections.Counter()
    for case in coverage['cases']:
        train.update(case['buckets']['TRAIN_BLOCKS'])
    assert train['rule_local_N01'] > train['rule_local_N10'], 'No positive frozen TRAIN signal'
    _, frames, _ = metadata(video)
    buckets = {key: collections.Counter() for key in ['TRAIN_BLOCKS', 'HELDOUT_TEMPORAL_BLOCKS']}

    def observed(sample):
        key = 'HELDOUT_TEMPORAL_BLOCKS' if sample['audit_block'] else 'TRAIN_BLOCKS'
        buckets[key].update(paired_commit_counts(sample))

    samples, result = collect(video, kind='rule', frames=frames, records=False,
                              report=False, diagnostic_sample_observer=observed)
    assert samples == [] and result['frames'] == frames
    counts = buckets['TRAIN_BLOCKS'] + buckets['HELDOUT_TEMPORAL_BLOCKS']
    assert counts['rows'] == result['counts']['rows']
    result.update(
        status='COMPLETE_FULL_TRAIN_NATIVE_RULE_COMMIT_VALIDATION',
        paired_commit_counts=dict(counts),
        paired_commit_buckets={key: dict(value) for key, value in buckets.items()},
        prerequisite_coverage=ref(coverage_path),
        no_gradient_updates=True, no_persistent_corpus=True,
        training_authorized_by_this_report_alone=False,
        counter_interpretation='Native Rule commits versus accepted raw-GMT MATCH on the identical current Rule-mutated state. Original rejected MATCH has UNKNOWN Bank/birth outcome. Certified correct MATCH changed to a newborn is an identity-fragment N10. No full-plan causal claim, cross-run numeric-ID comparison, HOTA claim, or independent validation.',
        legacy_collect_counts_interpretation='The existing collect counts describe the actually committed Rule state; GMT_correct naming does not mean an independently running original baseline. Use paired_commit_counts for this audit.')
    save(REPORTS / f'FULL_TRAIN_NATIVE_RULE_VIDEO{video:02d}.json', result)
    print('TRAIN_NATIVE_RULE_VALIDATION_COMPLETE', video, dict(counts), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=int, choices=[12, 13, 14], required=True)
    main(parser.parse_args().video)
