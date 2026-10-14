"""Offline bounded residual witnesses; never execute GT-derived tracking actions."""
import collections
from gmt_jev_runtime import *
from gmt_jev_opportunity import collect, native_assignment
from gmt_jev_speed_queue import publish_step

CAP = .05


def probe(sample, widen_gate=False):
    scores = sample['raw_scores']
    targets = sample['targets']
    known = targets >= 0
    allowed = torch.ones_like(sample['eligible']) if widen_gate else sample['eligible']
    signs = torch.zeros_like(scores)
    signs[known] = -1
    for row in torch.where(known)[0].tolist():
        signs[row, targets[row]] = 1
    residual = CAP * signs * allowed
    assert bool((residual.abs() <= CAP).all())
    assert not bool(residual[~known].ne(0).any())
    if not widen_gate:
        assert not bool(residual[~sample['eligible']].ne(0).any())
    changed = native_assignment(scores + residual * sample['lengths'][None], sample['lengths'])
    counts = collections.Counter()
    for row, (actual, target, person, base, alternative) in enumerate(zip(
            sample['native_ids'], targets.tolist(), sample['GT_labels_OFFLINE_ONLY'],
            sample['baseline_accepted_cols'], changed)):
        if target < 0:
            continue
        target_id = sample['refs'][target]
        wrong = actual != target_id
        counts['unique_active_rows'] += 1
        counts['E1_original_committed_wrong'] += wrong
        counts['E1_with_any_allowed_edge'] += wrong and bool(allowed[row].any())
        counts['E1_target_edge_allowed'] += wrong and bool(allowed[row, target])
        maximum = scores[row, target] + CAP * allowed[row, target] * sample['lengths'][target]
        threshold = scores.new_tensor(.1 * float(sample['lengths'][target]))
        counts['E1_target_cannot_pass_active_MATCH_threshold_at_max_FP32_score'] += wrong and bool(maximum <= threshold)
        if wrong and alternative == target:
            counts['ORACLE_target_accept_on_E1'] += 1
        if base < 0:
            counts['UNKNOWN_reference_Bank_or_birth'] += 1
            continue
        if alternative == base:
            continue
        if sample['owners'].get(sample['refs'][base]) is None:
            counts['UNKNOWN_reference_owner'] += 1
            continue
        if alternative < 0:
            counts['rejected_accepted_reference_UNKNOWN_Bank_outcome'] += 1
        elif base != target and sample['owners'][sample['refs'][base]] != person and alternative == target:
            counts['ORACLE_N01_accepted_reference'] += 1
        elif base == target and sample['owners'].get(sample['refs'][alternative]) is not None and sample['owners'][sample['refs'][alternative]] != person:
            counts['ORACLE_N10_accepted_reference'] += 1
        else:
            counts['UNKNOWN_changed_owner_or_alias'] += 1
    return counts


def main():
    native = read(REPORTS/'FULL_TRAIN_NATIVE_RULE_AUDIT.json')
    assert native['status'] == 'COMPLETE_ALL_REGISTERED_TRAIN_NATIVE_RULE_VALIDATION'
    assert not native['native_TRAIN_signal_positive']
    reports = []
    for video in [12, 13, 14]:
        _, frames, _ = metadata(video)
        buckets = {kind: {key: collections.Counter() for key in ['TRAIN_BLOCKS', 'HELDOUT_TEMPORAL_BLOCKS']}
                   for kind in ['CURRENT_GATE_ORACLE', 'ALL_LEGAL_DIAGNOSTIC_ORACLE']}

        def observed(sample):
            key = 'HELDOUT_TEMPORAL_BLOCKS' if sample['audit_block'] else 'TRAIN_BLOCKS'
            for kind in buckets:
                buckets[kind][key].update(probe(sample, kind == 'ALL_LEGAL_DIAGNOSTIC_ORACLE'))

        samples, result = collect(video, frames=frames, records=False, report=False,
                                  diagnostic_sample_observer=observed)
        reference = REPORTS/f'FULL_TRAIN_OPPORTUNITY_VIDEO{video:02d}.json'
        assert samples == [] and result['counts'] == read(reference)['counts']
        result.update(status='COMPLETE_OFFLINE_BOUNDED_RESIDUAL_ORACLE_PROBE',
                      ORACLE='CAPPED_PROBES_ARE_IN_probes; legacy collect counts.ORACLE_correct_unique_rows remains the unconstrained candidate-existence diagnostic, not a capped result',
                      probes={kind: {key: dict(value) for key, value in parts.items()} for kind, parts in buckets.items()},
                      original_observe_counter_parity=ref(reference), residual_cap_average_score=CAP,
                      no_gradient_updates=True, no_persistent_corpus=True, ORACLE_actions_executed=False,
                      interpretation='GT-derived capped perturbations passed only to the same-state assignment diagnostic. Actual GMT observe commits remain untouched. A successful probe is a bounded action-space witness, not a neural representability proof, optimal upper bound, online benefit or HOTA bound. ALL_LEGAL_DIAGNOSTIC_ORACLE removes only risk gating diagnostically and is outside the frozen deployed policy space. Maximum target score failing the native threshold certifies only active-MATCH threshold infeasibility, not Bank recovery impossibility.')
        save(REPORTS/f'BOUNDED_ORACLE_VIDEO{video:02d}.json', result)
        reports.append(result)
        publish_step(f'full TRAIN{video} offline bounded residual oracle probe')
    totals = {kind: {key: collections.Counter() for key in ['TRAIN_BLOCKS', 'HELDOUT_TEMPORAL_BLOCKS']}
              for kind in ['CURRENT_GATE_ORACLE', 'ALL_LEGAL_DIAGNOSTIC_ORACLE']}
    for result in reports:
        for kind, parts in result['probes'].items():
            for key, value in parts.items():
                totals[kind][key].update(value)
    save(REPORTS/'BOUNDED_RESIDUAL_ORACLE_AUDIT.json', dict(
        status='COMPLETE_ALL_REGISTERED_TRAIN_OFFLINE_BOUNDED_ORACLE_PROBES', cases=reports,
        probes={kind: {key: dict(value) for key, value in parts.items()} for kind, parts in totals.items()},
        ORACLE_actions_executed=False, no_gradient_updates=True, independent_validation=False,
        training_authorized_by_this_diagnostic_alone=False, source=binding()))
    publish_step('all registered TRAIN offline bounded residual oracle summary')


if __name__ == '__main__':
    main()
