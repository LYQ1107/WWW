"""Post-decision diagnostic events, with corrections AND regressions.

These are clean-label identity events, never substitutes for TrackEval or the
native VisionTrack MATLAB metrics. Missing outputs are not counted as correct
identity predictions. A gap means an observation gap, not proven occlusion.
"""
from collections import Counter, defaultdict


def decision_events(observations, labels, mapping, prior_people, active_ids):
    """Mapping values are (H, existing_id), (N, virtual_new_id), or None."""
    events, counts = {}, Counter()
    labelled = [(j, o, label, mapping.get(j)) for j, (o, label) in enumerate(zip(observations, labels)) if label is not None]
    births_by_label = defaultdict(set)
    current_new_people = defaultdict(set)
    for j, o, label, token in labelled:
        counts["clean_label_observations"] += 1
        events["output_coverage", o.key] = token is not None
        if token is None:
            counts["suppressed_clean_label_observations"] += 1
            continue
        is_new = token[0] == "N"
        correct_history = [gid for gid in active_ids if prior_people.get(gid, set()) == {label}]
        wrong_reuse = not is_new and prior_people.get(token[1], set()) not in (set(), {label})
        events["wrong_historical_reuse_avoided", o.key] = not wrong_reuse
        if is_new:
            counts["new_ID_observations"] += 1
            births_by_label[label].add(token)
            current_new_people[token].add(label)
        else:
            counts["correct_pure_history_reuse" if prior_people.get(token[1], set()) == {label}
                   else "unknown_history_reuse" if not prior_people.get(token[1], set())
                   else "wrong_or_mixed_history_reuse"] += 1
        if correct_history:
            counts["observations_with_correct_active_history"] += 1
            counts["false_birth_observations"] += int(is_new)
            events["false_birth_avoided", o.key] = not is_new
            events["correct_history_retained", o.key] = token[0] == "H" and token[1] in correct_history
    counts["new_IDs_with_multiple_clean_people"] = sum(len(v) > 1 for v in current_new_people.values())
    counts["within_timestamp_extra_new_IDs_for_same_person"] = sum(max(0, len(v) - 1) for v in births_by_label.values())
    for at, (j, a, la, ga) in enumerate(labelled):
        for k, b, lb, gb in labelled[at + 1:]:
            if a.camera_id == b.camera_id:
                continue
            same = la == lb
            valid = ga is not None and gb is not None
            shared = valid and ga == gb
            kind = "same_person_shared" if same else "different_person_separate"
            counts[kind + "_opportunities"] += 1
            correct = valid and same == shared
            counts[kind + "_correct"] += int(correct)
            events[kind, (a.key, b.key)] = correct
            if same:
                events["duplicate_birth_avoided", (a.key, b.key)] = valid and not (
                    ga[0] == gb[0] == "N" and ga != gb)
    return events, counts


def compare_events(reference, candidate, totals):
    changed = []
    for key in reference.keys() & candidate.keys():
        kind = key[0]
        totals[kind + "/opportunities"] += 1
        a, b = bool(reference[key]), bool(candidate[key])
        totals[kind + "/corrections"] += int(b and not a)
        totals[kind + "/regressions"] += int(a and not b)
        totals[kind + "/net"] += int(b) - int(a)
        if a != b:
            changed.append({"kind": kind, "key": key[1], "correction": b})
    return changed


def plan_mapping(selected, unmatched=(), duplicates=None):
    mapping = {}
    for option in selected:
        token = ("H", int(option.global_id)) if option.global_id is not None else ("N", tuple(option.indices))
        for j in option.indices:
            if j in mapping:
                raise ValueError("COUNTERFACTUAL_DETECTION_ASSIGNED_TWICE")
            mapping[j] = token
    for j in unmatched:
        if j in mapping:
            raise ValueError("COUNTERFACTUAL_UNMATCHED_OVERLAP")
        mapping[j] = ("N", (int(j),))
    for j in (duplicates or {}):
        if j in mapping:
            raise ValueError("COUNTERFACTUAL_DUPLICATE_ALREADY_OUTPUT")
        mapping[j] = None
    return mapping
