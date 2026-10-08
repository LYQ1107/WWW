"""Explicit current-key causal interventions through the production MATCH hook.

Only runtime references and lawful actions enter this module. Annotation
anchors and future-action maps are deliberately absent from its interface.
"""
from dataclasses import replace
from types import SimpleNamespace

import torch


def apply_native_intervention(model, spec, original, scores, ids, match_i,
                              match_j, threshold, **context):
    key = (int(model._jev_context['video_id']), int(context['frame_index']),
           int(context['view']))
    if tuple(spec['key']) != key:
        raise ValueError('intervention belongs to another native boundary')
    tag, target = spec['tag'], int(spec['row'])
    if not 0 <= target < len(original):
        raise ValueError('intervention row is absent')
    packet = model._jev_candidate_last
    batch = packet['batch']
    expected = tuple(spec['candidate_ids'])
    if tuple(batch.candidate_ids) != expected:
        raise ValueError('frozen runtime candidate support changed')
    first_pairs = {int(r): int(c) for r, c in zip(match_i, match_j)}
    if tag in ('CONTROL', 'KEEP_FACTUAL', 'KEEP_ACCEPT'):
        model._jev_native_intervention_last = {
            'tag': tag, 'native_existing_ids': list(packet['assignment'].existing_ids),
            'legal_current_support_exact': True, 'second_validation': []}
        return original.new_tensor(packet['assignment'].existing_ids)

    submitted = spec.get('pairs')
    pairs = first_pairs if submitted is None else {
        int(r): expected.index(int(t)) for r, t in submitted.items()}
    if len(set(pairs.values())) != len(pairs):
        raise ValueError('intervention violates per-camera identity capacity')
    for r, c in pairs.items():
        if not (0 <= r < len(original) and bool(batch.legal_mask[r, c])):
            raise ValueError('intervention selects an illegal current edge')
    actions = {}
    for r in range(len(original)):
        if r not in pairs:
            actions[r] = 'START_NEW'
        elif submitted is not None and pairs[r] != first_pairs.get(r):
            actions[r] = 'ACCEPT_CURRENT'
        else:
            actions[r] = 'ACCEPT_CURRENT' if int(original[r]) >= 0 else 'START_NEW'
    if tag == 'START_NEW':
        actions[target] = 'START_NEW'
    elif tag == 'REASSOCIATE':
        if target not in first_pairs or len(expected) < 2:
            raise ValueError('REASSOCIATE is not legal at this boundary')
        actions[target] = 'REASSOCIATE'
    elif submitted is None:
        raise ValueError('candidate branch requires an explicit legal assignment')

    validations = []
    class CurrentActionsThenOFFValidation:
        mode = 'jev'
        def decide(self, feature, question, legal, *, off_action=None, context=None):
            if question != 'MATCH_DECISION':
                raise ValueError('current MATCH override leaked into lifecycle')
            context = context or {}
            action = off_action if context.get('proposal_round') == 2 else actions[int(context['detection_index'])]
            if action not in legal:
                raise ValueError('current intervention action is illegal')
            if context.get('proposal_round') == 2:
                validations.append({'row': int(context['detection_index']),
                    'candidate_id': int(context['proposal_track_id']), 'action': action})
            return SimpleNamespace(committed_action=action)

    policy, typed = model.jev_candidate_policy, model.jev_policy
    model.jev_candidate_policy = None
    model.jev_policy = CurrentActionsThenOFFValidation()
    try:
        rows = list(pairs)
        result = model._apply_jev_match_decisions(original, scores, ids, rows,
            [pairs[r] for r in rows], threshold, **context)
    finally:
        model.jev_candidate_policy, model.jev_policy = policy, typed
    references = tuple(int(t) for t in result.tolist())
    assignment = replace(packet['assignment'], existing_ids=references,
        pairs=tuple((r, expected.index(t)) for r, t in enumerate(references) if t >= 0),
        new_rows=tuple(r for r, t in enumerate(references) if t < 0),
        semantic_new=tag == 'START_NEW')
    packet['assignment'] = assignment
    # Ordinary residual unmatched rows retain native bank recovery. Only an
    # explicit START_NEW intervention vetoes recovery of its selected row.
    model._jev_candidate_new_rows = (target,) if tag == 'START_NEW' else ()
    model._jev_native_intervention_last = {'tag': tag,
        'native_existing_ids': list(references), 'submitted_runtime_pairs': submitted,
        'legal_current_support_exact': True, 'second_validation': validations,
        'actual_operator': 'GTRRCNN._apply_jev_match_decisions',
        'future_policy': 'fresh native GMT OFF; lifecycle OFF'}
    return result
