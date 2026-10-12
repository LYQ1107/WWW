"""Frozen-weight question-to-association interventions; no model/source edits."""
import collections
import torch
from jev_recovery_common import *
from jev_recovery_learning import load_records, load_model, batch, comparisons


def fused(actor, original, x, mode):
    d = dict(original)
    if mode == 'zero_auxiliary_logits':
        d['cross_logits'] = torch.zeros_like(d['cross_logits'])
        d['stability_logits'] = torch.zeros_like(d['stability_logits'])
        return d
    if mode in ('full', 'no_joint_plan_residual'):
        return d
    cross, stable = d['cross_features'], d['stability_features']
    if mode == 'cross_representation_only':
        stable = torch.zeros_like(stable)
    if mode == 'persistent_representation_only':
        cross = torch.zeros_like(cross)
    b, q, kp, width = cross.shape
    if mode == 'mean_representations':
        feature = (cross + stable) / 2
    else:
        keys = torch.stack([cross, stable], -2) + actor.tasks.weight[None, None, None]
        query = d['shared'].mean(1)[:, None, None].expand(-1, q, kp, -1)
        feature = actor.norm(actor.plan_reader(
            query.reshape(b*q*kp, 1, width),
            keys.reshape(b*q*kp, 2, width)).reshape(b, q, kp, width))
    d['features'] = feature
    d['edge_logits'] = d['base_logits'] + actor.edge_value(feature).squeeze(-1)
    # Observed auxiliary classifications are retained: intervene downstream
    # of the two auxiliary heads, not by pretending a new model was trained.
    return d


def scores(actor, d, x, pp, mode):
    if mode != 'no_joint_plan_residual':
        return actor.score_plans(d, x, 0, pp)
    q, k = int(x['question_mask'][0].sum()), int(x['identity_mask'][0].sum())
    edge = torch.cat([d['edge_logits'][0, :q, :k], d['edge_logits'][0, :q, -1:]], -1)
    idx = torch.tensor([[k if c < 0 else c for c in p] for p in pp], device=edge.device)
    return edge.double()[torch.arange(q, device=edge.device)[None], idx].sum(-1)


def selected_stats(r, pp, before, after, d, result, examples):
    k = len(r['refs'])
    for row, (u, v) in enumerate(zip(before, after)):
        a, b = k if u < 0 else u, k if v < 0 else v
        ka, kb = bool(r['known_options'][row, a]), bool(r['known_options'][row, b])
        result['rows'] += 1
        result['changed_Global_ID'] += u != v
        if ka and kb:
            result['N%d%d' % (int(r['positive'][row, a]), int(r['positive'][row, b]))] += 1
        else:
            result['UNKNOWN_comparison'] += 1
            result['known_to_UNKNOWN'] += ka and not kb
            result['UNKNOWN_changed_Global_ID'] += u != v
        legal_positive = [c for c in range(k) if bool(r['positive'][row, c])
                          and bool(r['known_options'][row, c]) and bool(r['inputs']['legal'][0,row, c])]
        evaluable_positive = [c for c in legal_positive if any(p[row] == c for p in pp)]
        result['legal_certified_correct_absent_from_plans'] += bool(legal_positive) and not evaluable_positive
        if v < 0 or not kb:
            result['selected_UNKNOWN_or_private_DEFER_for_auxiliary_cross_tab'] += 1
            continue
        cross = int(r['oneweek_labels']['cross'][row, v])
        persistent = int(r['oneweek_labels']['stability'][row, v])
        if cross < 0 or persistent < 0:
            result['auxiliary_UNKNOWN_selected'] += 1
            continue
        ac = bool(d['cross_logits'][0, row, v] >= 0) == bool(cross)
        bc = bool(d['stability_logits'][0, row, v] >= 0) == bool(persistent)
        correct = bool(r['positive'][row, v])
        category = 'both' if ac and bc else 'cross_only' if ac else 'persistent_only' if bc else 'neither'
        result[category + '_aux_correct_final_' + ('correct' if correct else 'wrong')] += 1
        if ac and bc and not correct:
            result['both_correct_final_wrong_with_legal_correct_alternative'] += bool(evaluable_positive)
            result['both_correct_final_wrong_without_legal_correct_alternative'] += not bool(evaluable_positive)
            if len(examples) < 40:
                options = [c for c in range(k) if bool(r['inputs']['legal'][0,row, c])]
                examples.append(dict(key=r['key'], row=row, selected_native_ID=r['refs'][v],
                    legal_correct_alternative=bool(evaluable_positive),
                    plans=[list(p) for p in pp], candidate_native_IDs=r['refs'],
                    candidate_details=[dict(native_ID=r['refs'][c], cross_target=int(r['oneweek_labels']['cross'][row,c]),
                        persistent_target=int(r['oneweek_labels']['stability'][row,c]),
                        known=bool(r['known_options'][row,c]), positive=bool(r['positive'][row,c]),
                        other_camera_history_present=bool(r['inputs']['history_mask'][0,c,3]),
                        first_appearance_present=bool(r['anchor_mask'][c]),
                        first_appearance_cosine=float(r['inputs']['detection_visual'][0,row] @ r['anchor_visual'][c]),
                        base_score=float(d['base_logits'][0,row,c]),
                        corrected_score=float(d['edge_logits'][0,row,c])) for c in options],
                    native_commit='NOT_EXECUTED_IN_FROZEN_STATE_DIAGNOSTIC'))


def main():
    protect(); torch.set_num_threads(1)
    protocol = read(REPORTS/'QUESTION_TO_ASSOCIATION_PROTOCOL.json')
    _, records, sources = load_records(2)
    actor, checkpoint, training = load_model('jev', 20261009, 'pilot', 2, 'XIV')
    counters = {mode: collections.Counter() for mode in protocol['interventions']}
    examples = []
    with torch.no_grad():
        for event, r in enumerate(records):
            x, _ = batch([r]); original = actor.details(x); pp = original['plans'][0]
            if not pp:
                continue
            full_scores = original['plan_scores'][0]; full = pp[int(full_scores.argmax())]
            pairs = comparisons(original, None, [r])
            for mode, count in counters.items():
                d = fused(actor, original, x, mode)
                value = scores(actor, d, x, pp, mode)
                if mode in ('full', 'zero_auxiliary_logits'):
                    assert torch.equal(value, full_scores), 'Frozen full/direct-head parity failed'
                after = pp[int(value.argmax())]
                count['events'] += 1
                count['choice_changed_vs_full'] += after != full
                count['rows_changed_vs_full'] += sum(a != b for a,b in zip(full,after))
                count['auxiliary_logits_changed'] += mode == 'zero_auxiliary_logits'
                for _, a, b, sign, _ in pairs:
                    count['rank_pairs'] += 1
                    count['rank_pair_correct'] += sign * float(value[a]-value[b]) > 0
                # WHO-only rank audit: no Persistent preference in the target.
                k = len(r['refs'])
                for a in range(len(pp)):
                    for b in range(a+1, len(pp)):
                        changed = [(row,k if u<0 else u,k if v<0 else v)
                                   for row,(u,v) in enumerate(zip(pp[a],pp[b])) if u!=v]
                        if not changed or any(not bool(r['known_options'][row,u]) or
                            not bool(r['known_options'][row,v]) for row,u,v in changed): continue
                        difference = sum(int(r['positive'][row,u])-int(r['positive'][row,v]) for row,u,v in changed)
                        if difference:
                            count['WHO_only_rank_pairs'] += 1
                            count['WHO_only_rank_correct'] += difference*float(value[a]-value[b]) > 0
                selected_stats(r, pp, pp[0], after, original, count, examples if mode=='full' else [])
            if event % 128 == 0:
                save(OUT/'question_association/PROGRESS.json',dict(status='RUNNING',events=event,total=len(records)))
                print('QUESTION_ASSOCIATION_EVENT',event,len(records),flush=True)
    result = dict(status='COMPLETE_FROZEN_WEIGHT_TRAIN16_DIAGNOSIS',binding=binding('Addendum A/B/C same checkpoint and plans'),
        checkpoint=checkpoint, protocol=ref(REPORTS/'QUESTION_TO_ASSOCIATION_PROTOCOL.json'), sources=sources,
        actual_training_question_metrics=training.get('final'), events=len(records),
        interventions={m:dict(c,net_certified_corrections=c['N01']-c['N10']) for m,c in counters.items()},
        examples=examples, direct_auxiliary_logits_used_in_final_score=False,
        labels_modified=False, weights_modified=False, new_gradients=0,
        native_mutated_state_online=False, independent_validation=False,
        convention='First past-certified native-ID/person binding; not true birth owner or future safety',
        pending=['Explicit prediction-score learned fusion with matched ordinary controls',
                 'Independent final Readers learned comparison with matched ordinary controls',
                 'Full online intervention attribution and native commit evidence'])
    save(REPORTS/'QUESTION_TO_ASSOCIATION_DIAGNOSIS.json',result)
    print('QUESTION_ASSOCIATION_COMPLETE',flush=True)


if __name__ == '__main__':
    try: main()
    except Exception:
        import traceback
        failure('question_to_association_diagnosis',traceback.format_exc()); raise
